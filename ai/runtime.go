package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"sync"
	"sync/atomic"
	"time"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
)

// Option is a setting of App.Open.
type Option func(*options)

type options struct {
	log     io.Writer
	auto    bool
	autoFn  func(Progress)
	license bool
	prepLog io.Writer // Plan.Log of the preparing (and of its compile)
	loading func(Progress)
}

// WithLog sends what loading the models does (the programs, where they run) to w.
func WithLog(w io.Writer) Option { return func(o *options) { o.log = w } }

// WithAutoPrepare makes the Runtime prepare a use that is not ready when it is first asked for
// (Plan and Run for that use alone, reported to fn, nil for nowhere), as hwnpu's commands do on
// their first run: a Get may then download gigabytes and take minutes. Without it the Runtime
// touches no network and converts nothing.
func WithAutoPrepare(fn func(Progress)) Option {
	return func(o *options) { o.auto, o.autoFn = true, fn }
}

// WithConsole is WithAutoPrepare as hwnpu's commands show it on the terminal (standard error):
// a line for each model made ready, its downloads, conversions and the compile at its first
// load with their bars redrawn in place, and what they log in gray (Console). An app's commands
// open their Runtime with it so that their first run makes the models ready, as "models
// prepare" (cli.Prepare) does ahead.
func WithConsole() Option {
	return func(o *options) {
		o.auto, o.prepLog = true, Console()
		o.autoFn = func(p Progress) {
			switch p.Stage {
			case StagePrepare:
				if p.Message == "" {
					fmt.Fprintf(os.Stderr, "%s (%s): not here yet, downloading and converting it\n", p.Use, p.Model)
				}
			case StageDone:
				fmt.Fprintf(os.Stderr, "%s (%s): ready\n", p.Use, p.Model)
			}
		}
	}
}

// Console is where Plan.Log writes as hwnpu's commands do (internal/console): standard error,
// in gray on a terminal, where the downloads' and compiles' bars are redrawn in place.
func Console() io.Writer { return console.Log(os.Stderr) }

// WithLoading tells fn of each model a Get makes ready, as it goes, for an app that shows it
// (a window that would otherwise look stuck): its preparing when WithAutoPrepare does it
// (StagePrepare, with the lines of its downloads and conversions in Message), then its load
// (StageCompile at its first load here, which compiles its programs for the NPU, else
// StageLoad), with how long that took the last time here (Expected) and the lines of the
// load's log, then StageDone. fn is called in the Get's goroutine, and from the goroutines
// that redraw a bar.
func WithLoading(fn func(Progress)) Option { return func(o *options) { o.loading = fn } }

// WithAcceptedLicenses lets WithAutoPrepare accept the licenses that ask for it
// (Plan.AcceptLicenses), for an app that has the user's agreement.
func WithAcceptedLicenses() Option { return func(o *options) { o.license = true } }

// Runtime runs the models an app's store has prepared. It touches no network and converts
// nothing (unless WithAutoPrepare): a use the store has not prepared is ErrNotPrepared. Each
// model is loaded at its first Get and stays until its Release or Close.
type Runtime struct {
	app *App
	npu airole.NPU
	o   options

	mu     sync.Mutex
	loaded map[string]*loadedModel
	closed bool
}

type loadedModel struct {
	closer io.Closer
	info   Loaded
}

// Loaded is a model a Runtime holds.
type Loaded struct {
	// Use is the name the model was picked by and Role the role it fills.
	Use, Role string
	// Model is the model's name in the definitions.
	Model string
	// MemoryGB is what the model takes running, from the definitions (0: not known).
	MemoryGB float64
	// At is when it was loaded and Took how long loading it took.
	At   time.Time
	Took time.Duration
}

// Open opens the app's models. A prepared cache or a store prepared for another NPU's programs
// alone is ErrBundleTarget when one of its uses is asked for. It sets the process's layers and
// reads the definitions as the app's Store says (modelstore.Open): a use is the model the
// definitions and the app's line-up (Use.At, Use.As) pick for this machine, and a store that
// holds it prepared as another is ErrNotPrepared.
func (a *App) Open(ctx context.Context, opts ...Option) (*Runtime, error) {
	openMu.Lock()
	err := a.Store.open()
	openMu.Unlock()
	if err != nil {
		return nil, err
	}
	rt := &Runtime{app: a, npu: airole.Here(ctx), loaded: map[string]*loadedModel{}}
	for _, o := range opts {
		o(&rt.o)
	}
	return rt, nil
}

// Target names this machine's NPU as the shared layers name it: coreai, intel or amd-xdna2; ""
// when no NPU runs the models here.
func (rt *Runtime) Target() string { return rt.npu.Target }

// Loaded are the models the Runtime holds now, in the order they were loaded.
func (rt *Runtime) Loaded() []Loaded {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	out := make([]Loaded, 0, len(rt.loaded))
	for _, l := range rt.loaded {
		out = append(out, l.info)
	}
	slices.SortFunc(out, func(a, b Loaded) int { return a.At.Compare(b.At) })
	return out
}

// get loads r at its first call. Loads run one at a time.
func (rt *Runtime) get(ctx context.Context, r requirement) (io.Closer, error) {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return nil, errors.New("ai: the runtime is closed")
	}
	if l, ok := rt.loaded[r.key()]; ok {
		return l.closer, nil
	}
	r, ok := rt.app.registered(r) // as registered: with its At and As
	if !ok {
		return nil, fmt.Errorf("ai: %s is not one of %s's models (App.Use registers it)", r.key(), rt.app.Name)
	}
	// the model the use is on this machine now: what the store holds prepared as another (the
	// app's line-up changed, or the machine's memory) is not loaded in its place
	m, _, err := resolve(r, rt.app.memory)
	if err != nil {
		return nil, err
	}
	// a use is run only when it is prepared and compiled for this program on this NPU: no
	// picture, no input waits for a compile. WithAutoPrepare does what "models prepare" does
	// first (Plan.Run: the download, the conversion, the load once that compiles every program)
	find := func() (located, error) {
		loc, err := rt.app.Store.find(r.key(), rt.npu, m, r.role.Revision)
		if err == nil && !loc.bundle && rt.npu.Target != "" && !loc.compiled() {
			err = fmt.Errorf("%w: %s (%s) is not compiled for %s on this machine's NPU yet (the app's Plan and its Run compile it)", ErrNotPrepared, r.key(), loc.Model, processName())
		}
		return loc, err
	}
	loc, err := find()
	if errors.Is(err, ErrNotPrepared) && rt.o.auto {
		if err = rt.prepare(ctx, r); err == nil {
			loc, err = find()
		}
	}
	if err != nil {
		return nil, err
	}
	// the others of its group of InTurns are released first: the app said it never needs them
	// together, and the plan counted the largest of them alone
	for _, k := range rt.app.turnOf(r.key()) {
		if l, ok := rt.loaded[k]; ok && k != r.key() {
			delete(rt.loaded, k)
			if err := l.closer.Close(); err != nil {
				return nil, fmt.Errorf("ai: releasing %s for %s: %w", k, r.key(), err)
			}
		}
	}
	if !loc.bundle {
		// used: kept by this machine's cache while the process runs, and last in its LRU
		loc.each(func(p *string) { cache.Touch(*p) })
	}
	t0 := time.Now()
	log := rt.o.log // compiled (find): the load takes its programs from the compiler's cache
	tell := func(Stage, string) {}
	var loaded atomic.Bool // its load is over: the lines of its log after are not told
	if fn := rt.o.loading; fn != nil {
		stage, expected := StageLoad, loc.Loaded
		tell = func(st Stage, msg string) {
			fn(Progress{Use: r.name, Role: r.role.Name, Model: loc.Model, Stage: st, Item: 1, Items: 1, Message: msg, Expected: expected, Since: t0})
		}
		// the model keeps the log after its load (a later size's programs, its steps): those
		// lines are not of the load, which has told it is done
		lw := &lineWriter{fn: func(line string) {
			if !loaded.Load() {
				tell(stage, line)
			}
		}}
		if log == nil {
			log = lw
		} else {
			log = io.MultiWriter(log, lw)
		}
		tell(stage, "")
	}
	l, err := r.role.Load(ctx, loc.Files, airole.LoadEnv{Model: loc.Model, NPU: rt.npu, Cache: rt.app.Store.cacheDir(), Log: log})
	if err != nil {
		loaded.Store(true)
		tell(StageDone, "")
		return nil, fmt.Errorf("ai: loading %s (%s): %w", r.name, loc.Model, err)
	}
	took := time.Since(t0)
	loaded.Store(true)
	tell(StageDone, "")
	if !loc.bundle && (loc.Loaded == 0 || took > loc.Loaded*5/4 || took < loc.Loaded*4/5) {
		// how long it takes, for the next load to show (written when it changed by a fifth)
		loc.Loaded = took
		_ = rt.app.Store.put(ctx, r.key(), rt.npu, loc.record)
	}
	rt.loaded[r.key()] = &loadedModel{closer: l, info: Loaded{Use: r.name, Role: r.role.Name, Model: loc.Model, MemoryGB: loc.MemoryGB, At: t0, Took: time.Since(t0)}}
	return l, nil
}

// prepare makes r ready (WithAutoPrepare) as "models prepare" does: downloaded, converted, and
// loaded once (warmed, ai.Warmer), which compiles its programs for this program on the NPU.
func (rt *Runtime) prepare(ctx context.Context, r requirement) error {
	p, err := rt.app.plan(ctx, []requirement{r})
	if err != nil {
		return err
	}
	p.AcceptLicenses, p.Log = rt.o.license, rt.o.prepLog
	fn := rt.o.autoFn
	if watch := rt.o.loading; watch != nil {
		t0, auto := time.Now(), fn
		var mu sync.Mutex
		stage := StagePrepare // the stage the log's lines are of
		fn = func(pr Progress) {
			if auto != nil {
				auto(pr)
			}
			mu.Lock()
			if pr.Stage != stage && pr.Stage != StageDone {
				stage, t0 = pr.Stage, time.Now() // the compile's time from its own start
			}
			pr.Since = t0
			mu.Unlock()
			if pr.Stage != StageDone { // the load that follows tells when it is done
				watch(pr)
			}
		}
		if p.Log != nil { // else Run sends the lines of its log to fn itself
			model := ""
			if len(p.Items) > 0 {
				model = p.Items[0].Model
			}
			p.Log = io.MultiWriter(p.Log, &lineWriter{fn: func(line string) {
				mu.Lock()
				st, since := stage, t0
				mu.Unlock()
				watch(Progress{Use: r.name, Role: r.role.Name, Model: model, Stage: st, Item: 1, Items: 1, Message: line, Since: since})
			}})
		}
	}
	return p.Run(ctx, fn)
}

// release unloads r, if it is loaded.
func (rt *Runtime) release(r requirement) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	l, ok := rt.loaded[r.key()]
	if !ok {
		return nil
	}
	delete(rt.loaded, r.key())
	return l.closer.Close()
}

// Close unloads the models.
func (rt *Runtime) Close() error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.closed {
		return nil
	}
	rt.closed = true
	var errs []error
	for _, l := range rt.loaded {
		errs = append(errs, l.closer.Close())
	}
	rt.loaded = nil
	return errors.Join(errs...)
}
