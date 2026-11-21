package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"sync"
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
	prepLog io.Writer // Plan.Log of the preparing, and the log of a model's first load
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
	loc, err := rt.app.Store.find(r.key(), rt.npu, m)
	if errors.Is(err, ErrNotPrepared) && rt.o.auto {
		if err = rt.prepare(ctx, r); err == nil {
			loc, err = rt.app.Store.find(r.key(), rt.npu, m)
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
	log := rt.o.log
	if log == nil && rt.o.prepLog != nil && !loc.bundle && loc.Compiled.IsZero() && rt.npu.Target != "" {
		log = rt.o.prepLog // its first load, which compiles it (minutes): shown as it goes
	}
	l, err := r.role.Load(ctx, loc.Files, airole.LoadEnv{Model: loc.Model, NPU: rt.npu, Cache: rt.app.Store.cacheDir(), Log: log})
	if err != nil {
		return nil, fmt.Errorf("ai: loading %s (%s): %w", r.name, loc.Model, err)
	}
	if !loc.bundle && loc.Compiled.IsZero() && rt.npu.Target != "" {
		// loaded here for the first time (WithAutoPrepare, or prepared without compiling): its
		// programs are compiled now. Not writing it only has "models prepare" load it once more.
		loc.Compiled = time.Now().UTC().Truncate(time.Second)
		_ = rt.app.Store.put(ctx, r.key(), rt.npu, loc.record)
	}
	rt.loaded[r.key()] = &loadedModel{closer: l, info: Loaded{Use: r.name, Role: r.role.Name, Model: loc.Model, MemoryGB: loc.MemoryGB, At: t0, Took: time.Since(t0)}}
	return l, nil
}

// prepare makes r ready (WithAutoPrepare) without loading it: the load that follows compiles it.
func (rt *Runtime) prepare(ctx context.Context, r requirement) error {
	p, err := rt.app.plan(ctx, []requirement{r})
	if err != nil {
		return err
	}
	p.NoCompile, p.AcceptLicenses, p.Log = true, rt.o.license, rt.o.prepLog
	return p.Run(ctx, rt.o.autoFn)
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
