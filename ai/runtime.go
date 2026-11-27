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
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/console"
)

// Option is a setting of App.Open.
type Option func(*options)

type options struct {
	log         io.Writer
	loading     func(Progress)
	compile     bool
	allowOffNPU bool
	settings    map[string]map[string]string
}

// CompileOnLoad lets a Get load a use that is prepared (downloaded and converted: Plan.Run with
// NoCompile, "models download") but not compiled for this machine's NPU yet: its first load
// compiles its programs, which takes from seconds to minutes. It is for a command line tool that
// prepares what it runs just before running it, so that the model is not loaded twice (once to
// compile it, once to run it); an app that must not wait at a Get prepares with Plan.Run.
func CompileOnLoad() Option { return func(o *options) { o.compile = true } }

// AllowOffNPU loads models part of whose programs the NPU's compiler placed off the NPU (on the
// GPU or the CPU): the roles that check where their programs run refuse them otherwise, as that
// has made a Mac sluggish and failed.
func AllowOffNPU() Option { return func(o *options) { o.allowOffNPU = true } }

// LoadSettings are settings of how the models of a role (chat, translate, embed, ...) run
// that change nothing of what is prepared: a role reads the names it knows when it loads a model
// (text's chat: idle, prompt_cache, experts_key, sparse, reserve_moe_memory). What changes the programs is a model
// of its own instead (Definition's Settings). Given twice for a role, the later names win.
func LoadSettings(role string, settings map[string]string) Option {
	return func(o *options) {
		if o.settings == nil {
			o.settings = map[string]map[string]string{}
		}
		m := o.settings[role]
		if m == nil {
			m = map[string]string{}
			o.settings[role] = m
		}
		for k, v := range settings {
			m[k] = v
		}
	}
}

// WithLog sends what loading the models does (the programs, where they run) to w.
func WithLog(w io.Writer) Option { return func(o *options) { o.log = w } }

// Console is where Plan.Log writes as hwnpu's commands do (internal/console): standard error,
// in gray on a terminal, where the downloads' and compiles' bars are redrawn in place.
func Console() io.Writer { return console.Log(os.Stderr) }

// WithLoading tells fn of each model a Get loads, for an app that shows it (a window that would
// otherwise look still): StageLoad as it begins, with how long its load took the last time
// here (Expected, 0 when not known) and the lines of the load's log in Message, then
// StageDone. fn is called in the Get's goroutine, and from the goroutines that redraw a bar.
func WithLoading(fn func(Progress)) Option { return func(o *options) { o.loading = fn } }

// Runtime runs the models an app's store has prepared and compiled (Plan.Run). It touches no
// network, converts and compiles nothing: a use not prepared and compiled is ErrNotPrepared. Each
// model is loaded at its first Get and stays until its Release or Close.
type Runtime struct {
	app *App
	npu airole.NPU
	o   options

	mu     sync.Mutex
	loaded map[string]*loadedModel
	closed bool
	// remote holds the models in hwnpud (nil: in this process)
	remote *remoteRuntime
}

type loadedModel struct {
	// value is the model Get gives, closer what releasing it closes: the model itself, or this
	// Runtime's share of a model hwnpud's Runtimes share (sharedModel)
	value, closer io.Closer
	info          Loaded
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
	c, err := a.through(ctx)
	if err != nil {
		return nil, err
	}
	if c != nil {
		return a.openRemote(ctx, c, opts)
	}
	openMu.Lock()
	err = a.Store.open()
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
		return l.value, nil
	}
	r, ok := rt.app.registered(r) // as registered: with its At and As
	if !ok {
		return nil, fmt.Errorf("ai: %s is not one of %s's models (App.Use registers it)", r.key(), rt.app.Name)
	}
	if rt.remote != nil {
		return rt.remote.get(ctx, rt, r)
	}
	// the model the use is on this machine now: what the store holds prepared as another (the
	// app's line-up changed, or the machine's memory) is not loaded in its place
	m, _, err := resolve(r, rt.app.memory)
	if err != nil {
		return nil, err
	}
	if err := rt.app.allowed(m); err != nil {
		return nil, err
	}
	// a use is run only when it is prepared and compiled for this program on this NPU (Plan.Run,
	// "models prepare"): no picture, no input waits for a download or a compile
	loc, err := rt.app.Store.find(r.key(), rt.npu, m, r.role.Revision, r.role.Usable)
	if err == nil && !loc.bundle && rt.npu.Target != "" && !rt.npu.CPU(m) && !loc.compiled() && !rt.o.compile {
		err = fmt.Errorf("%w: %s (%s) is not compiled for %s on this machine's NPU yet (the app's Plan and its Run compile it)", ErrNotPrepared, r.key(), loc.Model, processName())
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
	// hwnpud's Runtimes share one copy of a model loaded alike (the same model, files and
	// settings): one already loaded is this Runtime's too
	key := ""
	if rt.app.local {
		key = shareKey(r, loc, rt)
		if v, c, ok := shareHit(key, rt.o.log); ok {
			rt.loaded[r.key()] = &loadedModel{value: v, closer: c, info: Loaded{Use: r.name, Role: r.role.Name, Model: loc.Model, MemoryGB: loc.MemoryGB, At: t0}}
			return v, nil
		}
	}
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
	var share *relay
	if key != "" {
		// the shared model's log is that of a Runtime that holds it, the latest
		share = &relay{}
		share.attach(log)
		log = share
	}
	l, err := r.role.Load(chat.WithModel(ctx, m), loc.Files, airole.LoadEnv{Model: loc.Model, Def: m, NPU: rt.npu, Cache: rt.app.Store.cacheDir(), Log: log, AllowOffNPU: rt.o.allowOffNPU, Settings: rt.o.settings[r.role.Name]})
	if err != nil {
		loaded.Store(true)
		tell(StageDone, "")
		return nil, fmt.Errorf("ai: loading %s (%s): %w", r.name, loc.Model, err)
	}
	took := time.Since(t0)
	loaded.Store(true)
	tell(StageDone, "")
	if !loc.bundle && rt.npu.Target != "" && !loc.compiled() {
		// compiled by this load (CompileOnLoad): the next one need not
		loc.markCompiled()
		loc.Compile = took
		_ = rt.app.Store.put(ctx, r.key(), rt.npu, loc.record)
	} else if !loc.bundle && (loc.Loaded == 0 || took > loc.Loaded*5/4 || took < loc.Loaded*4/5) {
		// how long it takes, for the next load to show (written when it changed by a fifth)
		loc.Loaded = took
		_ = rt.app.Store.put(ctx, r.key(), rt.npu, loc.record)
	}
	lm := &loadedModel{value: l, closer: l, info: Loaded{Use: r.name, Role: r.role.Name, Model: loc.Model, MemoryGB: loc.MemoryGB, At: t0, Took: time.Since(t0)}}
	if key != "" {
		var paths []string
		loc.each(func(p *string) { paths = append(paths, *p) })
		lm.value, lm.closer = shareAdd(key, r.role.Name, loc.Model, paths, l, share)
	}
	rt.loaded[r.key()] = lm
	return lm.value, nil
}

// release unloads r, if it is loaded.
func (rt *Runtime) release(r requirement) error {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	if rt.remote != nil {
		return rt.remote.release(rt, r)
	}
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
	if rt.remote != nil {
		err := rt.remote.close(rt)
		rt.loaded = nil
		return err
	}
	var errs []error
	for _, l := range rt.loaded {
		errs = append(errs, l.closer.Close())
	}
	rt.loaded = nil
	return errors.Join(errs...)
}
