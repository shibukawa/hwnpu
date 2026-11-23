// Package aihost is hwnpud's side of the ai package (docs/design/hwnpud.md): it serves the
// sessions an app opens with hwnpud's engine (ai.Host over internal/airpc) and runs the app's
// plan, its preparing and compiling, and its models in the engine's process, so that Apple's
// compiler keeps one cache (coreai-cache/<OS>/hwnpud/) for every program.
//
// It imports the packages of models, whose roles (airole.Register) the apps' uses name; the
// engine's package (internal/daemon) does not, and cmd/hwnpud joins the two (daemon.Engine.AI).
package aihost

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/daemon"

	// the packages of models, for their roles
	_ "github.com/shibukawa/hwnpu/ai/imagegen"
	_ "github.com/shibukawa/hwnpu/ai/sound"
	_ "github.com/shibukawa/hwnpu/ai/speech"
	_ "github.com/shibukawa/hwnpu/ai/text"
	_ "github.com/shibukawa/hwnpu/ai/vision"
)

// Handler serves the sessions of the ai package (daemon.AIPath, upgraded to daemon.AIProtocol),
// logging what they do to log.
func Handler(log io.Writer) http.Handler {
	return airpc.Handler(daemon.AIProtocol, func(s *airpc.Session) any {
		h := &Host{s: s, log: log, handles: map[string]uint64{}}
		s.OnClose(h.closed)
		return h
	})
}

// loading keeps the loads and the runs of every session one at a time, as the engine's chat
// models load: the NPU's compiler and its memory take one model after another.
var loading sync.Mutex

// Host is a session's ai.Host: a client's app, planned and run, and one Runtime of it.
type Host struct {
	s   *airpc.Session
	log io.Writer

	mu      sync.Mutex
	app     string
	rt      *ai.Runtime
	handles map[string]uint64 // the models' objects of the session, by their uses' keys
}

var _ ai.Host = (*Host)(nil)

func (h *Host) logf(format string, args ...any) {
	if h.log != nil {
		fmt.Fprintf(h.log, "hwnpud engine: "+format+"\n", args...)
	}
}

// Plan plans the app for this machine (App.Plan).
func (h *Host) Plan(ctx context.Context, spec ai.AppSpec) (ai.PlanSpec, error) {
	app, err := ai.AppFromSpec(spec)
	if err != nil {
		return ai.PlanSpec{}, err
	}
	p, err := app.Plan(ctx)
	if err != nil {
		return ai.PlanSpec{}, err
	}
	return p.Spec(), nil
}

// Run plans the app and makes its uses ready (Plan.Run), its stages told to fn and the models'
// logs written to log; it returns the uses as they are after it.
func (h *Host) Run(ctx context.Context, spec ai.AppSpec, o ai.RunSpec, log func([]byte), fn func(ai.Progress)) ([]ai.Item, error) {
	app, err := ai.AppFromSpec(spec)
	if err != nil {
		return nil, err
	}
	loading.Lock()
	defer loading.Unlock()
	p, err := app.Plan(ctx)
	if err != nil {
		return nil, err
	}
	p.NoCompile, p.AcceptLicenses, p.AllowOffNPU = o.NoCompile, o.AcceptLicenses, o.AllowOffNPU
	if !p.Compiled() {
		h.logf("preparing %s's models", spec.Name)
	}
	err = p.RunTo(ctx, writer(log, o.Live), fn)
	return p.Items, err
}

// Open opens the app's Runtime of the session (App.Open). Its log and its loading's function
// stay the client's for the session.
func (h *Host) Open(ctx context.Context, spec ai.AppSpec, o ai.OpenSpec, log func([]byte), loading func(ai.Progress)) (string, error) {
	app, err := ai.AppFromSpec(spec)
	if err != nil {
		return "", err
	}
	var opts []ai.Option
	if o.Compile {
		opts = append(opts, ai.CompileOnLoad())
	}
	if o.AllowOffNPU {
		opts = append(opts, ai.AllowOffNPU())
	}
	for role, set := range o.Settings {
		opts = append(opts, ai.LoadSettings(role, set))
	}
	if log != nil {
		opts = append(opts, ai.WithLog(writer(log, o.Live)))
	}
	if loading != nil {
		opts = append(opts, ai.WithLoading(loading))
	}
	rt, err := app.Open(ctx, opts...)
	if err != nil {
		return "", err
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.rt != nil {
		rt.Close()
		return "", errors.New("aihost: a session opens one Runtime")
	}
	h.rt, h.app = rt, spec.Name
	return rt.Target(), nil
}

func (h *Host) runtime() (*ai.Runtime, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.rt == nil {
		return nil, errors.New("aihost: the session has no Runtime (Open)")
	}
	return h.rt, nil
}

// Get loads the use of key (Use.Get) and returns the session's object of its model.
func (h *Host) Get(ctx context.Context, key string) (ai.GotSpec, error) {
	rt, err := h.runtime()
	if err != nil {
		return ai.GotSpec{}, err
	}
	// the load goes on until it ends or the client cancels it, not only for the call: a model
	// may keep the context it was loaded with
	lctx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	stop := context.AfterFunc(ctx, cancel)
	loading.Lock()
	m, info, err := rt.GetKey(lctx, key)
	loading.Unlock()
	if !stop() && err != nil {
		cancel()
	}
	if err != nil {
		return ai.GotSpec{}, err
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	// the others of its group of InTurns were released by the load
	held := map[string]bool{}
	for _, k := range rt.LoadedKeys() {
		held[k] = true
	}
	for k, id := range h.handles {
		if !held[k] {
			h.s.Forget(id)
			delete(h.handles, k)
		}
	}
	id, ok := h.handles[key]
	if !ok {
		id = h.s.Register(m)
		h.handles[key] = id
		h.logf("%s: holds %s (%s) for %s", key, info.Model, info.Took.Round(1e6), h.app)
	}
	return ai.GotSpec{Handle: id, Loaded: info}, nil
}

// Release unloads the use of key (Use.Release).
func (h *Host) Release(ctx context.Context, key string) error {
	rt, err := h.runtime()
	if err != nil {
		return err
	}
	h.mu.Lock()
	if id, ok := h.handles[key]; ok {
		h.s.Forget(id)
		delete(h.handles, key)
	}
	h.mu.Unlock()
	return rt.ReleaseKey(key)
}

// CloseRuntime unloads the session's models (Runtime.Close).
func (h *Host) CloseRuntime(ctx context.Context) error {
	h.mu.Lock()
	rt := h.rt
	h.rt = nil
	for k, id := range h.handles {
		h.s.Forget(id)
		delete(h.handles, k)
	}
	h.mu.Unlock()
	if rt == nil {
		return nil
	}
	return rt.Close()
}

// closed unloads the models of a session that ended (the client exited or closed its Runtime).
func (h *Host) closed() {
	if err := h.CloseRuntime(context.Background()); err != nil {
		h.logf("closing %s's models: %v", h.app, err)
	}
}

// writer is the client's log as a writer: Terminal says whether it redraws its bars
// (console.Live), nil for none.
func writer(fn func([]byte), live bool) io.Writer {
	if fn == nil {
		return nil
	}
	return &clientLog{fn: fn, live: live}
}

type clientLog struct {
	fn   func([]byte)
	live bool
}

func (c *clientLog) Write(p []byte) (int, error) {
	c.fn(append([]byte(nil), p...))
	return len(p), nil
}

func (c *clientLog) Terminal() bool { return c.live }
