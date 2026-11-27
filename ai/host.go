package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"slices"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/internal/aieng"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/catalog"
)

// What hwnpud's engine (internal/aihost) does with the app a client sends: it makes the app
// again (AppFromSpec), whose models run in the engine's process, plans and runs it, opens it and
// gets its uses by their keys. These are for hwnpud; an app uses App, Plan and Runtime.

// AppFromSpec is the app a client sent (App.Spec), its roles those this process registered
// (airole.Lookup), its models run here whatever HWNPU_DAEMON says. The client's model policy
// narrows this process's (the engine's AllowedAlso).
func AppFromSpec(s AppSpec) (*App, error) {
	a := &App{Name: s.Name, Store: s.Store.store(), memory: s.Memory, turns: s.Turns, local: true, allowlist: s.Allowlist}
	for _, u := range s.Uses {
		role := airole.Lookup(u.Role)
		if role == nil {
			return nil, fmt.Errorf("ai: hwnpud knows no role %q (%s's use %s): hwnpud is of an older build than the app (go install ./cmd/hwnpud, then hwnpu service restart)", u.Role, s.Name, u.Name)
		}
		r := requirement{name: u.Name, role: role, as: u.As}
		for _, t := range u.Tiers {
			r.tiers = append(r.tiers, tier{t.GB, t.Name})
		}
		a.reqs = append(a.reqs, r)
	}
	return a, nil
}

// allowed is nil when the policy, narrowed by the client's of an app hwnpud made, allows m.
func (a *App) allowed(m catalog.Model) error {
	if strings.TrimSpace(a.allowlist) == "" {
		return nil
	}
	return aieng.Must().AllowedAlso(m, a.allowlist)
}

// Spec is the plan as hwnpud sends it to its client.
func (p *Plan) Spec() PlanSpec {
	return PlanSpec{Items: p.Items, Target: p.npu.Target, Machine: p.machine, Licenses: p.Licenses()}
}

// RunTo is Run with Quiet, the models' logs written to log as they come: hwnpud's run of a
// client's plan, whose client shows them.
func (p *Plan) RunTo(ctx context.Context, log io.Writer, fn func(Progress)) error {
	p.Quiet, p.logTo = true, log
	return p.Run(ctx, fn)
}

// GetKey is the model of the app's use of key (ROLE:NAME, as AppSpec's uses name them), loaded
// at its first call as Use.Get loads it, and what Loaded says of it.
func (rt *Runtime) GetKey(ctx context.Context, key string) (io.Closer, Loaded, error) {
	for _, r := range rt.app.requirements() {
		if r.key() != key {
			continue
		}
		l, err := rt.get(ctx, r)
		if err != nil {
			return nil, Loaded{}, err
		}
		rt.mu.Lock()
		defer rt.mu.Unlock()
		if lm, ok := rt.loaded[key]; ok {
			return l, lm.info, nil
		}
		return l, Loaded{}, nil
	}
	return nil, Loaded{}, fmt.Errorf("ai: %s is not one of %s's models", key, rt.app.Name)
}

// ReleaseKey is Use.Release of the app's use of key.
func (rt *Runtime) ReleaseKey(key string) error {
	for _, r := range rt.app.requirements() {
		if r.key() == key {
			return rt.release(r)
		}
	}
	return nil
}

// LoadedKeys are the keys of the uses the Runtime holds now.
func (rt *Runtime) LoadedKeys() []string {
	rt.mu.Lock()
	defer rt.mu.Unlock()
	var out []string
	for k := range rt.loaded {
		out = append(out, k)
	}
	return out
}

// sharing are the models hwnpud's Runtimes (of apps AppFromSpec made) hold, one copy of each: two
// clients that use the same model, loaded alike, load it once. Its calls take turns as its
// type makes them (each runs one request at a time); it is unloaded when the last Runtime
// releases it.
var sharing = struct {
	sync.Mutex
	m map[string]*sharedModel
}{m: map[string]*sharedModel{}}

// sharedModel is a model of sharing: what it is, the Runtimes that hold it and its files.
type sharedModel struct {
	key, role, model string
	value            io.Closer
	refs             int
	paths            []string
	log              *relay
}

// shareKey is what makes two loads of a use the same model: its role, its model and files, how
// it is run (LoadSettings, AllowOffNPU) and the NPU.
func shareKey(r requirement, loc located, rt *Runtime) string {
	files, _ := json.Marshal(loc.Files)
	set, _ := json.Marshal(rt.o.settings[r.role.Name])
	return strings.Join([]string{r.role.Name, loc.Model, string(files), string(set), fmt.Sprint(rt.o.allowOffNPU), rt.npu.Target}, "\x00")
}

// shareHit is the model of key when one is loaded (in sharing), and this Runtime's share of it (log its log).
func shareHit(key string, log io.Writer) (io.Closer, io.Closer, bool) {
	sharing.Lock()
	defer sharing.Unlock()
	s, ok := sharing.m[key]
	if !ok {
		return nil, nil, false
	}
	s.refs++
	return s.value, &shareRef{s: s, detach: s.log.attach(log)}, true
}

// shareAdd keeps a model just loaded as the model of key and returns it and this Runtime's share
// (log the relay it was loaded with). One loaded meanwhile by another Runtime is kept instead.
func shareAdd(key, role, model string, paths []string, m io.Closer, log *relay) (io.Closer, io.Closer) {
	sharing.Lock()
	s, ok := sharing.m[key]
	if ok {
		s.refs++
		sharing.Unlock()
		m.Close()
		return s.value, &shareRef{s: s, detach: s.log.attach(nil)}
	}
	s = &sharedModel{key: key, role: role, model: model, value: m, refs: 1, paths: paths, log: log}
	sharing.m[key] = s
	sharing.Unlock()
	return m, &shareRef{s: s, detach: func() { log.detach(0) }}
}

// shareRef is a Runtime's share of a sharedModel: closing it unloads the model when it was the
// last.
type shareRef struct {
	s      *sharedModel
	detach func()
	once   sync.Once
}

func (h *shareRef) Close() error {
	var err error
	h.once.Do(func() {
		h.detach()
		sharing.Lock()
		h.s.refs--
		last := h.s.refs == 0
		if last {
			delete(sharing.m, h.s.key)
		}
		sharing.Unlock()
		if last {
			err = h.s.value.Close()
		}
	})
	return err
}

// relay is a shared model's log: what it writes goes to the latest Runtime's log that holds it.
type relay struct {
	mu   sync.Mutex
	next int
	ws   []relayed
}

type relayed struct {
	id int
	w  io.Writer
}

// attach makes w the log (nil: none for this Runtime) until detach.
func (r *relay) attach(w io.Writer) func() {
	r.mu.Lock()
	defer r.mu.Unlock()
	id := r.next
	r.next++
	r.ws = append(r.ws, relayed{id, w})
	return func() { r.detach(id) }
}

func (r *relay) detach(id int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ws = slices.DeleteFunc(r.ws, func(x relayed) bool { return x.id == id })
}

func (r *relay) Write(p []byte) (int, error) {
	r.mu.Lock()
	var w io.Writer
	for i := len(r.ws) - 1; i >= 0 && w == nil; i-- {
		w = r.ws[i].w
	}
	r.mu.Unlock()
	if w != nil {
		w.Write(p)
	}
	return len(p), nil
}

// Terminal says whether the log written to now redraws its bars (console.Live).
func (r *relay) Terminal() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	for i := len(r.ws) - 1; i >= 0; i-- {
		if w := r.ws[i].w; w != nil {
			t, ok := w.(interface{ Terminal() bool })
			return ok && t.Terminal()
		}
	}
	return false
}

// HeldModel is a model hwnpud's Runtimes hold (Held).
type HeldModel struct {
	Role, Model string
	// Uses are the Runtimes that hold it; Paths the files it was loaded from.
	Uses  int
	Paths []string
}

// Held are the models hwnpud's Runtimes hold now, one copy each: for its status, and so that
// what removes the cache's units leaves the files of the models loaded.
func Held() []HeldModel {
	sharing.Lock()
	defer sharing.Unlock()
	var out []HeldModel
	for _, s := range sharing.m {
		out = append(out, HeldModel{Role: s.role, Model: s.model, Uses: s.refs, Paths: slices.Clone(s.paths)})
	}
	slices.SortFunc(out, func(a, b HeldModel) int { return strings.Compare(a.Model, b.Model) })
	return out
}
