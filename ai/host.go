package ai

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
)

// What hwnpud's engine (internal/aihost) does with the app a client sends: it makes the app
// again (AppFromSpec), whose models run in the engine's process, plans and runs it, opens it and
// gets its uses by their keys. These are for hwnpud; an app uses App, Plan and Runtime.

// AppFromSpec is the app a client sent (App.Spec), its roles those this process registered
// (airole.Lookup), its models run here whatever HWNPU_DAEMON says. The client's model policy
// narrows this process's (chat.AllowedAlso).
func AppFromSpec(s AppSpec) (*App, error) {
	a := &App{Name: s.Name, Store: s.Store.store(), memory: s.Memory, turns: s.Turns, local: true, allowlist: s.Allowlist}
	for _, u := range s.Uses {
		role := airole.Lookup(u.Role)
		if role == nil {
			return nil, fmt.Errorf("ai: hwnpud knows no role %q (%s's use %s): hwnpud is of an older build than the app (hwnpud stop, go install ./cmd/hwnpud)", u.Role, s.Name, u.Name)
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
func (a *App) allowed(m chat.Model) error {
	if strings.TrimSpace(a.allowlist) == "" {
		return nil
	}
	return chat.AllowedAlso(m, a.allowlist)
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
