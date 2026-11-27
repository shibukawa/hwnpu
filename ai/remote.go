package ai

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/daemon"
	"github.com/shibukawa/hwnpu/modelstore"
)

// An app's models run in hwnpud (docs/design/hwnpud.md) unless HWNPU_DAEMON is off: Plan, its
// Run and Open send the app (AppSpec) over a session with hwnpud's engine (internal/aihost),
// which plans, prepares, compiles and loads the models in its own process, so that Apple's
// compiler keeps one cache for every program (coreai-cache/<OS>/hwnpud/) and a model one
// program prepared is compiled for the next. A Use's Get gives a model of the app's own type
// whose methods call the one hwnpud loaded (airole.Role.Remote; every role of hwnpu's packages
// has one). The app's code is the same
// either way. A test (go test) and hwnpud's engine itself run the models in their own process.

// Host is what hwnpud's engine serves on a session of the ai package (internal/aihost): an app's
// plan and its run, and one Runtime with its models. It is for hwnpud; an app uses App, Plan and
// Runtime.
type Host interface {
	Plan(ctx context.Context, app AppSpec) (PlanSpec, error)
	Run(ctx context.Context, app AppSpec, o RunSpec, log func([]byte), fn func(Progress)) ([]Item, error)
	Open(ctx context.Context, app AppSpec, o OpenSpec, log func([]byte), loading func(Progress)) (string, error)
	Get(ctx context.Context, key string) (GotSpec, error)
	Release(ctx context.Context, key string) error
	CloseRuntime(ctx context.Context) error
}

// AppSpec is an app as it is sent to hwnpud: its uses, their groups of InTurns, its memory and
// its store.
type AppSpec struct {
	Name   string
	Uses   []UseSpec
	Turns  [][]string
	Memory int
	Store  StoreSpec
	// Allowlist is the client's model policy (its HWNPU_MODEL_ALLOWLIST): it narrows hwnpud's.
	Allowlist string
}

// UseSpec is a use of an AppSpec: its role's name, the name it is picked by, Use.As and Use.At.
type UseSpec struct {
	Role, Name string
	As         int
	Tiers      []TierSpec
}

// TierSpec is a Use.At.
type TierSpec struct {
	GB   int
	Name string
}

// StoreSpec is an app's Store as it is sent: what is unset (the environment's sources, Hugging
// Face, the limits) stays unset, for hwnpud's settings. The paths are the client's, which are
// hwnpud's too (the same user on the same machine).
type StoreSpec struct {
	Root           string
	Sources        []string
	SourcesSet     bool
	HuggingFace    bool
	HuggingFaceSet bool
	Limits         modelstore.Limits
	LimitsSet      bool
	Files          []string
	Definitions    []modelstore.Definitions
	Prepared       []string
	AppDefinitions []Definition
}

// PlanSpec is a Plan as hwnpud sends it back.
type PlanSpec struct {
	Items    []Item
	Target   string
	Machine  int
	Licenses []License
}

// RunSpec is how hwnpud runs a Plan (its fields of the same names); Live says that the client's
// log redraws its bars.
type RunSpec struct {
	NoCompile, AcceptLicenses, AllowOffNPU bool
	Live                                   bool
}

// OpenSpec are an Open's options as hwnpud takes them (CompileOnLoad, AllowOffNPU,
// LoadSettings); Live says that the client's log (WithLog) redraws its bars.
type OpenSpec struct {
	Compile, AllowOffNPU bool
	Settings             map[string]map[string]string
	Live                 bool
}

// GotSpec is a model hwnpud loaded for a Get: the object of the session that is the model, and
// what Runtime.Loaded says of it.
type GotSpec struct {
	Handle uint64
	Loaded Loaded
}

func init() {
	airpc.RegisterError("ai.ErrNotPrepared", ErrNotPrepared)
	airpc.RegisterError("ai.ErrBundleTarget", ErrBundleTarget)
	airpc.RegisterError("modelstore.ErrNoRoom", modelstore.ErrNoRoom)
	airpc.RegisterErrorType("ai.LicenseError", (*LicenseError)(nil))
	airpc.RegisterErrorType("chat.PolicyError", (*chat.PolicyError)(nil))
}

// Spec is the app as it is sent to hwnpud. It is for hwnpud's sessions.
func (a *App) Spec() (AppSpec, error) {
	s := AppSpec{Name: a.Name, Memory: a.memory, Turns: a.turns, Allowlist: os.Getenv(chat.PolicyEnv)}
	for _, r := range a.requirements() {
		u := UseSpec{Role: r.role.Name, Name: r.name, As: r.as}
		for _, t := range r.tiers {
			u.Tiers = append(u.Tiers, TierSpec{t.gb, t.name})
		}
		s.Uses = append(s.Uses, u)
	}
	o := a.Store.Models
	if o.Bundle != nil {
		return s, fmt.Errorf("ai: %s's models bundled with it (Store.Models.Bundle) are not read through hwnpud yet (%s=off reads them here)", a.Name, daemon.ModeEnv)
	}
	s.Store = StoreSpec{Root: o.Root, Sources: o.Sources, SourcesSet: o.Sources != nil, Files: o.Files, Definitions: o.Definitions,
		Prepared: a.Store.Prepared, AppDefinitions: a.Store.Definitions}
	if o.HuggingFace != nil {
		s.Store.HuggingFace, s.Store.HuggingFaceSet = *o.HuggingFace, true
	}
	if o.Limits != nil {
		s.Store.Limits, s.Store.LimitsSet = *o.Limits, true
	}
	return s, nil
}

// store is the Store the spec was made of.
func (s StoreSpec) store() Store {
	o := modelstore.Options{Root: s.Root, Files: s.Files, Definitions: s.Definitions}
	if s.SourcesSet {
		o.Sources = append([]string{}, s.Sources...)
	}
	if s.HuggingFaceSet {
		hf := s.HuggingFace
		o.HuggingFace = &hf
	}
	if s.LimitsSet {
		l := s.Limits
		o.Limits = &l
	}
	return Store{Models: o, Prepared: s.Prepared, Definitions: s.AppDefinitions}
}

// daemonClient is this process's client of hwnpud, asked once for a socket (daemon.Use:
// HWNPU_DAEMON, the protocol, the settings that differ): nil to run the models here.
var daemonClient struct {
	sync.Mutex
	socket string // asked of the hwnpud of this socket ("": not asked)
	c      *daemon.Client
}

func init() {
	// hwnpud and its clients compare the definitions they were built with (daemon.Hello)
	daemon.Builtin = [][]byte{hwnpu.ModelsYAML, hwnpu.RecipesYAML}
}

// through is hwnpud's client when the app's models run there: never for an app hwnpud made
// (AppFromSpec), else as HWNPU_DAEMON says (on unless it is set, off in a test).
func (a *App) through(ctx context.Context) (*daemon.Client, error) {
	if a.local || daemon.Mode(daemon.DefaultMode()) == "off" {
		return nil, nil
	}
	daemonClient.Lock()
	defer daemonClient.Unlock()
	if daemonClient.socket == daemon.SocketPath() {
		return daemonClient.c, nil
	}
	c, err := daemon.Use(ctx, daemon.DefaultMode(), Console())
	if err != nil {
		return nil, fmt.Errorf("ai: %w", err)
	}
	daemonClient.socket, daemonClient.c = daemon.SocketPath(), c
	return c, nil
}

// remoteRoles is nil when every use of the app has a role that runs through hwnpud.
func (a *App) remoteRoles() error {
	var missing []string
	for _, r := range a.requirements() {
		if r.role.Remote == nil && !slices.Contains(missing, r.role.Name) {
			missing = append(missing, r.role.Name)
		}
	}
	if len(missing) == 0 {
		return nil
	}
	return fmt.Errorf("ai: the models of %s (%s) do not run through hwnpud yet; %s=off runs them in this process",
		a.Name, strings.Join(missing, ", "), daemon.ModeEnv)
}

// session opens a session with hwnpud's engine and the handle of its Host.
func (a *App) session(ctx context.Context, c *daemon.Client) (*airpc.Session, *airpc.Handle, AppSpec, error) {
	spec, err := a.Spec()
	if err != nil {
		return nil, nil, spec, err
	}
	s, err := c.DialAI(ctx)
	if err != nil {
		return nil, nil, spec, fmt.Errorf("ai: %w", err)
	}
	return s, s.Object(0).Bind(reflect.TypeFor[Host]()), spec, nil
}

// planRemote is Plan made by hwnpud.
func (a *App) planRemote(ctx context.Context, c *daemon.Client) (*Plan, error) {
	if err := a.remoteRoles(); err != nil {
		return nil, err
	}
	s, host, spec, err := a.session(ctx, c)
	if err != nil {
		return nil, err
	}
	defer s.Close()
	ps, err := airpc.Call1[PlanSpec](host, ctx, "Plan", spec)
	if err != nil {
		return nil, err
	}
	p := &Plan{Items: ps.Items, store: a.Store, npu: airole.NPU{Target: ps.Target}, machine: ps.Machine, turns: a.turns,
		remote: a, client: c, licenses: ps.Licenses}
	reqs := a.requirements()
	for i := range p.Items {
		it := &p.Items[i]
		if j := slices.IndexFunc(reqs, func(r requirement) bool { return r.role.Name == it.Role && r.name == it.Use }); j >= 0 {
			it.req = reqs[j]
		}
	}
	return p, nil
}

// runRemote is Run in hwnpud: the stages are told to fn, the models' logs written to log (nil
// for none) as they come.
func (p *Plan) runRemote(ctx context.Context, fn func(Progress), log io.Writer) error {
	s, host, spec, err := p.remote.session(ctx, p.client)
	if err != nil {
		return err
	}
	defer s.Close()
	var logFn func([]byte)
	if log != nil {
		logFn = func(b []byte) { log.Write(b) }
	}
	items, err := airpc.Call1[[]Item](host, ctx, "Run", spec,
		RunSpec{NoCompile: p.NoCompile, AcceptLicenses: p.AcceptLicenses, AllowOffNPU: p.AllowOffNPU, Live: log != nil && console.Live(log)}, logFn, fn)
	for i := range min(len(items), len(p.Items)) {
		it := &p.Items[i]
		it.Ready, it.Compiled, it.Download, it.Bundle = items[i].Ready, items[i].Compiled, items[i].Download, items[i].Bundle
	}
	return err
}

// remoteRuntime is a Runtime whose models hwnpud holds: a session, its Host, and the target
// hwnpud's NPU is.
type remoteRuntime struct {
	s    *airpc.Session
	host *airpc.Handle
}

// openRemote is Open in hwnpud.
func (a *App) openRemote(ctx context.Context, c *daemon.Client, opts []Option) (*Runtime, error) {
	if err := a.remoteRoles(); err != nil {
		return nil, err
	}
	rt := &Runtime{app: a, loaded: map[string]*loadedModel{}}
	for _, o := range opts {
		o(&rt.o)
	}
	s, host, spec, err := a.session(ctx, c)
	if err != nil {
		return nil, err
	}
	// a setting the engine does not know would be dropped there without a word
	if err := c.Server.CheckSettings(rt.o.settings); err != nil {
		s.Close()
		return nil, fmt.Errorf("ai: %w", err)
	}
	o := OpenSpec{Compile: rt.o.compile, AllowOffNPU: rt.o.allowOffNPU, Settings: rt.o.settings}
	var logFn func([]byte)
	if w := rt.o.log; w != nil {
		logFn = func(b []byte) { w.Write(b) }
		o.Live = console.Live(w)
	}
	// the log and the loading's are kept: hwnpud's models write their logs after their loads too
	out, err := host.CallKeep(ctx, "Open", spec, o, logFn, rt.o.loading)
	if err != nil {
		s.Close()
		return nil, err
	}
	rt.npu.Target, _ = out[0].Interface().(string)
	rt.remote = &remoteRuntime{s: s, host: host}
	return rt, nil
}

// get is Runtime.get in hwnpud: the model hwnpud loaded, as the role's type of this process.
// The caller holds rt.mu.
func (r *remoteRuntime) get(ctx context.Context, rt *Runtime, req requirement) (io.Closer, error) {
	if req.role.Remote == nil {
		return nil, rt.app.remoteRoles()
	}
	// hwnpud releases the others of the use's group of InTurns as it loads it
	for _, k := range rt.app.turnOf(req.key()) {
		if l, ok := rt.loaded[k]; ok && k != req.key() {
			delete(rt.loaded, k)
			l.closer.Close()
		}
	}
	got, err := airpc.Call1[GotSpec](r.host, ctx, "Get", req.key())
	if err != nil {
		return nil, err
	}
	h := r.s.Object(got.Handle)
	m := req.role.Remote(h)
	h.Bind(m)
	rt.loaded[req.key()] = &loadedModel{value: m, closer: m, info: got.Loaded}
	return m, nil
}

// release is Runtime.release in hwnpud.
func (r *remoteRuntime) release(rt *Runtime, req requirement) error {
	l, ok := rt.loaded[req.key()]
	if !ok {
		return nil
	}
	delete(rt.loaded, req.key())
	l.closer.Close()
	return airpc.Call0(r.host, context.Background(), "Release", req.key())
}

// close is Runtime.Close in hwnpud: its models unloaded, the session ended.
func (r *remoteRuntime) close(rt *Runtime) error {
	for _, l := range rt.loaded {
		l.closer.Close()
	}
	err := airpc.Call0(r.host, context.Background(), "CloseRuntime")
	r.s.Close()
	if errors.Is(err, airpc.ErrClosed) {
		return nil // the engine ended: it unloaded them
	}
	return err
}
