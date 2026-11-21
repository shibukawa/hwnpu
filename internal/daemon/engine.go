package daemon

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/experts"
	"github.com/shibukawa/hwnpu/internal/mem"
)

// Engine loads the chat models and runs them, behind the supervisor. It is the only process of
// hwnpud on the NPU: it takes the NPU's lock with its first model (llm.OpenDevice) and gives it
// back when it exits, after Idle without a request.
type Engine struct {
	// Idle is how long it waits without a request before it closes its models and exits; 0 never.
	Idle time.Duration
	// Log receives what goes on outside a load (os.Stderr when nil, the service's log).
	Log io.Writer

	load    sync.Mutex // one load at a time; the next waits
	mu      sync.Mutex
	models  map[string]*held  // by the model's name
	aliases map[string]string // what a load named (and the memory) → the model's name
	relay   relay             // where the models' logs go: the load's while one goes on
	active  atomic.Int64
	last    atomic.Int64
	count   atomic.Int64 // the chat.Servers' request ids
}

type held struct {
	e     *chat.Engine
	model chat.Model
	info  ModelInfo
}

// relay is a writer whose destination changes: the load's stream while a load goes on, the
// service's log otherwise (a model keeps its Config.Log after the load).
type relay struct {
	mu   sync.Mutex
	base io.Writer
	to   io.Writer
	live bool
}

func (r *relay) Write(p []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.to != nil {
		r.to.Write(p)
	}
	return r.base.Write(p)
}

// Terminal says whether the client of the load going on shows a terminal (console.Live).
func (r *relay) Terminal() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.to != nil && r.live
}

func (r *relay) set(to io.Writer, live bool) {
	r.mu.Lock()
	r.to, r.live = to, live
	r.mu.Unlock()
}

// Run serves on socket until ctx ends or the idle time passes, then closes the models.
func (e *Engine) Run(ctx context.Context, socket string) error {
	if e.Log == nil {
		e.Log = os.Stderr
	}
	e.models, e.aliases = map[string]*held{}, map[string]string{}
	e.relay.base = e.Log
	if err := chat.LoadUserModels("", e.Log); err != nil {
		return err
	}
	ln, err := ListenUnix(socket)
	if err != nil {
		return err
	}
	defer os.Remove(socket)
	e.last.Store(time.Now().UnixNano())
	srv := &http.Server{Handler: e.Handler()}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		t := time.NewTicker(5 * time.Second)
		defer t.Stop()
		for {
			select {
			case <-ctx.Done():
				srv.Close()
				return
			case <-t.C:
			}
			if e.Idle > 0 && e.active.Load() == 0 && time.Since(time.Unix(0, e.last.Load())) > e.Idle {
				fmt.Fprintf(e.Log, "hwnpud engine: idle for %s; closing the models\n", e.Idle)
				srv.Close()
				return
			}
		}
	}()
	err = srv.Serve(sameUser{ln, e.Log})
	e.closeAll()
	if errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return err
}

func (e *Engine) closeAll() {
	e.mu.Lock()
	defer e.mu.Unlock()
	for name, h := range e.models {
		h.e.Close()
		delete(e.models, name)
	}
}

// Handler is the engine's routes: status, load, and the OpenAI compatible API of the models it
// holds (a chat completion naming a model it does not hold loads it first).
func (e *Engine) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /hwnpud/v1/status", e.status)
	mux.HandleFunc("POST /hwnpud/v1/load", e.serveLoad)
	mux.HandleFunc("POST /v1/chat/completions", e.completions)
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		e.server(nil, r).Handler().ServeHTTP(w, r)
	})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		e.active.Add(1)
		defer func() {
			e.last.Store(time.Now().UnixNano())
			e.active.Add(-1)
		}()
		mux.ServeHTTP(w, r)
	})
}

func (e *Engine) status(w http.ResponseWriter, r *http.Request) {
	st := Status{Hello: Hello{Protocol: Protocol, Build: Build(), PID: os.Getpid()}}
	e.mu.Lock()
	for _, h := range e.models {
		st.Models = append(st.Models, h.info)
	}
	e.mu.Unlock()
	if b, ok := mem.NPUWired(); ok {
		st.NPUBytes = int64(b)
	}
	// this request is one: the idle time is since the one before
	st.Idle = time.Since(time.Unix(0, e.last.Load())).Round(time.Second).String()
	writeJSONValue(w, http.StatusOK, st)
}

// frames writes a load's log to its client as LoadEvent lines. A client that went away fails no
// load: what it loads stays for the next.
type frames struct {
	mu sync.Mutex
	w  http.ResponseWriter
	rc *http.ResponseController
	ok bool
}

func (f *frames) send(ev LoadEvent) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if !f.ok {
		return
	}
	b, _ := json.Marshal(ev)
	if _, err := f.w.Write(append(b, '\n')); err != nil {
		f.ok = false
		return
	}
	f.rc.Flush()
}

func (f *frames) Write(p []byte) (int, error) {
	f.send(LoadEvent{Log: string(p)})
	return len(p), nil
}

func (e *Engine) serveLoad(w http.ResponseWriter, r *http.Request) {
	var req LoadRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || strings.TrimSpace(req.Model) == "" {
		writeError(w, http.StatusBadRequest, "a load names a model")
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	f := &frames{w: w, rc: http.NewResponseController(w), ok: true}
	f.rc.Flush()
	// the load goes on when its client goes away: the next client gets the model
	h, err := e.loadModel(context.WithoutCancel(r.Context()), req, f)
	if err != nil {
		f.send(LoadEvent{Error: err.Error()})
		return
	}
	info := h.info
	f.send(LoadEvent{Model: &info})
}

// loadModel returns the model a load names, loading it unless the engine holds it; its log goes
// to log (and the service's) while it loads. A model the request's policy refuses is an error,
// held or not.
func (e *Engine) loadModel(ctx context.Context, req LoadRequest, log io.Writer) (*held, error) {
	h, err := e.loadAny(ctx, req, log)
	if err != nil {
		return nil, err
	}
	if err := chat.AllowedAlso(h.model, req.Allowlist); err != nil {
		return nil, err
	}
	return h, nil
}

func (e *Engine) loadAny(ctx context.Context, req LoadRequest, log io.Writer) (*held, error) {
	key := req.Model + "@" + strconv.Itoa(req.Memory)
	if h := e.lookup(key); h != nil {
		return h, nil
	}
	e.load.Lock()
	defer e.load.Unlock()
	if h := e.lookup(key); h != nil {
		return h, nil
	}
	if log != nil {
		e.relay.set(log, req.Terminal)
		defer e.relay.set(nil, false)
	}
	m, ce, err := loadChat(ctx, req, &e.relay, func(name string) *held {
		e.mu.Lock()
		defer e.mu.Unlock()
		return e.models[name]
	})
	if err != nil {
		return nil, err
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if h, ok := e.models[m.Name]; ok { // held already under another name
		e.aliases[key] = m.Name
		return h, nil
	}
	h := &held{e: ce, model: m, info: ModelInfo{Name: m.Name, Context: ce.Context(), Images: ce.Images(), Audio: ce.Audio(),
		Translator: ce.Template == chat.Plamo3, NoThink: ce.NoThink, OnNPU: ce.OnNPU(), Device: ce.DeviceName()}}
	e.models[m.Name] = h
	e.aliases[key] = m.Name
	fmt.Fprintf(e.Log, "hwnpud engine: holds %s\n", m.Name)
	return h, nil
}

func (e *Engine) lookup(key string) *held {
	e.mu.Lock()
	defer e.mu.Unlock()
	if name, ok := e.aliases[key]; ok {
		return e.models[name]
	}
	return nil
}

// server is a chat.Server of the model h (all the models held when nil), with the request's
// think budget.
func (e *Engine) server(h *held, r *http.Request) *chat.Server {
	s := &chat.Server{Think: true, ThinkBudget: 1024, Defaults: chat.Sampling{Temperature: 0.1, TopK: 50, Repetition: 1.05}, Log: e.Log, Count: &e.count}
	if v, err := strconv.Atoi(r.Header.Get(ThinkBudgetHeader)); err == nil && v >= 0 {
		s.ThinkBudget = v
	}
	if h != nil {
		s.Chats = []*chat.Engine{h.e}
		return s
	}
	e.mu.Lock()
	for _, h := range e.models {
		s.Chats = append(s.Chats, h.e)
	}
	e.mu.Unlock()
	return s
}

func (e *Engine) completions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	var head struct {
		Model string `json:"model"`
	}
	json.Unmarshal(body, &head)
	if strings.TrimSpace(head.Model) == "" {
		writeError(w, http.StatusBadRequest, "a chat completion names its model")
		return
	}
	h, err := e.loadModel(context.WithoutCancel(r.Context()), LoadRequest{Model: head.Model, Allowlist: r.Header.Get(AllowlistHeader)}, nil)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	r.Body = io.NopCloser(bytes.NewReader(body))
	e.server(h, r).Handler().ServeHTTP(w, r)
}

// chatBackend is the backend the engine converts for and runs on: Core AI on a Mac (as
// hwnpu-chat), else HWNPU_BACKEND or the NPU found.
func chatBackend() string {
	name := strings.TrimSpace(os.Getenv(llm.BackendEnv))
	if runtime.GOOS == "darwin" {
		switch name {
		case "", llm.BackendCoreML, llm.BackendCoreAI:
			return llm.BackendCoreAI
		}
	}
	return name
}

// loadChat makes ready and loads the chat model req names, as hwnpu-chat does with its default
// flags: the model's definition for the memory it runs as. holding gives the model of a name
// already held, which is not loaded twice.
func loadChat(ctx context.Context, req LoadRequest, log io.Writer, holding func(string) *held) (chat.Model, *chat.Engine, error) {
	backend := chatBackend()
	if backend == llm.BackendCoreAI {
		if _, _, err := llm.FindDevice(ctx, backend); err != nil {
			return chat.Model{}, nil, fmt.Errorf("hwnpud runs on Apple's Neural Engine through Core AI, which needs macOS 27: %w", err)
		}
	}
	root := cache.Root()
	promptDir := chat.PromptStore("", root)
	if iron, err := chat.AMDExport(ctx, req.Model, backend, log); err != nil {
		return chat.Model{}, nil, err
	} else if iron != "" {
		m, _ := chat.LookupFor(req.Model)
		m.Name = cmp.Or(m.Name, filepath.Base(iron))
		if err := chat.AllowedAlso(m, req.Allowlist); err != nil {
			return m, nil, err
		}
		if h := holding(m.Name); h != nil {
			return m, h.e, nil
		}
		c := chat.Config{IRON: iron, Log: log, PromptDir: promptDir, Idle: DefaultIdle}
		if m.Name != "" {
			c.Experts = placement(m)
		}
		ce, err := chat.Load(ctx, c)
		return m, ce, err
	}
	gb := req.Memory
	if _, tag, tagged := strings.Cut(req.Model, ":"); tagged {
		if v, ok := chat.MemoryTag(tag); ok {
			gb = v
		}
	}
	m, err := chat.ResolveOn(ctx, req.Model, gb, log)
	if err != nil {
		return m, nil, err
	}
	if m.Kind != chat.KindChat {
		return m, nil, fmt.Errorf("%s is a %s model, not a chat model", m.Name, m.Kind)
	}
	if err := chat.AllowedAlso(m, req.Allowlist); err != nil {
		return m, nil, err // before it is downloaded
	}
	if h := holding(m.Name); h != nil {
		return m, h.e, nil
	}
	run := m.Run.Over(chat.Run{})
	tiers, err := chat.Tiers(cmp.Or(run.Tier, chat.DefaultTiers))
	if err != nil {
		return m, nil, fmt.Errorf("tier %q: %w", run.Tier, err)
	}
	c, err := chat.Prepare(ctx, chat.Setup{Dir: chat.ModelDir(m), Repo: m.Source(), Arch: m.Arch, Name: m.Name,
		Context: run.Context, Paged: run.Paged, Prefill: run.Prefill, Format: run.Format, Draft: 8, MTP: chat.MTPAuto,
		Tiers: tiers, Backend: backend, CodesHash: run.Codes, Log: log})
	if err != nil {
		return m, nil, err
	}
	c.CacheDir, c.Log, c.PromptDir, c.Idle = root, log, promptDir, DefaultIdle
	c.Experts = placement(m)
	ce, err := chat.Load(ctx, c)
	if err != nil {
		return m, nil, err
	}
	ce.NoThink = m.NoThink
	return m, ce, nil
}

// placement is how a mixture of experts' experts are placed: the model's core first, then by
// hwnpud's own counts of the experts it runs.
func placement(m chat.Model) experts.Placement {
	return experts.Placement{Core: m.ExpertsCore, Idle: DefaultIdle, Ceiling: filepath.Join(chat.ModelDir(m), "experts-usage", "ceiling"),
		Usage: chat.ExpertsUsage(chat.ModelDir(m), "hwnpud")}
}
