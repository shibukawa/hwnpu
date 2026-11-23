package daemon

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// TestMain runs a fake engine when the supervisor under test starts this test binary as its
// engine ("engine --socket PATH --idle D"): it answers status and load as the engine does,
// without a model or the NPU.
func TestMain(m *testing.M) {
	if len(os.Args) > 3 && os.Args[1] == "engine" && os.Args[2] == "--socket" {
		os.Exit(fakeEngine(os.Args[3]))
	}
	os.Exit(m.Run())
}

func fakeEngine(socket string) int {
	ln, err := ListenUnix(socket)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /hwnpud/v1/status", func(w http.ResponseWriter, r *http.Request) {
		writeJSONValue(w, http.StatusOK, Status{Hello: Hello{Protocol: Protocol, PID: os.Getpid()}, Models: []ModelInfo{{Name: "fake", Context: 4096, OnNPU: true}}})
	})
	mux.HandleFunc("POST /hwnpud/v1/load", func(w http.ResponseWriter, r *http.Request) {
		var req LoadRequest
		json.NewDecoder(r.Body).Decode(&req)
		f := &frames{w: w, rc: http.NewResponseController(w), ok: true}
		if req.Model == "missing" {
			f.send(LoadEvent{Log: "looking for missing\n"})
			f.send(LoadEvent{Error: "no model missing"})
			return
		}
		fmt.Fprintf(f, "downloading %s\n", req.Model)
		fmt.Fprintf(f, "terminal %v\n", req.Terminal)
		if v := os.Getenv("HWNPU_CACHE"); v != "" {
			fmt.Fprintf(f, "cache %s\n", v)
		}
		f.send(LoadEvent{Model: &ModelInfo{Name: req.Model, Context: 4096, OnNPU: true}})
	})
	mux.HandleFunc("POST /crash", func(w http.ResponseWriter, r *http.Request) { os.Exit(3) })
	srv := &http.Server{Handler: mux}
	go func() {
		ch := make(chan os.Signal, 1)
		signal.Notify(ch, syscall.SIGTERM)
		<-ch
		srv.Close()
	}()
	srv.Serve(ln)
	return 0
}

// shortDir is a directory whose sockets' paths fit in sun_path (104 bytes on a Mac; t.TempDir is
// long there).
func shortDir(t *testing.T) string {
	d, err := os.MkdirTemp("/tmp", "hwnpud")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(d) })
	return d
}

func startSupervisor(t *testing.T) (*Client, chan error) {
	t.Helper()
	return startSupervisorWith(t, nil)
}

// startSupervisorWith starts a supervisor whose engines get the environment environ gives.
func startSupervisorWith(t *testing.T, environ func() ([]string, []string, error)) (*Client, chan error) {
	t.Helper()
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	socket := filepath.Join(shortDir(t), "d.sock")
	var log bytes.Buffer
	s := &Supervisor{Socket: socket, Executable: exe, Log: &syncWriter{w: &log}, environ: environ}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()
	t.Cleanup(func() {
		cancel()
		select {
		case <-done:
		case <-time.After(40 * time.Second):
			t.Error("the supervisor did not end")
		}
		if t.Failed() {
			t.Logf("supervisor's log:\n%s", log.String())
		}
	})
	c := NewClient(socket)
	for i := 0; ; i++ {
		if _, err := c.Hello(context.Background()); err == nil {
			break
		}
		if i > 200 {
			t.Fatal("the supervisor does not answer")
		}
		time.Sleep(10 * time.Millisecond)
	}
	return c, done
}

func TestSupervisor(t *testing.T) {
	c, done := startSupervisor(t)
	ctx := context.Background()
	h, err := c.Hello(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if h.Protocol != Protocol || h.PID != os.Getpid() || h.Engine != 0 {
		t.Fatalf("hello %+v: want protocol %d, this process, no engine yet", h, Protocol)
	}
	st, err := c.Status(ctx)
	if err != nil || st.Engine != 0 || len(st.Models) != 0 {
		t.Fatalf("status before the engine: %+v, %v", st, err)
	}

	// a load starts the engine, and its log comes before the model
	var log bytes.Buffer
	info, err := c.Load(ctx, LoadRequest{Model: "LFM3-1.7B", Terminal: true}, &log)
	if err != nil {
		t.Fatal(err)
	}
	if info.Name != "LFM3-1.7B" || log.String() != "downloading LFM3-1.7B\nterminal true\n" {
		t.Fatalf("load: %+v, log %q", info, log.String())
	}
	st, err = c.Status(ctx)
	if err != nil || st.Engine == 0 || len(st.Models) != 1 || st.PID != os.Getpid() {
		t.Fatalf("status with the engine: %+v, %v", st, err)
	}
	first := st.Engine

	// an error of the load comes back as an error, after its log
	log.Reset()
	if _, err := c.Load(ctx, LoadRequest{Model: "missing"}, &log); err == nil || err.Error() != "no model missing" || log.String() != "looking for missing\n" {
		t.Fatalf("load of a missing model: %v, log %q", err, log.String())
	}

	// the engine dies: the next request starts another
	req, _ := http.NewRequest(http.MethodPost, c.URL("/crash"), nil)
	if resp, err := c.HTTP.Do(req); err == nil {
		resp.Body.Close()
	}
	for i := 0; ; i++ {
		h, _ := c.Hello(ctx)
		if h.Engine == 0 {
			break
		}
		if i > 300 {
			t.Fatal("the supervisor did not see the engine end")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := c.Load(ctx, LoadRequest{Model: "again"}, nil); err != nil {
		t.Fatal(err)
	}
	if h, _ := c.Hello(ctx); h.Engine == 0 || h.Engine == first {
		t.Fatalf("engine after the crash: %d (first %d)", h.Engine, first)
	}

	// quit ends the engine and the supervisor
	if err := c.Quit(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
		done <- nil // for the cleanup
	case <-time.After(40 * time.Second):
		t.Fatal("the supervisor did not quit")
	}
	if _, err := c.Hello(ctx); !errors.Is(err, ErrNotRunning) {
		t.Fatalf("hello after quit: %v, want ErrNotRunning", err)
	}
}

// TestRestartByHand: a supervisor not started by the service manager ends at restart with
// ErrRestart, for hwnpud serve to run itself again.
func TestRestartByHand(t *testing.T) {
	c, done := startSupervisor(t)
	ctx := context.Background()
	h, err := c.Hello(ctx)
	if err != nil || h.Service != "" || h.Started == 0 {
		t.Fatalf("hello %+v, %v", h, err)
	}
	if _, err := c.Start(ctx); err != nil {
		t.Fatal(err)
	}
	req, _ := http.NewRequest(http.MethodPost, c.URL("/hwnpud/v1/restart"), nil)
	resp, err := c.HTTP.Do(req)
	if err != nil || resp.StatusCode != http.StatusNoContent {
		t.Fatalf("restart: %v, %v", resp, err)
	}
	resp.Body.Close()
	select {
	case err := <-done:
		if !errors.Is(err, ErrRestart) {
			t.Fatalf("run ended with %v, want ErrRestart", err)
		}
		done <- nil // for the cleanup
	case <-time.After(40 * time.Second):
		t.Fatal("the supervisor did not end")
	}
}

func TestListenUnixReplacesStaleSocket(t *testing.T) {
	path := filepath.Join(shortDir(t), "s.sock")
	ln, err := ListenUnix(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ListenUnix(path); err == nil || !strings.Contains(err.Error(), "another hwnpud") {
		t.Fatalf("a second listener on a live socket: %v", err)
	}
	ln.Close() // the file stays, as when a process dies
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("the socket file is gone: %v", err)
	}
	ln, err = ListenUnix(path)
	if err != nil {
		t.Fatalf("over a stale socket: %v", err)
	}
	ln.Close()
	if fi, err := os.Stat(path); err != nil || fi.Mode().Perm() != 0o600 {
		t.Fatalf("socket mode: %v, %v", fi.Mode(), err)
	}
}

func TestMode(t *testing.T) {
	for in, want := range map[string]string{"": "off", "on": "on", "AUTO": "auto", "1": "on", "no": "off", "x": "off"} {
		t.Setenv(ModeEnv, in)
		if got := Mode("off"); got != want {
			t.Errorf("Mode(%q) = %q, want %q", in, got, want)
		}
	}
	t.Setenv(ModeEnv, "")
	if got := Mode("auto"); got != "auto" {
		t.Errorf("unset with auto: %q", got)
	}
}

func TestUseOffAndAuto(t *testing.T) {
	t.Setenv(SocketEnv, filepath.Join(shortDir(t), "none.sock"))
	t.Setenv(ModeEnv, "")
	if c, err := Use(context.Background(), "off", io.Discard); c != nil || err != nil {
		t.Fatalf("off: %v, %v", c, err)
	}
	t.Setenv(ModeEnv, "auto")
	if c, err := Use(context.Background(), "off", io.Discard); c != nil || err != nil {
		t.Fatalf("auto without a daemon: %v, %v", c, err)
	}
	t.Setenv(ModeEnv, "on")
	if _, err := Use(context.Background(), "off", io.Discard); !errors.Is(err, ErrNotRunning) {
		t.Fatalf("on without a daemon: %v", err)
	}
}

// TestSettings: hwnpud tells its settings, a changed .env starts the next engine with the new
// ones, and a client of other settings is told (on) or runs by itself (auto).
func TestSettings(t *testing.T) {
	var mu sync.Mutex
	cache, extra := "/a", "OTHER=1"
	environ := func() ([]string, []string, error) {
		mu.Lock()
		defer mu.Unlock()
		return []string{"PATH=" + os.Getenv("PATH"), "HWNPU_CACHE=" + cache, "HWNPU_MODEL_SOURCE=https://me:pw@nas/llm", "HF_TOKEN=hf_x", extra}, []string{"/u/.env"}, nil
	}
	c, _ := startSupervisorWith(t, environ)
	ctx := context.Background()
	h, err := c.Hello(ctx)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]string{"HWNPU_CACHE": "/a", "HWNPU_MODEL_SOURCE": "https://me:xxxxx@nas/llm", "HF_TOKEN": Secret}
	if fmt.Sprint(h.Settings) != fmt.Sprint(want) || len(h.Files) != 1 {
		t.Fatalf("hello's settings %v %v, want %v", h.Settings, h.Files, want)
	}
	load := func() (string, int) {
		var log bytes.Buffer
		if _, err := c.Load(ctx, LoadRequest{Model: "m"}, &log); err != nil {
			t.Fatal(err)
		}
		h, _ := c.Hello(ctx)
		return log.String(), h.Engine
	}
	log, first := load()
	if !strings.Contains(log, "cache /a\n") {
		t.Fatalf("the engine's environment: %q", log)
	}
	mu.Lock()
	cache = "/b"
	mu.Unlock()
	if h, _ := c.Hello(ctx); fmt.Sprint(h.Pending) != "[HWNPU_CACHE]" {
		t.Fatalf("pending %v", h.Pending)
	}
	// reload (hwnpu config set, and the supervisor every few seconds) stops the engine of the old
	if h, err := c.Reload(ctx); err != nil || h.Engine != 0 || len(h.Pending) != 0 {
		t.Fatalf("reload: %+v, %v", h, err)
	}
	log, second := load()
	if !strings.Contains(log, "cache /b\n") || second == first {
		t.Fatalf("after the change: engine %d (was %d), %q", second, first, log)
	}
	if h, _ := c.Hello(ctx); len(h.Pending) != 0 {
		t.Fatalf("pending %v after the new engine", h.Pending)
	}
	// start and restart: the engine runs after them, a new one after restart
	if h, err := c.Start(ctx); err != nil || h.Engine != second {
		t.Fatalf("start with the engine running: %+v, %v (engine %d)", h, err, second)
	}
	_ = second
	// the model policy is reported, but a client's own is not a difference
	mu.Lock()
	extra = "HWNPU_MODEL_ALLOWLIST=+nsfw"
	mu.Unlock()
	if h, _ := c.Hello(ctx); h.Settings["HWNPU_MODEL_ALLOWLIST"] != "+nsfw" || fmt.Sprint(h.Pending) != "[HWNPU_MODEL_ALLOWLIST]" {
		t.Fatalf("hello with a policy: %v, pending %v", h.Settings, h.Pending)
	}

	t.Setenv(SocketEnv, c.Socket)
	t.Setenv("HWNPU_CACHE", "/here")
	t.Setenv("HWNPU_MODEL_SOURCE", "https://nas/llm")
	t.Setenv("HF_TOKEN", "hf_y")                                 // set on both sides: not a difference
	t.Setenv("HWNPU_MODEL_ALLOWLIST", "-license:non-commercial") // the client's: it narrows, not a difference
	t.Setenv(ModeEnv, "auto")
	var out bytes.Buffer
	if cl, err := Use(ctx, "off", &out); cl != nil || err != nil {
		t.Fatalf("auto with other settings: %v, %v", cl, err)
	}
	if s := out.String(); !strings.Contains(s, "HWNPU_CACHE is /here here, /b in hwnpud") || !strings.Contains(s, "running here") || strings.Contains(s, "HF_TOKEN") {
		t.Fatalf("auto's log: %q", s)
	}
	t.Setenv(ModeEnv, "on")
	out.Reset()
	if cl, err := Use(ctx, "off", &out); cl == nil || err != nil {
		t.Fatalf("on with other settings: %v, %v", cl, err)
	}
	if s := out.String(); !strings.Contains(s, "through hwnpud with its settings") || !strings.Contains(s, "/u/.env") {
		t.Fatalf("on's log: %q", s)
	}
	t.Setenv("HWNPU_CACHE", "/b")
	t.Setenv("HWNPU_MODEL_SOURCE", "https://me:other@nas/llm") // passwords are not compared
	out.Reset()
	if cl, _ := Use(ctx, "off", &out); cl == nil || out.Len() != 0 {
		t.Fatalf("the same settings: %v, %q", cl, out.String())
	}
}

func TestEngineSocketPath(t *testing.T) {
	if got := EngineSocketPath("/a/hwnpud.sock"); got != "/a/hwnpud-engine.sock" {
		t.Fatal(got)
	}
}

type syncWriter struct {
	mu sync.Mutex
	w  io.Writer
}

func (s *syncWriter) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.w.Write(p)
}

// The engine is not given systemd's activation variables, which name the supervisor's
// descriptors.
func TestWithoutActivation(t *testing.T) {
	got := withoutActivation([]string{"A=1", "LISTEN_PID=4", "LISTEN_FDS=1", "LISTEN_FDNAMES=hwnpud", "LISTEN_FDSX=2"})
	if strings.Join(got, " ") != "A=1 LISTEN_FDSX=2" {
		t.Fatalf("%v", got)
	}
}
