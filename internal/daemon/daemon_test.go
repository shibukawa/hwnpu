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
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	socket := filepath.Join(shortDir(t), "d.sock")
	var log bytes.Buffer
	s := &Supervisor{Socket: socket, Executable: exe, Log: &syncWriter{w: &log}}
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
	if c, err := Use(context.Background(), "off"); c != nil || err != nil {
		t.Fatalf("off: %v, %v", c, err)
	}
	t.Setenv(ModeEnv, "auto")
	if c, err := Use(context.Background(), "off"); c != nil || err != nil {
		t.Fatalf("auto without a daemon: %v, %v", c, err)
	}
	t.Setenv(ModeEnv, "on")
	if _, err := Use(context.Background(), "off"); !errors.Is(err, ErrNotRunning) {
		t.Fatalf("on without a daemon: %v", err)
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
