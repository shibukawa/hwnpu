package daemon

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/shibukawa/bonsaiai/internal/envfile"
)

// Supervisor holds bonsaiaid's socket and the engine behind it. It never touches the NPU: when the
// engine dies (Core AI takes a process down when its own conversion fails) the socket stays and
// the next request starts a new engine.
type Supervisor struct {
	// Socket is where it listens when the service manager did not start it with one (SocketPath).
	Socket string
	// Idle is how long the engine waits without a request before it exits, and, when the service
	// manager (launchd, systemd) started the supervisor, how long the supervisor waits after that
	// before it exits too; 0 never.
	Idle time.Duration
	// Executable is what runs the engine ("engine" and its flags after it); this executable
	// when "".
	Executable string
	// Log receives what the supervisor does and the engine's standard error (os.Stderr when nil).
	Log io.Writer

	mu       sync.Mutex
	child    *child
	proxy    *httputil.ReverseProxy
	active   atomic.Int64
	last     atomic.Int64 // unix nanoseconds of the last request's end
	started  time.Time    // the executable's modification time when the supervisor started
	exe      string
	srv      *http.Server
	launched bool              // the service manager started it
	since    time.Time         // when it started (Hello.Started)
	restart  atomic.Bool       // a restart was asked: Run returns ErrRestart
	running  map[string]string // the variables the engine started with (watched)
	// environ is the environment of the next engine (and its .env files): the base and the files
	// as they are now; tests give their own
	environ func() ([]string, []string, error)
}

type child struct {
	cmd  *exec.Cmd
	done chan struct{}
}

// ErrRestart is what Run returns when a restart was asked of a supervisor the service manager did
// not start: its caller runs the executable again (bonsaiaid serve does, in the same process).
var ErrRestart = errors.New("bonsaiaid: restart")

// Run serves until ctx ends, the idle time passes (when the service manager started it), or a
// quit request.
func (s *Supervisor) Run(ctx context.Context) error {
	if s.Log == nil {
		s.Log = os.Stderr
	}
	if s.Socket == "" {
		s.Socket = SocketPath()
	}
	exe := s.Executable
	if exe == "" {
		var err error
		if exe, err = os.Executable(); err != nil {
			return err
		}
	}
	s.exe = exe
	s.since = time.Now()
	if s.environ == nil {
		base := envfile.Original()
		s.environ = func() ([]string, []string, error) { return envfile.Environ(base) }
	}
	if fi, err := os.Stat(exe); err == nil {
		s.started = fi.ModTime()
	}
	lns, err := activated(SocketName)
	if err != nil {
		return err
	}
	launched := len(lns) > 0
	s.launched = launched
	if launched {
		// the service manager's socket, which the engine's is beside. launchd binds it in a
		// directory of root's (/var/run/com.apple.launchd.*) and moves it to the plist's path, and
		// the listener keeps the first name, where the engine's cannot be made: on the Mac it is
		// the path given (the plist's, DefaultSocketPath), on Linux systemd's own
		if a, ok := lns[0].Addr().(*net.UnixAddr); ok && a.Name != "" && runtime.GOOS != "darwin" {
			s.Socket = a.Name
		}
	} else {
		ln, err := ListenUnix(s.Socket)
		if err != nil {
			return err
		}
		lns = []net.Listener{ln}
	}
	engineSocket := EngineSocketPath(s.Socket)
	s.proxy = &httputil.ReverseProxy{
		Rewrite: func(pr *httputil.ProxyRequest) {
			pr.SetURL(&url.URL{Scheme: "http", Host: "engine"})
		},
		Transport:     UnixTransport(engineSocket),
		FlushInterval: -1, // the tokens and the log as they come
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if r.Context().Err() != nil {
				return // the client went away
			}
			writeError(w, http.StatusBadGateway, fmt.Sprintf("bonsaiaid's engine: %v (its log: %s)", err, LogWhere()))
		},
	}
	s.last.Store(time.Now().UnixNano())

	mux := http.NewServeMux()
	mux.HandleFunc("GET /bonsaiaid/v1/hello", func(w http.ResponseWriter, r *http.Request) {
		writeJSONValue(w, http.StatusOK, s.hello())
	})
	mux.HandleFunc("GET /bonsaiaid/v1/status", s.status)
	mux.HandleFunc("POST /bonsaiaid/v1/start", func(w http.ResponseWriter, r *http.Request) {
		if err := s.ensureEngine(r.Context(), engineSocket); err != nil {
			writeError(w, http.StatusServiceUnavailable, err.Error())
			return
		}
		writeJSONValue(w, http.StatusOK, s.hello())
	})
	mux.HandleFunc("POST /bonsaiaid/v1/restart", func(w http.ResponseWriter, r *http.Request) {
		// the service manager starts the next at the next connection; one started by hand runs
		// its executable again itself
		s.restart.Store(true)
		w.WriteHeader(http.StatusNoContent)
		go s.shutdown()
	})
	mux.HandleFunc("POST /bonsaiaid/v1/reload", func(w http.ResponseWriter, r *http.Request) {
		s.reload(1) // this request is one
		writeJSONValue(w, http.StatusOK, s.hello())
	})
	mux.HandleFunc("POST /bonsaiaid/v1/quit", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
		go s.shutdown()
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if err := s.ensureEngine(r.Context(), engineSocket); err != nil {
			writeError(w, http.StatusServiceUnavailable, err.Error())
			return
		}
		s.proxy.ServeHTTP(w, r)
	})
	s.srv = &http.Server{Handler: s.track(mux)}
	how := "listening"
	if launched {
		how = "started by " + serviceManager()
	}
	fmt.Fprintf(s.Log, "bonsaiaid %s (protocol %d, pid %d): %s on %s\n", Build(), Protocol, os.Getpid(), how, s.Socket)

	errs := make(chan error, len(lns))
	for _, ln := range lns {
		go func() { errs <- s.srv.Serve(sameUser{ln, s.Log}) }()
	}
	stopIdle := make(chan struct{})
	defer close(stopIdle)
	go s.watch(ctx, launched, stopIdle)
	err = <-errs
	s.stopEngine()
	if !launched {
		os.Remove(s.Socket)
	}
	if errors.Is(err, http.ErrServerClosed) {
		if s.restart.Load() && !launched {
			return ErrRestart
		}
		return nil
	}
	return err
}

// watch shuts the supervisor down when ctx ends, and, when the service manager started it, after
// the idle time without an engine or a request, or when its executable has been replaced (go
// install): the service manager starts the new one at the next connection.
func (s *Supervisor) watch(ctx context.Context, launched bool, stop chan struct{}) {
	t := time.NewTicker(5 * time.Second)
	defer t.Stop()
	for {
		select {
		case <-stop:
			return
		case <-ctx.Done():
			s.shutdown()
			return
		case <-t.C:
		}
		s.reload(0)
		if !launched || s.active.Load() > 0 || s.engineRunning() {
			continue
		}
		since := time.Since(time.Unix(0, s.last.Load()))
		if s.replaced() {
			fmt.Fprintf(s.Log, "bonsaiaid: %s has been replaced; exiting for %s to start the new one\n", s.exe, serviceManager())
			s.shutdown()
			return
		}
		if s.Idle > 0 && since > s.Idle {
			fmt.Fprintf(s.Log, "bonsaiaid: idle for %s; exiting\n", since.Round(time.Second))
			s.shutdown()
			return
		}
	}
}

// reload stops the engine when the .env files changed what it was started with and no request
// but the others of the caller (others) goes on: the next request starts one with the new
// settings. It reports the names that changed, nil when it did not stop the engine.
func (s *Supervisor) reload(others int64) []string {
	if !s.engineRunning() || s.active.Load() > others {
		return nil
	}
	env, _, err := s.environ()
	if err != nil {
		return nil // the file is being written, or wrong: hello tells
	}
	s.mu.Lock()
	names := changed(s.running, watched(env))
	s.mu.Unlock()
	if len(names) == 0 {
		return nil
	}
	fmt.Fprintf(s.Log, "bonsaiaid: %s changed; stopping the engine for the next request to start it with them\n", strings.Join(names, ", "))
	s.stopEngine()
	return names
}

// replaced reports whether the executable on disk is not the one that started.
func (s *Supervisor) replaced() bool {
	fi, err := os.Stat(s.exe)
	return err == nil && !s.started.IsZero() && !fi.ModTime().Equal(s.started)
}

func (s *Supervisor) shutdown() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	s.stopEngine()
	s.srv.Shutdown(ctx)
}

// track counts the requests going on and when the last ended.
func (s *Supervisor) track(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.active.Add(1)
		defer func() {
			s.last.Store(time.Now().UnixNano())
			s.active.Add(-1)
		}()
		h.ServeHTTP(w, r)
	})
}

func (s *Supervisor) hello() Hello {
	h := Hello{Protocol: Protocol, Build: Build(), PID: os.Getpid(), Executable: s.exe, Started: s.since.UnixNano()}
	if s.launched {
		h.Service = serviceManager()
	}
	env, files, err := s.environ()
	h.Settings, h.Files = Settings(env), files
	if err != nil {
		h.SettingsError = err.Error()
	}
	s.mu.Lock()
	if s.child != nil {
		h.Engine = s.child.cmd.Process.Pid
		h.Pending = changed(s.running, watched(env))
	}
	s.mu.Unlock()
	return h
}

// status is the engine's status when it runs (without starting it), else the supervisor's.
func (s *Supervisor) status(w http.ResponseWriter, r *http.Request) {
	if !s.engineRunning() {
		writeJSONValue(w, http.StatusOK, Status{Hello: s.hello()})
		return
	}
	c := NewClient(EngineSocketPath(s.Socket))
	st, err := c.Status(r.Context())
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error())
		return
	}
	st.Hello = s.hello()
	writeJSONValue(w, http.StatusOK, st)
}

func (s *Supervisor) engineRunning() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.child != nil
}

// ensureEngine starts the engine unless it runs, and waits until it answers on its socket.
func (s *Supervisor) ensureEngine(ctx context.Context, socket string) error {
	env, _, err := s.environ()
	if err != nil {
		fmt.Fprintf(s.Log, "bonsaiaid: %v\n", err)
	}
	set := watched(env)
	s.mu.Lock()
	// the .env files changed: a new engine reads them, when no other request needs this one
	if s.child != nil && s.active.Load() <= 1 {
		if names := changed(s.running, set); len(names) > 0 {
			s.mu.Unlock()
			fmt.Fprintf(s.Log, "bonsaiaid: %s changed; starting the engine again\n", strings.Join(names, ", "))
			s.stopEngine()
			s.mu.Lock()
		}
	}
	c := s.child
	if c == nil {
		os.Remove(socket)
		cmd := exec.Command(s.exe, "engine", "--socket", socket, "--idle", s.Idle.String())
		cmd.Stdout, cmd.Stderr, cmd.Env = s.Log, s.Log, withoutActivation(env)
		cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} // a Ctrl-C at the terminal of a supervisor run by hand is the supervisor's to pass on
		if err := cmd.Start(); err != nil {
			s.mu.Unlock()
			return fmt.Errorf("bonsaiaid: starting the engine: %w", err)
		}
		c = &child{cmd: cmd, done: make(chan struct{})}
		s.child, s.running = c, set
		fmt.Fprintf(s.Log, "bonsaiaid: engine started (pid %d)\n", cmd.Process.Pid)
		go func() {
			err := cmd.Wait()
			s.mu.Lock()
			if s.child == c {
				s.child = nil
			}
			s.mu.Unlock()
			close(c.done)
			if err != nil {
				fmt.Fprintf(s.Log, "bonsaiaid: engine (pid %d) ended: %v\n", cmd.Process.Pid, err)
			} else {
				fmt.Fprintf(s.Log, "bonsaiaid: engine (pid %d) exited\n", cmd.Process.Pid)
			}
		}()
	}
	s.mu.Unlock()
	deadline := time.After(60 * time.Second)
	for {
		conn, err := net.Dial("unix", socket)
		if err == nil {
			conn.Close()
			return nil
		}
		select {
		case <-c.done:
			return fmt.Errorf("bonsaiaid: the engine ended as it started (its log: %s)", LogWhere())
		case <-deadline:
			return errors.New("bonsaiaid: the engine does not answer on " + socket)
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(20 * time.Millisecond):
		}
	}
}

// stopEngine asks the engine to end (SIGTERM: it closes its models) and kills it after a while.
func (s *Supervisor) stopEngine() {
	s.mu.Lock()
	c := s.child
	s.mu.Unlock()
	if c == nil {
		return
	}
	c.cmd.Process.Signal(syscall.SIGTERM)
	select {
	case <-c.done:
	case <-time.After(30 * time.Second):
		c.cmd.Process.Kill()
		<-c.done
	}
}

// ListenUnix listens on a Unix domain socket at path, only for this user: it makes the directory,
// and replaces a socket file that nothing answers on (one left by a process that died). A socket
// that answers is an error: another bonsaiaid runs.
func ListenUnix(path string) (net.Listener, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	if _, err := os.Lstat(path); err == nil {
		if conn, err := net.DialTimeout("unix", path, time.Second); err == nil {
			conn.Close()
			return nil, fmt.Errorf("another bonsaiaid answers on %s (bonsaiai service stop ends it)", path)
		}
		if err := os.Remove(path); err != nil {
			return nil, err
		}
	}
	ln, err := net.Listen("unix", path)
	if err != nil {
		return nil, err
	}
	ln.(*net.UnixListener).SetUnlinkOnClose(false)
	if err := os.Chmod(path, 0o600); err != nil {
		ln.Close()
		return nil, err
	}
	return ln, nil
}

func writeJSONValue(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeJSONValue(w, code, map[string]any{"error": map[string]any{"message": msg, "type": "bonsaiaid_error"}})
}
