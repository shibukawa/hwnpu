// Package daemon is bonsaiaid, the per-user process that owns the NPU and the models
// (docs/design/bonsaiaid.md), and what its clients need of it: where its socket is, whether to use
// it (BONSAIAI_DAEMON), and a client over the socket.
//
// bonsaiaid is two processes. The supervisor (Supervisor) holds the socket that the service manager
// (launchd, systemd) or the supervisor itself listens on, answers hello and status, and passes
// the rest to the engine (Engine), the same executable run again with "engine", which loads and
// runs the models. The
// engine exits after it has been idle, giving the NPU's memory and its lock back; the supervisor
// starts it again at the next request.
//
// The protocol is HTTP over the Unix domain socket: bonsaiaid's own routes under /bonsaiaid/v1/ and the
// OpenAI compatible API of chat.Server under /v1/.
package daemon

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/bonsaiai/internal/airpc"
	"github.com/shibukawa/bonsaiai/internal/envfile"
)

// Protocol is the version of the routes and messages; a client and a daemon of different
// protocols do not talk.
const Protocol = 1

// The variables a client reads.
const (
	// ModeEnv says whether to go through bonsaiaid: on (always, an error when it does not answer),
	// off (never), auto (when it answers with this Protocol).
	ModeEnv = "BONSAIAI_DAEMON"
	// SocketEnv is another socket, to try a bonsaiaid built from a worktree beside the installed one.
	SocketEnv = "BONSAIAI_DAEMON_SOCKET"
	// IdleEnv is how long the engine and the supervisor wait without a request before they exit
	// (a Go duration; 0 never).
	IdleEnv = "BONSAIAI_DAEMON_IDLE"
)

// DefaultIdle is how long the engine keeps the models without a request.
const DefaultIdle = 5 * time.Minute

// Label is the name of the service on the Mac (launchd's label).
const Label = "com.github.shibukawa.bonsaiaid"

// SocketName is the name of the socket in the service's definition (launchd's Sockets key,
// systemd's FileDescriptorName).
const SocketName = "bonsaiaid"

// SocketPath is the socket clients connect to: BONSAIAI_DAEMON_SOCKET, else DefaultSocketPath. It does not follow BONSAIAI_CACHE: the
// service is started without the shell's variables, and every client must find the same socket.
func SocketPath() string {
	if p := strings.TrimSpace(os.Getenv(SocketEnv)); p != "" {
		return p
	}
	return DefaultSocketPath()
}

// DefaultSocketPath is the socket of the installed service: bonsaiaid.sock in XDG_RUNTIME_DIR on
// Linux (where systemd's user units listen), else in the user's cache directory.
func DefaultSocketPath() string {
	if runtime.GOOS == "linux" {
		if d := os.Getenv("XDG_RUNTIME_DIR"); d != "" {
			return filepath.Join(d, "bonsaiaid.sock")
		}
	}
	dir, err := os.UserCacheDir()
	if err != nil {
		dir = os.TempDir()
	}
	return filepath.Join(dir, "bonsaiai", "bonsaiaid.sock")
}

// EngineSocketPath is the engine's own socket, beside the supervisor's.
func EngineSocketPath(socket string) string {
	return strings.TrimSuffix(socket, ".sock") + "-engine.sock"
}

// Idle is BONSAIAI_DAEMON_IDLE, else DefaultIdle.
func Idle() time.Duration {
	if s := strings.TrimSpace(os.Getenv(IdleEnv)); s != "" {
		if d, err := time.ParseDuration(s); err == nil && d >= 0 {
			return d
		}
	}
	return DefaultIdle
}

// DefaultMode is what BONSAIAI_DAEMON is when it is not set: on (every program goes through
// bonsaiaid, and fails when it does not answer), except in a test (go test), which runs in its own
// process. bonsaiaid's engine sets it off for itself.
func DefaultMode() string {
	if testing.Testing() {
		return "off"
	}
	return "on"
}

// Mode is BONSAIAI_DAEMON: "on", "auto" or "off" (anything else, and unset, is def).
func Mode(def string) string {
	switch m := strings.ToLower(strings.TrimSpace(os.Getenv(ModeEnv))); m {
	case "on", "auto", "off":
		return m
	case "1", "true", "yes":
		return "on"
	case "0", "false", "no":
		return "off"
	}
	return def
}

// Build names this executable's build: its module version and VCS revision ("devel" without them).
func Build() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "devel"
	}
	rev, dirty := "", false
	for _, s := range info.Settings {
		switch s.Key {
		case "vcs.revision":
			rev = s.Value
		case "vcs.modified":
			dirty = s.Value == "true"
		}
	}
	if rev == "" {
		return "devel"
	}
	if len(rev) > 12 {
		rev = rev[:12]
	}
	if dirty {
		rev += "+dirty"
	}
	return rev
}

// Hello is the answer of GET /bonsaiaid/v1/hello.
type Hello struct {
	Protocol int    `json:"protocol"`
	Build    string `json:"build"`
	PID      int    `json:"pid"`
	// Engine is the engine's process, 0 when it is not running.
	Engine int `json:"engine"`
	// Executable is the supervisor's executable.
	Executable string `json:"executable"`
	// Service is the service manager that started the supervisor (launchd, systemd), "" when it
	// was started by hand (bonsaiaid serve): one that quits is started again only by the former.
	Service string `json:"service,omitempty"`
	// Started is when the supervisor started (unix nanoseconds): a restart by hand keeps the
	// process, and this tells the new from the old.
	Started int64 `json:"started,omitempty"`
	// Settings are bonsaiaid's settings (Settings): those the next engine starts with, read from the
	// .env files now.
	Settings map[string]string `json:"settings,omitempty"`
	// Files are the .env files they were read from.
	Files []string `json:"files,omitempty"`
	// Pending are the variables (bonsaiai's, Hugging Face's, AWS's) the running engine has other
	// values of: it is stopped when no request goes on, and the next starts one with these.
	Pending []string `json:"pending,omitempty"`
	// SettingsError is what was wrong with a .env file.
	SettingsError string `json:"settings_error,omitempty"`
}

// Status is the answer of GET /bonsaiaid/v1/status.
type Status struct {
	Hello
	// Models are the chat models the engine holds for its chat routes, Held the models the apps'
	// sessions hold (one copy each, however many sessions use it).
	Models []ModelInfo `json:"models"`
	Held   []HeldModel `json:"held,omitempty"`
	// NPUBytes is what the NPU holds, 0 when it is not known: on the Mac the Neural Engine's wired
	// memory (the whole machine's), on Linux the engine's NPU buffers (pinned).
	NPUBytes int64 `json:"npu_bytes"`
	// Idle is how long the engine has had no request.
	Idle string `json:"idle,omitempty"`
}

// LoadRequest is the body of POST /bonsaiaid/v1/load: a chat model to make ready and load.
type LoadRequest struct {
	// Model is a name, a family or a use of models.yaml, or a Hugging Face repository.
	Model string `json:"model"`
	// Memory runs the model as on a machine of this many GB (bonsaiai-chat --memory); 0 for the one
	// chosen for it.
	Memory int `json:"memory,omitempty"`
	// Terminal says that the client shows the log on a terminal, where bars are redrawn.
	Terminal bool `json:"terminal,omitempty"`
	// Allowlist is the client's model policy (its BONSAIAI_MODEL_ALLOWLIST): it narrows bonsaiaid's,
	// never widens it. "" for none.
	Allowlist string `json:"allowlist,omitempty"`
}

// AllowlistHeader is a chat completion's model policy (LoadRequest.Allowlist).
const AllowlistHeader = "X-Bonsaiai-Model-Allowlist"

// ModelInfo is a chat model the engine holds.
type ModelInfo struct {
	Name       string `json:"name"`
	Context    int    `json:"context"`
	Images     bool   `json:"images,omitempty"`
	Audio      bool   `json:"audio,omitempty"`
	Translator bool   `json:"translator,omitempty"`
	NoThink    bool   `json:"no_think,omitempty"`
	OnNPU      bool   `json:"on_npu"`
	Device     string `json:"device,omitempty"`
}

// HeldModel is a model the apps' sessions hold in the engine (internal/aihost).
type HeldModel struct {
	Role  string `json:"role"`
	Model string `json:"model"`
	// Uses are the sessions (the apps' Runtimes) that hold it.
	Uses int `json:"uses"`
}

// LoadEvent is a line of the answer of POST /bonsaiaid/v1/load (JSON lines): some of the log, then
// the model or an error.
type LoadEvent struct {
	Log   string     `json:"log,omitempty"`
	Model *ModelInfo `json:"model,omitempty"`
	Error string     `json:"error,omitempty"`
}

// ThinkBudgetHeader carries bonsaiai-chat's --think-budget in a chat completion request.
const ThinkBudgetHeader = "X-Bonsaiai-Think-Budget"

// Client talks to bonsaiaid over its socket.
type Client struct {
	Socket string
	HTTP   *http.Client
}

// NewClient is a client of the socket (SocketPath when "").
func NewClient(socket string) *Client {
	if socket == "" {
		socket = SocketPath()
	}
	return &Client{Socket: socket, HTTP: &http.Client{Transport: UnixTransport(socket)}}
}

// UnixTransport is an HTTP transport whose connections all go to the socket.
func UnixTransport(socket string) *http.Transport {
	return &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			var d net.Dialer
			return d.DialContext(ctx, "unix", socket)
		},
		DisableCompression: true,
	}
}

// URL is the URL of a route on the socket.
func (c *Client) URL(path string) string { return "http://bonsaiaid" + path }

// ErrNotRunning says that nothing answers on the socket.
var ErrNotRunning = errors.New("bonsaiaid does not answer")

// Hello asks the daemon who it is.
func (c *Client) Hello(ctx context.Context) (Hello, error) {
	var h Hello
	err := c.getJSON(ctx, "/bonsaiaid/v1/hello", &h)
	return h, err
}

// Status asks the daemon what it holds.
func (c *Client) Status(ctx context.Context) (Status, error) {
	var s Status
	err := c.getJSON(ctx, "/bonsaiaid/v1/status", &s)
	return s, err
}

// Quit asks the supervisor to stop the engine and exit (the service manager starts it again at
// the next connection).
func (c *Client) Quit(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL("/bonsaiaid/v1/quit"), nil)
	if err != nil {
		return err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return c.notRunning(err)
	}
	resp.Body.Close()
	return nil
}

// Start starts the engine unless it runs (and the supervisor, when the service manager starts it
// at a connection), and answers hello.
func (c *Client) Start(ctx context.Context) (Hello, error) {
	return c.postHello(ctx, "/bonsaiaid/v1/start")
}

// Restart ends the supervisor and its engine (the requests going on fail) and waits until a new
// supervisor answers, which the service manager starts, or, when it was started by hand, it runs
// itself again; then it starts the engine. It answers the old supervisor's hello and the new's.
func (c *Client) Restart(ctx context.Context) (old, now Hello, err error) {
	if old, err = c.Hello(ctx); err != nil {
		return old, now, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL("/bonsaiaid/v1/restart"), nil)
	if err != nil {
		return old, now, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return old, now, c.notRunning(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		return old, now, fmt.Errorf("bonsaiaid %s does not restart (%s): bonsaiai service stop, then start", old.Build, resp.Status)
	}
	// it stops its engine (up to 30 seconds) and exits or runs itself again
	deadline := time.Now().Add(90 * time.Second)
	for {
		hctx, cancel := context.WithTimeout(ctx, 5*time.Second)
		h, err := c.Hello(hctx)
		cancel()
		if err == nil && (h.PID != old.PID || h.Started != old.Started) {
			break
		}
		if ctx.Err() != nil {
			return old, now, ctx.Err()
		}
		if time.Now().After(deadline) {
			return old, now, fmt.Errorf("bonsaiaid did not start again (pid %d before; its log: %s)", old.PID, LogWhere())
		}
		time.Sleep(200 * time.Millisecond)
	}
	now, err = c.Start(ctx)
	return old, now, err
}

// Reload has bonsaiaid apply the .env files now: when they changed what the engine runs with and no
// request goes on, it stops the engine (the next request starts one with them). Hello's Pending
// are what is left for later.
func (c *Client) Reload(ctx context.Context) (Hello, error) {
	return c.postHello(ctx, "/bonsaiaid/v1/reload")
}

func (c *Client) postHello(ctx context.Context, path string) (Hello, error) {
	var h Hello
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL(path), nil)
	if err != nil {
		return h, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return h, c.notRunning(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return h, ResponseError(resp)
	}
	return h, json.NewDecoder(resp.Body).Decode(&h)
}

func (c *Client) getJSON(ctx context.Context, path string, v any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.URL(path), nil)
	if err != nil {
		return err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return c.notRunning(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ResponseError(resp)
	}
	return json.NewDecoder(resp.Body).Decode(v)
}

func (c *Client) notRunning(err error) error {
	var op *net.OpError
	if errors.As(err, &op) {
		return fmt.Errorf("%w at %s: %v", ErrNotRunning, c.Socket, err)
	}
	return err
}

// AIPath is the route of the sessions of the ai package (an app's Plan, Runtime and models run
// in the engine: internal/aihost), upgraded from HTTP to AIProtocol.
const (
	AIPath     = "/bonsaiaid/v1/ai"
	AIProtocol = "bonsaiai-ai/1"
)

// DialAI opens a session of the ai package with the engine (airpc over the socket).
func (c *Client) DialAI(ctx context.Context) (*airpc.Session, error) {
	var d net.Dialer
	conn, err := d.DialContext(ctx, "unix", c.Socket)
	if err != nil {
		return nil, c.notRunning(err)
	}
	s, err := airpc.Dial(ctx, conn, AIPath, AIProtocol)
	if err != nil {
		return nil, fmt.Errorf("bonsaiaid at %s: %w", c.Socket, err)
	}
	return s, nil
}

// Load makes the model ready and loads it, writing the engine's log to log as it comes.
func (c *Client) Load(ctx context.Context, r LoadRequest, log io.Writer) (ModelInfo, error) {
	body, _ := json.Marshal(r)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL("/bonsaiaid/v1/load"), bytes.NewReader(body))
	if err != nil {
		return ModelInfo{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return ModelInfo{}, c.notRunning(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return ModelInfo{}, ResponseError(resp)
	}
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 64<<10), 16<<20)
	for sc.Scan() {
		var ev LoadEvent
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			return ModelInfo{}, fmt.Errorf("bonsaiaid: %w", err)
		}
		if ev.Log != "" && log != nil {
			io.WriteString(log, ev.Log)
		}
		if ev.Error != "" {
			return ModelInfo{}, errors.New(ev.Error)
		}
		if ev.Model != nil {
			return *ev.Model, nil
		}
	}
	if err := sc.Err(); err != nil {
		return ModelInfo{}, err
	}
	return ModelInfo{}, errors.New("bonsaiaid: the engine ended before the model was loaded (its log: " + LogWhere() + ")")
}

// ResponseError is the error of an answer that is not 200: its JSON error's message, else its body.
func ResponseError(resp *http.Response) error {
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	var e struct {
		Error any `json:"error"`
	}
	if json.Unmarshal(b, &e) == nil && e.Error != nil {
		switch v := e.Error.(type) {
		case string:
			return errors.New(v)
		case map[string]any:
			if m, ok := v["message"].(string); ok {
				return errors.New(m)
			}
		}
	}
	return fmt.Errorf("bonsaiaid: %s: %s", resp.Status, strings.TrimSpace(string(b)))
}

// Use says whether a client goes through bonsaiaid, by BONSAIAI_DAEMON (def when unset): on needs it to
// answer with this Protocol, auto takes it when it does. It returns the client when it does.
//
// The settings of where the models are and come from are bonsaiaid's (Settings). When this process's
// differ (a variable set only in the shell, or in a project's .env), on writes to log that
// bonsaiaid's are used, and auto runs here instead, with this process's own.
func Use(ctx context.Context, def string, log io.Writer) (*Client, error) {
	mode := Mode(def)
	if mode == "off" {
		return nil, nil
	}
	c := NewClient("")
	hctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	h, err := c.Hello(hctx)
	if err == nil && h.Protocol != Protocol {
		err = fmt.Errorf("bonsaiaid at %s speaks protocol %d, this client %d: bonsaiai service stop, then install the same version (go install ./cmd/bonsaiaid)", c.Socket, h.Protocol, Protocol)
	}
	if err != nil {
		if mode == "auto" {
			return nil, nil
		}
		if errors.Is(err, ErrNotRunning) {
			return nil, fmt.Errorf("%s=on: %w (go install ./cmd/bonsaiai ./cmd/bonsaiaid && bonsaiai service install, or %s=off)", ModeEnv, err, ModeEnv)
		}
		return nil, fmt.Errorf("%s=on: %w", ModeEnv, err)
	}
	if ms := Compare(localSettings(), h.Settings); len(ms) > 0 {
		var parts []string
		for _, m := range ms {
			parts = append(parts, m.String())
		}
		files := h.Files
		if len(files) == 0 {
			files = []string{filepath.Join(envfile.UserDir(), ".env")}
		}
		where := "bonsaiaid reads " + strings.Join(files, ", ") + ", not this shell"
		if mode == "auto" {
			fmt.Fprintf(log, "not through bonsaiaid, whose settings differ (%s; %s): running here\n", strings.Join(parts, "; "), where)
			return nil, nil
		}
		fmt.Fprintf(log, "through bonsaiaid with its settings, not these: %s (%s)\n", strings.Join(parts, "; "), where)
	}
	return c, nil
}
