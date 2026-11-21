// Package daemon is hwnpud, the per-user process that owns the NPU and the models
// (docs/design/hwnpud.md), and what its clients need of it: where its socket is, whether to use
// it (HWNPU_DAEMON), and a client over the socket.
//
// hwnpud is two processes. The supervisor (Supervisor) holds the socket that launchd (or the
// supervisor itself) listens on, answers hello and status, and passes the rest to the engine
// (Engine), the same executable run again with "engine", which loads and runs the models. The
// engine exits after it has been idle, giving the NPU's memory and its lock back; the supervisor
// starts it again at the next request.
//
// The protocol is HTTP over the Unix domain socket: hwnpud's own routes under /hwnpud/v1/ and the
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
	"runtime/debug"
	"strings"
	"time"
)

// Protocol is the version of the routes and messages; a client and a daemon of different
// protocols do not talk.
const Protocol = 1

// The variables a client reads.
const (
	// ModeEnv says whether to go through hwnpud: on (always, an error when it does not answer),
	// off (never), auto (when it answers with this Protocol).
	ModeEnv = "HWNPU_DAEMON"
	// SocketEnv is another socket, to try a hwnpud built from a worktree beside the installed one.
	SocketEnv = "HWNPU_DAEMON_SOCKET"
	// IdleEnv is how long the engine and the supervisor wait without a request before they exit
	// (a Go duration; 0 never).
	IdleEnv = "HWNPU_DAEMON_IDLE"
)

// DefaultIdle is how long the engine keeps the models without a request.
const DefaultIdle = 5 * time.Minute

// Label is the name of the service (launchd's label).
const Label = "com.github.shibukawa.hwnpud"

// SocketName is the name of the socket in the service's definition (launchd's Sockets key).
const SocketName = "hwnpud"

// SocketPath is the socket clients connect to: HWNPU_DAEMON_SOCKET, else hwnpud.sock in the
// user's cache directory (~/Library/Caches/hwnpu on a Mac). It does not follow HWNPU_CACHE: the
// service is started without the shell's variables, and every client must find the same socket.
func SocketPath() string {
	if p := strings.TrimSpace(os.Getenv(SocketEnv)); p != "" {
		return p
	}
	return DefaultSocketPath()
}

// DefaultSocketPath is the socket of the installed service.
func DefaultSocketPath() string {
	dir, err := os.UserCacheDir()
	if err != nil {
		dir = os.TempDir()
	}
	return filepath.Join(dir, "hwnpu", "hwnpud.sock")
}

// EngineSocketPath is the engine's own socket, beside the supervisor's.
func EngineSocketPath(socket string) string {
	return strings.TrimSuffix(socket, ".sock") + "-engine.sock"
}

// Idle is HWNPU_DAEMON_IDLE, else DefaultIdle.
func Idle() time.Duration {
	if s := strings.TrimSpace(os.Getenv(IdleEnv)); s != "" {
		if d, err := time.ParseDuration(s); err == nil && d >= 0 {
			return d
		}
	}
	return DefaultIdle
}

// Mode is HWNPU_DAEMON: "on", "auto" or "off" (anything else, and unset, is def).
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

// Hello is the answer of GET /hwnpud/v1/hello.
type Hello struct {
	Protocol int    `json:"protocol"`
	Build    string `json:"build"`
	PID      int    `json:"pid"`
	// Engine is the engine's process, 0 when it is not running.
	Engine int `json:"engine"`
	// Executable is the supervisor's executable.
	Executable string `json:"executable"`
}

// Status is the answer of GET /hwnpud/v1/status.
type Status struct {
	Hello
	// Models are the models the engine holds.
	Models []ModelInfo `json:"models"`
	// NPUBytes is what the Neural Engine holds (the whole machine's), 0 when it is not known.
	NPUBytes int64 `json:"npu_bytes"`
	// Idle is how long the engine has had no request.
	Idle string `json:"idle,omitempty"`
}

// LoadRequest is the body of POST /hwnpud/v1/load: a chat model to make ready and load.
type LoadRequest struct {
	// Model is a name, a family or a use of models.yaml, or a Hugging Face repository.
	Model string `json:"model"`
	// Memory runs the model as on a machine of this many GB (hwnpu-chat --memory); 0 for the one
	// chosen for it.
	Memory int `json:"memory,omitempty"`
	// Terminal says that the client shows the log on a terminal, where bars are redrawn.
	Terminal bool `json:"terminal,omitempty"`
}

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

// LoadEvent is a line of the answer of POST /hwnpud/v1/load (JSON lines): some of the log, then
// the model or an error.
type LoadEvent struct {
	Log   string     `json:"log,omitempty"`
	Model *ModelInfo `json:"model,omitempty"`
	Error string     `json:"error,omitempty"`
}

// ThinkBudgetHeader carries hwnpu-chat's --think-budget in a chat completion request.
const ThinkBudgetHeader = "X-Hwnpu-Think-Budget"

// Client talks to hwnpud over its socket.
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
func (c *Client) URL(path string) string { return "http://hwnpud" + path }

// ErrNotRunning says that nothing answers on the socket.
var ErrNotRunning = errors.New("hwnpud does not answer")

// Hello asks the daemon who it is.
func (c *Client) Hello(ctx context.Context) (Hello, error) {
	var h Hello
	err := c.getJSON(ctx, "/hwnpud/v1/hello", &h)
	return h, err
}

// Status asks the daemon what it holds.
func (c *Client) Status(ctx context.Context) (Status, error) {
	var s Status
	err := c.getJSON(ctx, "/hwnpud/v1/status", &s)
	return s, err
}

// Quit asks the supervisor to stop the engine and exit (launchd starts it again at the next
// connection).
func (c *Client) Quit(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL("/hwnpud/v1/quit"), nil)
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

// Load makes the model ready and loads it, writing the engine's log to log as it comes.
func (c *Client) Load(ctx context.Context, r LoadRequest, log io.Writer) (ModelInfo, error) {
	body, _ := json.Marshal(r)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL("/hwnpud/v1/load"), bytes.NewReader(body))
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
			return ModelInfo{}, fmt.Errorf("hwnpud: %w", err)
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
	return ModelInfo{}, errors.New("hwnpud: the engine ended before the model was loaded (its log: " + LogPath() + ")")
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
	return fmt.Errorf("hwnpud: %s: %s", resp.Status, strings.TrimSpace(string(b)))
}

// Use says whether a client goes through hwnpud, by HWNPU_DAEMON (def when unset): on needs it to
// answer with this Protocol, auto takes it when it does. It returns the client when it does.
func Use(ctx context.Context, def string) (*Client, error) {
	mode := Mode(def)
	if mode == "off" {
		return nil, nil
	}
	c := NewClient("")
	hctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	h, err := c.Hello(hctx)
	if err == nil && h.Protocol != Protocol {
		err = fmt.Errorf("hwnpud at %s speaks protocol %d, this client %d: hwnpud stop, then install the same version (go install ./cmd/hwnpud)", c.Socket, h.Protocol, Protocol)
	}
	if err != nil {
		if mode == "auto" {
			return nil, nil
		}
		if errors.Is(err, ErrNotRunning) {
			return nil, fmt.Errorf("%s=on: %w (go install ./cmd/hwnpud && hwnpud install, or %s=off)", ModeEnv, err, ModeEnv)
		}
		return nil, fmt.Errorf("%s=on: %w", ModeEnv, err)
	}
	return c, nil
}

// LogPath is where the service writes its log.
func LogPath() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "Library", "Logs", "hwnpud.log")
}
