// Package daemon is hwnpud, the per-user process that owns the NPU and the models
// (docs/design/hwnpud.md), and what its clients need of it: where its socket is, whether to use
// it (HWNPU_DAEMON), and a client over the socket.
//
// hwnpud is two processes. The supervisor (Supervisor) holds the socket that the service manager
// (launchd, systemd) or the supervisor itself listens on, answers hello and status, and passes
// the rest to the engine (Engine), the same executable run again with "engine", which loads and
// runs the models. The
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
	"crypto/sha256"
	"encoding/hex"
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
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/envfile"
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

// Label is the name of the service on the Mac (launchd's label).
const Label = "com.github.shibukawa.hwnpud"

// SocketName is the name of the socket in the service's definition (launchd's Sockets key,
// systemd's FileDescriptorName).
const SocketName = "hwnpud"

// SocketPath is the socket clients connect to: HWNPU_DAEMON_SOCKET, else DefaultSocketPath. It does not follow HWNPU_CACHE: the
// service is started without the shell's variables, and every client must find the same socket.
func SocketPath() string {
	if p := strings.TrimSpace(os.Getenv(SocketEnv)); p != "" {
		return p
	}
	return DefaultSocketPath()
}

// DefaultSocketPath is the socket of the installed service: hwnpud.sock in XDG_RUNTIME_DIR on
// Linux (where systemd's user units listen), else in the user's cache directory.
func DefaultSocketPath() string {
	if runtime.GOOS == "linux" {
		if d := os.Getenv("XDG_RUNTIME_DIR"); d != "" {
			return filepath.Join(d, "hwnpud.sock")
		}
	}
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

// DefaultMode is what HWNPU_DAEMON is when it is not set: on (every program goes through
// hwnpud, and fails when it does not answer), except in a test (go test), which runs in its own
// process. hwnpud's engine sets it off for itself.
func DefaultMode() string {
	if testing.Testing() {
		return "off"
	}
	return "on"
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
	// Definitions is the hash of the model definitions and recipes hwnpud was built with
	// (Definitions): what decides the models' exports and how they run. A client of other ones
	// says so; "" from a hwnpud from before it told them.
	Definitions string `json:"definitions,omitempty"`
	// Roles are the settings each role of the engine takes (ai.LoadSettings, airole.Role's
	// LoadSettings): a client stops before sending one hwnpud does not know.
	Roles map[string][]string `json:"roles,omitempty"`
	// Engine is the engine's process, 0 when it is not running.
	Engine int `json:"engine"`
	// Executable is the supervisor's executable.
	Executable string `json:"executable"`
	// Service is the service manager that started the supervisor (launchd, systemd), "" when it
	// was started by hand (hwnpud serve): one that quits is started again only by the former.
	Service string `json:"service,omitempty"`
	// Started is when the supervisor started (unix nanoseconds): a restart by hand keeps the
	// process, and this tells the new from the old.
	Started int64 `json:"started,omitempty"`
	// Settings are hwnpud's settings (Settings): those the next engine starts with, read from the
	// .env files now.
	Settings map[string]string `json:"settings,omitempty"`
	// Files are the .env files they were read from.
	Files []string `json:"files,omitempty"`
	// Pending are the variables (hwnpu's, Hugging Face's, AWS's) the running engine has other
	// values of: it is stopped when no request goes on, and the next starts one with these.
	Pending []string `json:"pending,omitempty"`
	// SettingsError is what was wrong with a .env file.
	SettingsError string `json:"settings_error,omitempty"`
}

// Definitions is the hash of the model definitions and recipes built in (models.yaml and
// model-recipes.yaml, Builtin), 12 hex digits; "" without them.
func Definitions() string {
	if len(Builtin) == 0 {
		return ""
	}
	h := sha256.New()
	for _, b := range Builtin {
		h.Write(b)
		h.Write([]byte{0})
	}
	return hex.EncodeToString(h.Sum(nil))[:12]
}

// Builtin are the model definitions and recipes this process was built with (hwnpu.ModelsYAML and
// hwnpu.RecipesYAML), set by the package ai, which has them: this package stays without the
// engine's.
var Builtin [][]byte

// RoleSettings gives the settings of the engine's roles (Hello.Roles): set by hwnpud, whose engine
// has the roles (internal/aihost); nil in a client.
var RoleSettings func() map[string][]string

// Base is the Hello of this process's version.
func Base() Hello {
	h := Hello{Protocol: Protocol, Build: Build(), PID: os.Getpid(), Definitions: Definitions()}
	if RoleSettings != nil {
		h.Roles = RoleSettings()
	}
	return h
}

// Status is the answer of GET /hwnpud/v1/status.
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

// LoadRequest is the body of POST /hwnpud/v1/load: a chat model to make ready and load.
type LoadRequest struct {
	// Model is a name, a family or a use of models.yaml, or a Hugging Face repository.
	Model string `json:"model"`
	// Memory runs the model as on a machine of this many GB (hwnpu-chat --memory); 0 for the one
	// chosen for it.
	Memory int `json:"memory,omitempty"`
	// Terminal says that the client shows the log on a terminal, where bars are redrawn.
	Terminal bool `json:"terminal,omitempty"`
	// Allowlist is the client's model policy (its HWNPU_MODEL_ALLOWLIST): it narrows hwnpud's,
	// never widens it. "" for none.
	Allowlist string `json:"allowlist,omitempty"`
}

// AllowlistHeader is a chat completion's model policy (LoadRequest.Allowlist).
const AllowlistHeader = "X-Hwnpu-Model-Allowlist"

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
	// Server is what hwnpud answered Use's hello with.
	Server Hello
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
	return doJSON[Hello](c, ctx, http.MethodGet, "/hwnpud/v1/hello")
}

// Status asks the daemon what it holds.
func (c *Client) Status(ctx context.Context) (Status, error) {
	return doJSON[Status](c, ctx, http.MethodGet, "/hwnpud/v1/status")
}

// Quit asks the supervisor to stop the engine and exit (the service manager starts it again at
// the next connection).
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

// Start starts the engine unless it runs (and the supervisor, when the service manager starts it
// at a connection), and answers hello.
func (c *Client) Start(ctx context.Context) (Hello, error) {
	return doJSON[Hello](c, ctx, http.MethodPost, "/hwnpud/v1/start")
}

// Restart ends the supervisor and its engine (the requests going on fail) and waits until a new
// supervisor answers, which the service manager starts, or, when it was started by hand, it runs
// itself again; then it starts the engine. It answers the old supervisor's hello and the new's.
func (c *Client) Restart(ctx context.Context) (old, now Hello, err error) {
	if old, err = c.Hello(ctx); err != nil {
		return old, now, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.URL("/hwnpud/v1/restart"), nil)
	if err != nil {
		return old, now, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return old, now, c.notRunning(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusNoContent {
		return old, now, fmt.Errorf("hwnpud %s does not restart (%s): hwnpu service stop, then start", old.Build, resp.Status)
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
			return old, now, fmt.Errorf("hwnpud did not start again (pid %d before; its log: %s)", old.PID, LogWhere())
		}
		time.Sleep(200 * time.Millisecond)
	}
	now, err = c.Start(ctx)
	return old, now, err
}

// Reload has hwnpud apply the .env files now: when they changed what the engine runs with and no
// request goes on, it stops the engine (the next request starts one with them). Hello's Pending
// are what is left for later.
func (c *Client) Reload(ctx context.Context) (Hello, error) {
	return doJSON[Hello](c, ctx, http.MethodPost, "/hwnpud/v1/reload")
}

func doJSON[T any](c *Client, ctx context.Context, method, path string) (T, error) {
	var out T
	req, err := http.NewRequestWithContext(ctx, method, c.URL(path), nil)
	if err != nil {
		return out, err
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return out, c.notRunning(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return out, ResponseError(resp)
	}
	return out, json.NewDecoder(resp.Body).Decode(&out)
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
	AIPath     = "/hwnpud/v1/ai"
	AIProtocol = "hwnpu-ai/1"
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
		return nil, fmt.Errorf("hwnpud at %s: %w", c.Socket, err)
	}
	return s, nil
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
	return ModelInfo{}, errors.New("hwnpud: the engine ended before the model was loaded (its log: " + LogWhere() + ")")
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
//
// The settings of where the models are and come from are hwnpud's (Settings). When this process's
// differ (a variable set only in the shell, or in a project's .env), on writes to log that
// hwnpud's are used, and auto runs here instead, with this process's own.
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
		err = fmt.Errorf("hwnpud at %s speaks protocol %d, this client %d: hwnpu service stop, then install the same version (go install ./cmd/hwnpud)", c.Socket, h.Protocol, Protocol)
	}
	if err != nil {
		if mode == "auto" {
			return nil, nil
		}
		if errors.Is(err, ErrNotRunning) {
			return nil, fmt.Errorf("%s=on: %w (go install ./cmd/hwnpu ./cmd/hwnpud && hwnpu service install, or %s=off)", ModeEnv, err, ModeEnv)
		}
		return nil, fmt.Errorf("%s=on: %w", ModeEnv, err)
	}
	c.Server = h
	if note := VersionNote(h); note != "" {
		fmt.Fprintln(log, note)
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
		where := "hwnpud reads " + strings.Join(files, ", ") + ", not this shell"
		if mode == "auto" {
			fmt.Fprintf(log, "not through hwnpud, whose settings differ (%s; %s): running here\n", strings.Join(parts, "; "), where)
			return nil, nil
		}
		fmt.Fprintf(log, "through hwnpud with its settings, not these: %s (%s)\n", strings.Join(parts, "; "), where)
	}
	return c, nil
}

// VersionNote is what a client says of hwnpud's version, "" when its definitions are this
// client's: a hwnpud left running (or installed) from an earlier build decides the exports and
// how the models run by its own definitions, and silently, unless told.
func VersionNote(h Hello) string {
	const fix = "go install ./cmd/hwnpud && hwnpu service restart, or HWNPU_DAEMON=off"
	switch {
	case Definitions() == "":
		return ""
	case h.Definitions == "":
		return fmt.Sprintf("hwnpud (build %s) is older than this client (build %s): it does not tell its model definitions and settings (%s)", h.Build, Build(), fix)
	case h.Definitions != Definitions():
		return fmt.Sprintf("hwnpud (build %s) has other model definitions and recipes than this client (build %s): the models run as hwnpud's say (%s)", h.Build, Build(), fix)
	}
	return ""
}

// CheckSettings is nil when hwnpud's engine takes every setting of settings (by role, as
// ai.LoadSettings), as its Hello says; a role it does not tell is not checked.
func (h Hello) CheckSettings(settings map[string]map[string]string) error {
	for role, set := range settings {
		names, ok := h.Roles[role]
		if !ok {
			continue
		}
		var unknown []string
		for k := range set {
			if !slices.Contains(names, k) {
				unknown = append(unknown, k)
			}
		}
		if len(unknown) > 0 {
			slices.Sort(unknown)
			why := "it takes " + strings.Join(names, ", ")
			if h.Definitions != Definitions() {
				why += "; it is not this client's version: go install ./cmd/hwnpud && hwnpu service restart"
			}
			return fmt.Errorf("hwnpud (build %s) takes no setting %s of %s (%s)", h.Build, strings.Join(unknown, ", "), role, why)
		}
	}
	return nil
}

// LoadStream writes a load's log to its client as LoadEvent lines (POST /hwnpud/v1/load's
// answer, which Client.Load reads). A client that went away fails no load: what it loads stays
// for the next.
type LoadStream struct {
	mu sync.Mutex
	w  http.ResponseWriter
	rc *http.ResponseController
	ok bool
}

// NewLoadStream starts the answer of a load on w.
func NewLoadStream(w http.ResponseWriter) *LoadStream {
	f := &LoadStream{w: w, rc: http.NewResponseController(w), ok: true}
	f.rc.Flush()
	return f
}

// Send writes ev.
func (f *LoadStream) Send(ev LoadEvent) {
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

// Write sends p as a line of the log.
func (f *LoadStream) Write(p []byte) (int, error) {
	f.Send(LoadEvent{Log: string(p)})
	return len(p), nil
}
