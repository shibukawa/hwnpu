// hwnpud is the per-user process that owns the NPU and the models (docs/design/hwnpud.md):
// hwnpu-chat and the apps reach it over a Unix domain socket instead of running the Neural Engine
// in their own processes, so that one Core AI cache (under the name hwnpud) and one copy of a
// model on the Neural Engine serve them all.
//
//	go install ./cmd/hwnpud && hwnpud install   # a LaunchAgent (Mac) or systemd user units (Linux): started at the first connection
//	HWNPU_DAEMON=on hwnpu-chat "hello"          # hwnpu-chat through it
//	hwnpud status                               # what it holds
//	hwnpud stop                                 # end it now (the service manager starts it again when needed)
//	hwnpud serve                                # run it by hand, in the foreground
//
// The supervisor (serve) holds the socket and starts the engine (the same executable with
// "engine"), which loads and runs the models and exits after HWNPU_DAEMON_IDLE (5m) without a
// request, giving the NPU's memory and its lock back; the supervisor exits after that too when
// launchd or systemd started it, and when its executable has been replaced (go install).
//
// The service starts without the shell's variables: hwnpu's settings (HWNPU_MODEL_SOURCE and the
// like) go in ~/.config/hwnpu/.env, which hwnpud reads (install warns about those set only in the
// shell). On Linux, systemd's user manager also gives it ~/.config/environment.d's.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/alecthomas/kong"
	"github.com/shibukawa/hwnpu/internal/daemon"
	"github.com/shibukawa/hwnpu/internal/envfile"
)

type cli struct {
	Serve     ServeCmd     `cmd:"" default:"1" help:"Run the supervisor: listen on the socket (or take launchd's or systemd's) and start the engine for the requests (the command when none is named)."`
	Engine    EngineCmd    `cmd:"" hidden:"" help:"Run the engine (the supervisor starts it)."`
	Install   InstallCmd   `cmd:"" help:"Install the service (a LaunchAgent on the Mac, systemd user units on Linux): it starts hwnpud when a client connects."`
	Uninstall UninstallCmd `cmd:"" help:"Remove the service."`
	Status    StatusCmd    `cmd:"" help:"Show whether hwnpud runs and what it holds."`
	Stop      StopCmd      `cmd:"" help:"End hwnpud and its engine now (the service manager starts it again at the next connection)."`
}

func main() {
	// Ctrl-C, launchd's bootout, systemd's stop and the supervisor's stop: the engine closes its models and
	// removes its socket, the supervisor stops the engine first
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	var c cli
	k := kong.Parse(&c,
		kong.Name("hwnpud"),
		kong.Description("The per-user process that owns the NPU and the models, for hwnpu-chat and the apps (docs/design/hwnpud.md)."),
		kong.UsageOnError(),
		kong.BindTo(ctx, (*context.Context)(nil)))
	k.FatalIfErrorf(k.Run())
}

// ServeCmd runs the supervisor.
type ServeCmd struct {
	Socket string         `placeholder:"PATH" help:"The socket to listen on when the service manager did not start it with one (default: HWNPU_DAEMON_SOCKET, else hwnpud.sock in XDG_RUNTIME_DIR on Linux, in the user cache directory on the Mac)."`
	Idle   *time.Duration `help:"How long the engine keeps the models without a request (0: always) (default: HWNPU_DAEMON_IDLE, else 5m)."`
}

func (c *ServeCmd) Run(ctx context.Context) error {
	s := &daemon.Supervisor{Socket: c.Socket, Idle: idle(c.Idle)}
	return s.Run(ctx)
}

// EngineCmd runs the engine.
type EngineCmd struct {
	Socket string         `required:"" placeholder:"PATH" help:"The engine's socket."`
	Idle   *time.Duration `help:"How long it keeps the models without a request before it exits (0: always)."`
}

func (c *EngineCmd) Run(ctx context.Context) error {
	e := &daemon.Engine{Idle: idle(c.Idle)}
	return e.Run(ctx, c.Socket)
}

func idle(d *time.Duration) time.Duration {
	if d != nil {
		return *d
	}
	return daemon.Idle()
}

// InstallCmd installs the service.
type InstallCmd struct{}

func (c *InstallCmd) Run() error {
	if err := daemon.Install("", os.Stdout); err != nil {
		return err
	}
	if unseen := shellOnly(); len(unseen) > 0 {
		fmt.Fprintf(os.Stderr, "hwnpud does not see what only the shell sets: %s. Put them in %s for hwnpud to read them.\n",
			strings.Join(unseen, ", "), filepath.Join(envfile.UserDir(), ".env"))
	}
	return nil
}

// shellOnly are hwnpu's variables (and Hugging Face's and AWS's) the environment sets that no
// .env file does and the service manager does not give the service either (launchd gives none
// of them, systemd those of environment.d): the service does not get them.
func shellOnly() []string {
	fromFiles := map[string]bool{}
	for _, f := range envfile.Loaded {
		for _, name := range f.Set {
			fromFiles[name] = true
		}
	}
	service := map[string]bool{}
	for _, kv := range daemon.ServiceEnviron() {
		service[kv] = true
	}
	var out []string
	for _, kv := range os.Environ() {
		name, _, _ := strings.Cut(kv, "=")
		if fromFiles[name] || service[kv] || strings.HasPrefix(name, "HWNPU_DAEMON") {
			continue
		}
		for _, p := range envfile.Prefixes {
			if strings.HasPrefix(name, p) {
				out = append(out, name)
				break
			}
		}
	}
	sort.Strings(out)
	return out
}

// printSettings shows hwnpud's settings, where they were read from, and what differs here.
func printSettings(h daemon.Hello) {
	files := "none"
	if len(h.Files) > 0 {
		files = strings.Join(h.Files, ", ")
	}
	fmt.Printf("settings: from %s and the environment hwnpud was started with\n", files)
	if h.SettingsError != "" {
		fmt.Printf("  error: %s\n", h.SettingsError)
	}
	names := make([]string, 0, len(h.Settings))
	for k := range h.Settings {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, k := range names {
		fmt.Printf("  %s=%s\n", k, h.Settings[k])
	}
	if _, set := h.Settings["HWNPU_CACHE"]; !set {
		fmt.Println("  HWNPU_CACHE unset: the models are in hwnpu in the user cache directory")
	}
	if len(h.Pending) > 0 {
		fmt.Printf("  the engine still runs with the old %s: it starts again with these at the next request\n", strings.Join(h.Pending, ", "))
	}
	for _, m := range daemon.Compare(daemon.Settings(os.Environ()), h.Settings) {
		fmt.Printf("  differs here: %s (hwnpud's is used)\n", m)
	}
}

// UninstallCmd removes the service.
type UninstallCmd struct{}

func (c *UninstallCmd) Run(ctx context.Context) error {
	daemon.NewClient(daemon.DefaultSocketPath()).Quit(ctx)
	return daemon.Uninstall(os.Stdout)
}

// StatusCmd shows the daemon.
type StatusCmd struct{}

func (c *StatusCmd) Run(ctx context.Context) error {
	cl := daemon.NewClient("")
	if exe := daemon.Installed(); exe != "" {
		fmt.Printf("installed: %s (%s)\n", exe, daemon.ServiceFile())
	} else {
		fmt.Println("installed: no (hwnpud install)")
	}
	st, err := cl.Status(ctx)
	if err != nil {
		fmt.Printf("running: no (%v)\n", err)
		return nil
	}
	fmt.Printf("running: pid %d, build %s, protocol %d, %s\n", st.PID, st.Build, st.Protocol, cl.Socket)
	printSettings(st.Hello)
	if st.Engine == 0 {
		fmt.Println("engine: not running (it starts at the next request)")
		return nil
	}
	fmt.Printf("engine: pid %d, idle %s", st.Engine, st.Idle)
	if st.NPUBytes > 0 {
		if runtime.GOOS == "darwin" {
			fmt.Printf(", the Neural Engine holds %.2f GiB", float64(st.NPUBytes)/(1<<30))
		} else {
			fmt.Printf(", its NPU buffers %.2f GiB (pinned)", float64(st.NPUBytes)/(1<<30))
		}
	}
	fmt.Println()
	for _, m := range st.Models {
		where := "all on the NPU"
		if !m.OnNPU {
			where = "NOT all on the NPU"
		}
		fmt.Printf("  %s: context %d tokens, %s (%s)\n", m.Name, m.Context, where, m.Device)
	}
	return nil
}

// StopCmd ends the daemon.
type StopCmd struct{}

func (c *StopCmd) Run(ctx context.Context) error {
	cl := daemon.NewClient("")
	if err := cl.Quit(ctx); err != nil {
		return err
	}
	fmt.Println("hwnpud stopped")
	return nil
}
