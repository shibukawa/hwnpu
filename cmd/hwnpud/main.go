// hwnpud is the per-user process that owns the NPU and the models (docs/design/hwnpud.md):
// hwnpu-chat and the apps reach it over a Unix domain socket instead of running the Neural Engine
// in their own processes, so that one Core AI cache (under the name hwnpud) and one copy of a
// model on the Neural Engine serve them all.
//
//	go install ./cmd/hwnpud && hwnpud install   # a LaunchAgent: launchd starts it at the first connection
//	HWNPU_DAEMON=on hwnpu-chat "hello"          # hwnpu-chat through it
//	hwnpud status                               # what it holds
//	hwnpud stop                                 # end it now (launchd starts it again when needed)
//	hwnpud serve                                # run it by hand, in the foreground
//
// The supervisor (serve) holds the socket and starts the engine (the same executable with
// "engine"), which loads and runs the models and exits after HWNPU_DAEMON_IDLE (5m) without a
// request, giving the Neural Engine's memory and its lock back; the supervisor exits after that
// too when launchd started it, and when its executable has been replaced (go install).
//
// The service starts without the shell's variables: hwnpu's settings (HWNPU_MODEL_SOURCE and the
// like) go in ~/.config/hwnpu/.env, which hwnpud reads (install warns about those set only in the
// shell).
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/alecthomas/kong"
	"github.com/shibukawa/hwnpu/internal/daemon"
	"github.com/shibukawa/hwnpu/internal/envfile"
)

type cli struct {
	Serve     ServeCmd     `cmd:"" default:"1" help:"Run the supervisor: listen on the socket (or take launchd's) and start the engine for the requests (the command when none is named)."`
	Engine    EngineCmd    `cmd:"" hidden:"" help:"Run the engine (the supervisor starts it)."`
	Install   InstallCmd   `cmd:"" help:"Install the LaunchAgent: launchd starts hwnpud when a client connects."`
	Uninstall UninstallCmd `cmd:"" help:"Remove the LaunchAgent."`
	Status    StatusCmd    `cmd:"" help:"Show whether hwnpud runs and what it holds."`
	Stop      StopCmd      `cmd:"" help:"End hwnpud and its engine now (launchd starts it again at the next connection)."`
}

func main() {
	// Ctrl-C, launchd's bootout and the supervisor's stop: the engine closes its models and
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
	Socket string         `placeholder:"PATH" help:"The socket to listen on when launchd did not start it with one (default: HWNPU_DAEMON_SOCKET, else hwnpud.sock in the user cache directory)."`
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

// InstallCmd installs the LaunchAgent.
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
// .env file does: the service, started by launchd, does not get them.
func shellOnly() []string {
	fromFiles := map[string]bool{}
	for _, f := range envfile.Loaded {
		for _, name := range f.Set {
			fromFiles[name] = true
		}
	}
	var out []string
	for _, kv := range os.Environ() {
		name, _, _ := strings.Cut(kv, "=")
		if fromFiles[name] || strings.HasPrefix(name, "HWNPU_DAEMON") {
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

// UninstallCmd removes the LaunchAgent.
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
		fmt.Printf("installed: %s (%s)\n", exe, daemon.PlistPath())
	} else {
		fmt.Println("installed: no (hwnpud install)")
	}
	st, err := cl.Status(ctx)
	if err != nil {
		fmt.Printf("running: no (%v)\n", err)
		return nil
	}
	fmt.Printf("running: pid %d, build %s, protocol %d, %s\n", st.PID, st.Build, st.Protocol, cl.Socket)
	if st.Engine == 0 {
		fmt.Println("engine: not running (it starts at the next request)")
		return nil
	}
	fmt.Printf("engine: pid %d, idle %s", st.Engine, st.Idle)
	if st.NPUBytes > 0 {
		fmt.Printf(", the Neural Engine holds %.2f GiB", float64(st.NPUBytes)/(1<<30))
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
