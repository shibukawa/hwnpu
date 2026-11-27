// hwnpud is the per-user process that owns the NPU and the models (docs/design/hwnpud.md):
// hwnpu-chat, hwnpu's commands and the apps of the ai package reach it over a Unix domain socket
// instead of running the Neural Engine in their own processes (HWNPU_DAEMON is on unless set
// off), so that one Core AI cache (under the name hwnpud) and one copy of a model on the Neural
// Engine serve them all.
//
// It is what the service manager (launchd, systemd) runs; it is looked after with hwnpu service
// (internal/servicecmd):
//
//	go install ./cmd/hwnpu ./cmd/hwnpud
//	hwnpu service install    # a LaunchAgent (Mac) or systemd user units (Linux): started at the first connection
//	hwnpu service status     # what it holds
//	hwnpu service restart    # a new executable, the settings as they are now
//	hwnpu service start      # not installed: run it here, in the foreground (hwnpud serve)
//
// The supervisor (serve) holds the socket and starts the engine (the same executable with
// "engine"), which loads and runs the models and exits after HWNPU_DAEMON_IDLE (5m) without a
// request, giving the NPU's memory and its lock back; the supervisor exits after that too when
// launchd or systemd started it, and when its executable has been replaced (go install).
//
// The service starts without the shell's variables: hwnpu's settings (model-source and the like)
// are kept in ~/.config/hwnpu/.env, which hwnpu config writes and hwnpud reads. On Linux,
// systemd's user manager also gives it ~/.config/environment.d's. A change of the files is
// applied without a restart: the supervisor reads them every few seconds and stops the engine
// when it has no request, and the next request starts one with them.
package main

import (
	"context"
	"errors"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/alecthomas/kong"
	"github.com/shibukawa/hwnpu/internal/aihost"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/daemon"
	"github.com/shibukawa/hwnpu/internal/daemon/engine"
	"github.com/shibukawa/hwnpu/internal/envfile"
)

type cli struct {
	Serve  ServeCmd  `cmd:"" default:"1" help:"Run the supervisor: listen on the socket (or take launchd's or systemd's) and start the engine for the requests (the command when none is named). hwnpu service looks after it."`
	Engine EngineCmd `cmd:"" hidden:"" help:"Run the engine (the supervisor starts it)."`
}

func main() {
	// Ctrl-C, launchd's bootout, systemd's stop and the supervisor's stop: the engine closes its models and
	// removes its socket, the supervisor stops the engine first
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	var c cli
	k := kong.Parse(&c,
		kong.Name("hwnpud"),
		kong.Description("The per-user process that owns the NPU and the models, for hwnpu-chat and the apps (docs/design/hwnpud.md); hwnpu service installs, starts, stops and shows it."),
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
	err := s.Run(ctx)
	if errors.Is(err, daemon.ErrRestart) {
		// hwnpu service restart of a supervisor run by hand: the executable as it is now (go
		// install), in this process (the terminal it runs in), reading the .env files again
		exe, xerr := os.Executable()
		if xerr != nil {
			return xerr
		}
		return syscall.Exec(exe, os.Args, envfile.Original())
	}
	return err
}

// EngineCmd runs the engine.
type EngineCmd struct {
	Socket string         `required:"" placeholder:"PATH" help:"The engine's socket."`
	Idle   *time.Duration `help:"How long it keeps the models without a request before it exits (0: always)."`
}

func (c *EngineCmd) Run(ctx context.Context) error {
	// the engine runs the models itself: its apps' (internal/aihost) are not sent on to hwnpud
	os.Setenv(daemon.ModeEnv, "off")
	e := &engine.Engine{Idle: idle(c.Idle), Log: os.Stderr, Held: aihost.Held}
	o := aihost.Options{Log: os.Stderr, Held: e.HeldDirs}
	e.AI = o.Handler()
	// what the cache keeps of the engine's are the models it holds loaded, not all it ever used
	cache.SetInUse(o.UnitInUse)
	return e.Run(ctx, c.Socket)
}

func idle(d *time.Duration) time.Duration {
	if d != nil {
		return *d
	}
	return daemon.Idle()
}
