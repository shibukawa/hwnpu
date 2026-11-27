// Package servicecmd is hwnpu service: the commands that look after hwnpud (docs/design/hwnpud.md),
// the per-user process that owns the NPU and the models. hwnpud itself is what the service manager
// runs (serve, engine); the commands for the user are hwnpu's.
//
//	hwnpu service install    # a LaunchAgent (Mac) or systemd user units (Linux): hwnpud starts at the first connection
//	hwnpu service status     # whether it runs, its settings, what it holds
//	hwnpu service start      # installed: start it now; not: run hwnpud serve here, in the foreground
//	hwnpu service stop       # end it (installed: it starts again at the next connection)
//	hwnpu service restart    # end it and start it again: a new executable, the settings as they are now
//	hwnpu service uninstall
//
// A hwnpud run in the foreground (start without install, hwnpud serve) is stopped and restarted
// from another terminal the same way: it runs itself again in its own process.
package servicecmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"sort"
	"strings"
	"syscall"

	"github.com/shibukawa/hwnpu/internal/daemon"
	"github.com/shibukawa/hwnpu/internal/envfile"
)

// Cmd is hwnpu service.
type Cmd struct {
	Status    StatusCmd    `cmd:"" default:"1" help:"Whether hwnpud is installed and runs, its settings, and what it holds (the command when none is named)."`
	Install   InstallCmd   `cmd:"" help:"Install hwnpud as a service (a LaunchAgent on the Mac, systemd user units on Linux): it starts when a client connects, and exits when idle."`
	Uninstall UninstallCmd `cmd:"" help:"Stop hwnpud and remove the service."`
	Start     StartCmd     `cmd:"" help:"Installed: start hwnpud and its engine now. Not installed: run hwnpud here, in the foreground (Ctrl-C, or hwnpu service stop in another terminal, ends it)."`
	Stop      StopCmd      `cmd:"" help:"End hwnpud and its engine (installed: the service manager starts it again at the next connection)."`
	Restart   RestartCmd   `cmd:"" help:"End hwnpud and its engine (the requests going on fail) and start them again, with the executable and the settings as they are now."`
}

// Hwnpud is hwnpud's executable: this one when it is hwnpud, else the hwnpud beside it, else the
// one on PATH.
func Hwnpud() (string, error) {
	self, err := os.Executable()
	if err != nil {
		return "", err
	}
	if strings.TrimSuffix(filepath.Base(self), ".exe") == "hwnpud" {
		return self, nil
	}
	if p := filepath.Join(filepath.Dir(self), "hwnpud"); fileExists(p) {
		return p, nil
	}
	if p, err := exec.LookPath("hwnpud"); err == nil {
		return filepath.Abs(p)
	}
	return "", errors.New("no hwnpud beside " + self + " or on PATH: go install ./cmd/hwnpud")
}

// installed says whether the service listens on the client's socket: the installed service's,
// not another of HWNPU_DAEMON_SOCKET (a hwnpud of a worktree, run by hand beside it).
func installed(cl *daemon.Client) bool {
	return daemon.Installed() != "" && cl.Socket == daemon.DefaultSocketPath()
}

func fileExists(p string) bool {
	fi, err := os.Stat(p)
	return err == nil && !fi.IsDir()
}

// InstallCmd installs the service.
type InstallCmd struct{}

func (c *InstallCmd) Run() error {
	exe, err := Hwnpud()
	if err != nil {
		return err
	}
	if err := daemon.Install(exe, os.Stdout); err != nil {
		return err
	}
	if unseen := shellOnly(); len(unseen) > 0 {
		fmt.Fprintf(os.Stderr, "hwnpud does not see what only the shell sets: %s. hwnpu config sets them for hwnpud too.\n", strings.Join(unseen, ", "))
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
				if d := envfile.Display(name); !slices.Contains(out, d) {
					out = append(out, d)
				}
				break
			}
		}
	}
	sort.Strings(out)
	return out
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
		fmt.Printf("installed: %s (%s)", exe, daemon.ServiceFile())
		if !installed(cl) {
			fmt.Printf(", on %s, not this socket", daemon.DefaultSocketPath())
		}
		fmt.Println()
	} else {
		fmt.Println("installed: no (hwnpu service install; hwnpu service start runs it in the foreground)")
	}
	st, err := cl.Status(ctx)
	if err != nil {
		fmt.Printf("running: no (%v)\n", err)
		return nil
	}
	how := "in the foreground"
	if st.Service != "" {
		how = "started by " + st.Service
	}
	fmt.Printf("running: pid %d, %s, build %s, protocol %d, %s\n", st.PID, how, st.Build, st.Protocol, cl.Socket)
	if note := daemon.VersionNote(st.Hello); note != "" {
		fmt.Println("note: " + note)
	}
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
	for _, m := range st.Held {
		fmt.Printf("  %s (%s): loaded once, for %d %s\n", m.Model, m.Role, m.Uses, map[bool]string{true: "client", false: "clients"}[m.Uses == 1])
	}
	for _, m := range st.Models {
		where := "all on the NPU"
		if !m.OnNPU {
			where = "NOT all on the NPU"
		}
		fmt.Printf("  %s: context %d tokens, %s (%s)\n", m.Name, m.Context, where, m.Device)
	}
	return nil
}

// printSettings shows hwnpud's settings by their keys, where they were read from, and what
// differs here.
func printSettings(h daemon.Hello) {
	files := "none"
	if len(h.Files) > 0 {
		files = strings.Join(h.Files, ", ")
	}
	fmt.Printf("settings: from %s and the environment hwnpud was started with (hwnpu config changes them)\n", files)
	if h.SettingsError != "" {
		fmt.Printf("  error: %s\n", h.SettingsError)
	}
	for _, k := range envfile.Keys {
		items, set := k.Get(h.Settings)
		switch {
		case !set:
		case k.List():
			fmt.Printf("  %s = %s\n", k.Name, strings.Join(items, ", "))
		default:
			fmt.Printf("  %s = %s\n", k.Name, items[0])
		}
	}
	var other []string
	for name := range h.Settings {
		if _, ok := envfile.KeyOf(name); !ok {
			other = append(other, name)
		}
	}
	sort.Strings(other)
	for _, name := range other {
		fmt.Printf("  %s=%s\n", name, h.Settings[name])
	}
	if _, set := h.Settings["HWNPU_CACHE"]; !set {
		fmt.Println("  cache unset: the models are in hwnpu in the user cache directory")
	}
	if _, set := h.Settings[envfile.PolicyEnv]; !set {
		fmt.Println("  model-allowlist and model-denylist unset: every model but those that may make NSFW pictures (and what models.yaml's policy: refuses)")
	}
	if len(h.Pending) > 0 {
		var shown []string
		for _, n := range h.Pending {
			if d := envfile.Display(n); !slices.Contains(shown, d) {
				shown = append(shown, d)
			}
		}
		fmt.Printf("  the engine still runs with the old %s: it is stopped when no request goes on, and starts again with these (hwnpu service restart: now)\n", strings.Join(shown, ", "))
	}
	here := daemon.Settings(os.Environ())
	for _, m := range daemon.Compare(here, h.Settings) {
		fmt.Printf("  differs here: %s (hwnpud's is used)\n", m)
	}
	if v := here[envfile.PolicyEnv]; v != "" && v != h.Settings[envfile.PolicyEnv] {
		fmt.Printf("  here %s=%s: it narrows hwnpud's for the clients of this shell\n", envfile.PolicyEnv, v)
	}
}

// StartCmd starts the daemon: through the service when it is installed, else in the foreground.
type StartCmd struct{}

func (c *StartCmd) Run(ctx context.Context) error {
	cl := daemon.NewClient("")
	_, err := cl.Hello(ctx)
	if errors.Is(err, daemon.ErrNotRunning) && !installed(cl) {
		return foreground(cl.Socket)
	}
	if err != nil {
		return err
	}
	h, err := cl.Start(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("hwnpud runs: pid %d, engine pid %d, build %s\n", h.PID, h.Engine, h.Build)
	return nil
}

// foreground runs hwnpud serve in this process, in place of this command: its log comes here, and
// Ctrl-C ends it. It gets the environment this command was started with, without what the .env
// files set, to read the files itself (and again when they change).
func foreground(socket string) error {
	exe, err := Hwnpud()
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "no service of hwnpud listens on %s (hwnpu service install): running it here, in the foreground; Ctrl-C or hwnpu service stop ends it\n", socket)
	return syscall.Exec(exe, []string{exe, "serve"}, envfile.Original())
}

// StopCmd ends the daemon.
type StopCmd struct{}

func (c *StopCmd) Run(ctx context.Context) error {
	cl := daemon.NewClient("")
	h, err := cl.Hello(ctx)
	if err != nil {
		return err
	}
	if err := cl.Quit(ctx); err != nil {
		return err
	}
	if h.Service != "" {
		fmt.Printf("hwnpud stopped (%s starts it again at the next connection; hwnpu service uninstall keeps it stopped)\n", h.Service)
	} else {
		fmt.Println("hwnpud stopped")
	}
	return nil
}

// RestartCmd ends the daemon and starts it again.
type RestartCmd struct{}

func (c *RestartCmd) Run(ctx context.Context) error {
	cl := daemon.NewClient("")
	if _, err := cl.Hello(ctx); errors.Is(err, daemon.ErrNotRunning) {
		if !installed(cl) {
			return fmt.Errorf("%w: hwnpu service start", err)
		}
		return (&StartCmd{}).Run(ctx)
	}
	old, h, err := cl.Restart(ctx)
	if err != nil {
		return err
	}
	fmt.Printf("hwnpud restarted: pid %d, engine pid %d, build %s (was %s)\n", h.PID, h.Engine, h.Build, old.Build)
	return nil
}
