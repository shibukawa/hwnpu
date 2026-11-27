// Package daemoncmd is hwnpu-dev daemon: a hwnpud built from this checkout, on a socket of its
// own, beside the installed one. The commands are clients of hwnpud (docs/design/hwnpud.md,
// "Clients without the engine"), so a worktree's code is tried through its own hwnpud.
package daemoncmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/shibukawa/hwnpu/internal/daemon"
)

// Cmd is hwnpu-dev daemon.
type Cmd struct {
	Socket  string        `placeholder:"PATH" help:"The socket to serve on (default: hwnpud.sock in a new folder of the user's cache)."`
	Idle    time.Duration `default:"10m" help:"How long the engine keeps its models without a request."`
	Command []string      `arg:"" optional:"" passthrough:"" help:"A command to run with HWNPU_DAEMON_SOCKET set to the socket (after --); the hwnpud stops when it ends. Without one, it serves until interrupted and prints the variable to set."`
}

// Run builds ./cmd/hwnpud of the module here, serves on the socket and runs the command.
func (c *Cmd) Run() error {
	ctx := context.Background()
	if len(c.Command) > 0 && c.Command[0] == "--" {
		c.Command = c.Command[1:]
	}
	root, err := moduleRoot()
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp(cacheDir(), "hwnpud-dev-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	bin := filepath.Join(dir, "hwnpud")
	build := exec.CommandContext(ctx, "go", "build", "-o", bin, "./cmd/hwnpud")
	build.Dir, build.Stdout, build.Stderr = root, os.Stderr, os.Stderr
	if err := build.Run(); err != nil {
		return fmt.Errorf("hwnpu-dev daemon: building hwnpud: %w", err)
	}
	sock := c.Socket
	if sock == "" {
		sock = filepath.Join(dir, "hwnpud.sock")
	}
	srv := exec.Command(bin, "serve", "--socket", sock, "--idle", c.Idle.String())
	srv.Env = append(os.Environ(), daemon.SocketEnv+"="+sock)
	srv.Stdout, srv.Stderr = os.Stderr, os.Stderr
	if err := srv.Start(); err != nil {
		return err
	}
	defer func() {
		srv.Process.Signal(syscall.SIGTERM)
		srv.Wait()
	}()
	if err := waitFor(ctx, sock); err != nil {
		return err
	}
	if len(c.Command) == 0 {
		fmt.Fprintf(os.Stderr, "hwnpud of %s serves on %s: in another shell,\n  export %s=%s\nInterrupt (Ctrl-C) to stop it.\n", root, sock, daemon.SocketEnv, sock)
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
		<-sig
		return nil
	}
	cmd := exec.CommandContext(ctx, c.Command[0], c.Command[1:]...)
	cmd.Env = append(os.Environ(), daemon.SocketEnv+"="+sock, daemon.ModeEnv+"=on")
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	return cmd.Run()
}

// moduleRoot is the folder of the go.mod of hwnpu here (go env GOMOD).
func moduleRoot() (string, error) {
	out, err := exec.Command("go", "env", "GOMOD").Output()
	mod := strings.TrimSpace(string(out))
	if err != nil || mod == "" || mod == os.DevNull {
		return "", errors.New("hwnpu-dev daemon: run it in a checkout of hwnpu (go env GOMOD names none)")
	}
	return filepath.Dir(mod), nil
}

// cacheDir is the user's cache folder: a socket's path must be short (104 bytes on a Mac).
func cacheDir() string {
	if d, err := os.UserCacheDir(); err == nil {
		return d
	}
	return os.TempDir()
}

// waitFor waits for hwnpud to answer on sock.
func waitFor(ctx context.Context, sock string) error {
	c := daemon.NewClient(sock)
	for range 100 {
		hctx, cancel := context.WithTimeout(ctx, time.Second)
		_, err := c.Hello(hctx)
		cancel()
		if err == nil {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	return fmt.Errorf("hwnpu-dev daemon: hwnpud does not answer on %s", sock)
}
