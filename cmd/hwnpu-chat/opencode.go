package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand/v2"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// Ports -opencode serves on: from 30000 up to the start of the ephemeral range macOS (and the
// IANA) give to outgoing connections, so that the port neither collides with well-known services
// nor is taken by a connection a moment later.
const (
	portLow  = 30000
	portHigh = 49152
)

// listenFree listens on a free port of 127.0.0.1 in [portLow, portHigh), trying from a random
// one so that several instances do not race for the same port. The listener is kept, so the port
// cannot be taken between finding it and serving on it.
func listenFree() (net.Listener, error) {
	start := portLow + rand.IntN(portHigh-portLow)
	for i := 0; i < portHigh-portLow; i++ {
		port := portLow + (start-portLow+i)%(portHigh-portLow)
		if l, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port)); err == nil {
			return l, nil
		}
	}
	return nil, fmt.Errorf("no free port from %d to %d", portLow, portHigh-1)
}

// openCodeConfig is an OpenCode config with the server as the provider hwnpu and its one model.
func openCodeConfig(baseURL, model string, context int) ([]byte, error) {
	return json.MarshalIndent(map[string]any{
		"$schema": "https://opencode.ai/config.json",
		"provider": map[string]any{
			"hwnpu": map[string]any{
				"npm":     "@ai-sdk/openai-compatible",
				"name":    "hwnpu (Neural Engine)",
				"options": map[string]any{"baseURL": baseURL},
				"models": map[string]any{
					model: map[string]any{
						"name":      model + " on the Neural Engine",
						"tool_call": true,
						"reasoning": true,
						"limit":     map[string]any{"context": context, "output": min(4096, context/4)},
					},
				},
			},
		},
		"model": "hwnpu/" + model,
	}, "", "  ")
}

// openCode serves s on addr (a free port from 30000 when empty), writes an OpenCode config for it
// into dir, and runs OpenCode (bin, with args after the model) on this terminal: it gets the
// standard input, output and error as they are, so its interface works as when started alone.
// The server's log goes to dir/server.log meanwhile. It returns OpenCode's exit code.
func openCode(s *chat.Server, addr, dir, bin string, args []string) (int, error) {
	path, err := exec.LookPath(bin)
	if err != nil {
		return 1, fmt.Errorf("%s not found; install OpenCode (https://opencode.ai) or name it with -opencode-bin: %w", bin, err)
	}
	var l net.Listener
	if addr == "" {
		l, err = listenFree()
	} else {
		l, err = net.Listen("tcp", addr)
	}
	if err != nil {
		return 1, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return 1, err
	}
	logPath := filepath.Join(dir, "server.log")
	logFile, err := os.Create(logPath)
	if err != nil {
		return 1, err
	}
	defer logFile.Close()
	s.Log = logFile
	baseURL := fmt.Sprintf("http://%s/v1", l.Addr())
	cfg, err := openCodeConfig(baseURL, s.E.Name, s.E.Context())
	if err != nil {
		return 1, err
	}
	cfgPath := filepath.Join(dir, "opencode.json")
	if err := os.WriteFile(cfgPath, cfg, 0o644); err != nil {
		return 1, err
	}
	srv := &http.Server{Handler: s.Handler()}
	served := make(chan error, 1)
	go func() { served <- srv.Serve(l) }()
	defer srv.Close()
	fmt.Fprintf(os.Stderr, "serving %s at %s (context %d tokens), log in %s\nstarting %s with %s\n", s.E.Name, baseURL, s.E.Context(), logPath, bin, cfgPath)

	cmd := exec.Command(path, append([]string{"-m", "hwnpu/" + s.E.Name}, args...)...)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	cmd.Env = append(os.Environ(), "OPENCODE_CONFIG="+cfgPath)
	// Ctrl-C in the terminal is OpenCode's: catch it here (a caught signal is reset for the
	// child, an ignored one would stay ignored) and let OpenCode decide.
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	defer signal.Stop(sig)
	go func() {
		for range sig {
		}
	}()
	if err := cmd.Start(); err != nil {
		return 1, err
	}
	waited := make(chan error, 1)
	go func() { waited <- cmd.Wait() }()
	select {
	case err := <-waited:
		var exit *exec.ExitError
		if errors.As(err, &exit) {
			return exit.ExitCode(), nil
		}
		return 0, err
	case err := <-served:
		cmd.Process.Kill()
		<-waited
		return 1, fmt.Errorf("the server stopped: %w", err)
	}
}

// openCodeDir is where -opencode keeps the generated config and the server's log.
func openCodeDir(cacheDir string) string {
	if cacheDir == "" {
		cacheDir = os.TempDir()
	}
	return filepath.Join(cacheDir, "opencode")
}
