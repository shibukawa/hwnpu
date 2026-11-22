package daemon

import (
	"runtime"
	"strings"
)

// activationEnv are the variables of systemd's socket activation: the supervisor's alone, never
// passed to the engine.
var activationEnv = []string{"LISTEN_PID", "LISTEN_FDS", "LISTEN_FDNAMES"}

// withoutActivation is env without activationEnv.
func withoutActivation(env []string) []string {
	out := env[:0:0]
	for _, kv := range env {
		k, _, _ := strings.Cut(kv, "=")
		drop := false
		for _, a := range activationEnv {
			drop = drop || k == a
		}
		if !drop {
			out = append(out, kv)
		}
	}
	return out
}

// serviceManager names what starts the service here.
func serviceManager() string {
	if runtime.GOOS == "darwin" {
		return "launchd"
	}
	return "systemd"
}
