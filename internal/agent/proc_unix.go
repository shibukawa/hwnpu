//go:build !windows

package agent

import (
	"os/exec"
	"syscall"
)

func shell() (name, flag string) { return "sh", "-c" }

// setGroup starts the command in its own process group, so that killing it takes the
// processes it started along.
func setGroup(cmd *exec.Cmd) { cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} }

func killGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err == nil {
		return nil
	}
	return cmd.Process.Kill()
}
