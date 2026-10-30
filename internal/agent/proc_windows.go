//go:build windows

package agent

import "os/exec"

func shell() (name, flag string) { return "cmd", "/c" }

func setGroup(cmd *exec.Cmd) {}

func killGroup(cmd *exec.Cmd) error {
	if cmd.Process == nil {
		return nil
	}
	return cmd.Process.Kill()
}
