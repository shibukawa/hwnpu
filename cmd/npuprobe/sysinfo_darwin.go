package main

import (
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func sysctl(name string) string {
	out, _ := exec.Command("sysctl", "-n", name).Output()
	return strings.TrimSpace(string(out))
}

func machine() (cpu, model, osVersion string, ram int64) {
	ram, _ = strconv.ParseInt(sysctl("hw.memsize"), 10, 64)
	ver, _ := exec.Command("sw_vers", "-productVersion").Output()
	return sysctl("machdep.cpu.brand_string"), sysctl("hw.model"), "macOS " + strings.TrimSpace(string(ver)), ram
}

// cpuTime is the CPU time this process has used, user and system.
func cpuTime() time.Duration {
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}
