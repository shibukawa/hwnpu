package main

import (
	"os"
	"strconv"
	"strings"
	"syscall"
	"time"
)

func field(path, key string) string {
	data, _ := os.ReadFile(path)
	for _, line := range strings.Split(string(data), "\n") {
		if k, v, ok := strings.Cut(line, ":"); ok && strings.TrimSpace(k) == key {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func machine() (cpu, model, osVersion string, ram int64) {
	kb, _ := strconv.ParseInt(strings.TrimSuffix(field("/proc/meminfo", "MemTotal"), " kB"), 10, 64)
	product, _ := os.ReadFile("/sys/class/dmi/id/product_name")
	osVersion = strings.Trim(field("/etc/os-release", "PRETTY_NAME"), `"`)
	if osVersion == "" {
		osVersion = "Linux"
	}
	var u syscall.Utsname
	if syscall.Uname(&u) == nil {
		var rel []byte
		for _, c := range u.Release {
			if c == 0 {
				break
			}
			rel = append(rel, byte(c))
		}
		osVersion += ", kernel " + string(rel)
	}
	return field("/proc/cpuinfo", "model name"), strings.TrimSpace(string(product)), osVersion, kb * 1024
}

func cpuTime() time.Duration {
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}
