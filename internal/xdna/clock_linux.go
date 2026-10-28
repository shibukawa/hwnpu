package xdna

import (
	"syscall"
	"unsafe"
)

// MonotonicNow returns CLOCK_MONOTONIC in nanoseconds, the clock DRM syncobj timeouts use.
func MonotonicNow() int64 {
	var ts syscall.Timespec
	syscall.Syscall(syscall.SYS_CLOCK_GETTIME, 1 /* CLOCK_MONOTONIC */, uintptr(unsafe.Pointer(&ts)), 0)
	return ts.Nano()
}
