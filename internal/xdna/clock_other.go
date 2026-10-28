//go:build !linux

package xdna

import "time"

var start = time.Now()

// MonotonicNow returns a monotonic clock in nanoseconds (the simulated driver uses the same).
func MonotonicNow() int64 { return int64(time.Since(start)) + 1 }
