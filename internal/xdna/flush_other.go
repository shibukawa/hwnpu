//go:build !amd64

package xdna

// FlushCPU reports false: without user space cache maintenance, sync through the driver
// (SYNC_BO).
func FlushCPU(b []byte) bool { return false }
