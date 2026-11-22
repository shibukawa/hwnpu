//go:build !linux

package mem

// DRMMemory is Linux's.
func DRMMemory() (bytes int64, ok bool) { return 0, false }
