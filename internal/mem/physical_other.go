//go:build !darwin && !linux

package mem

// Physical is the machine's memory in bytes; 0: not known on this system.
func Physical() int64 { return 0 }
