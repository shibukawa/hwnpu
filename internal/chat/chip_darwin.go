package chat

import "syscall"

// cpuBrand is the machine's CPU as macOS names it ("Apple M1", "Apple M2 Pro", ...), "" when it
// cannot be read.
func cpuBrand() string {
	s, _ := syscall.Sysctl("machdep.cpu.brand_string")
	return s
}
