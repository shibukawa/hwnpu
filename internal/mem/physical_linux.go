package mem

import "syscall"

// Physical is the machine's memory in bytes (sysinfo), 0 when it cannot be read.
func Physical() int64 {
	var si syscall.Sysinfo_t
	if err := syscall.Sysinfo(&si); err != nil {
		return 0
	}
	return int64(si.Totalram) * int64(si.Unit)
}
