package mem

import (
	"encoding/binary"
	"syscall"
)

// Physical is the machine's memory in bytes (hw.memsize), 0 when it cannot be read.
func Physical() int64 {
	s, err := syscall.Sysctl("hw.memsize")
	if err != nil {
		return 0
	}
	// the raw uint64, little endian, its trailing zero bytes cut off by Sysctl
	var b [8]byte
	copy(b[:], s)
	return int64(binary.LittleEndian.Uint64(b[:]))
}
