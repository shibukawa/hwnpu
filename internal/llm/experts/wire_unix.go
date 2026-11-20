//go:build unix

package experts

import (
	"os"
	"unsafe"

	"golang.org/x/sys/unix"
)

func mlock(b []byte) error { return unix.Mlock(b) }

func munlock(b []byte) { unix.Munlock(b) }

// willneed asks the kernel to read b's pages ahead.
func willneed(b []byte) { unix.Madvise(b, unix.MADV_WILLNEED) }

// incore is whether every page of b (whole pages) is in memory; true when it cannot tell.
func incore(b []byte) bool {
	page := os.Getpagesize()
	n := (len(b) + page - 1) / page
	if n == 0 {
		return true
	}
	var small [128]byte
	vec := small[:0]
	if n <= len(small) {
		vec = small[:n]
	} else {
		vec = make([]byte, n)
	}
	if _, _, e := unix.Syscall(unix.SYS_MINCORE, uintptr(unsafe.Pointer(unsafe.SliceData(b))), uintptr(len(b)), uintptr(unsafe.Pointer(&vec[0]))); e != 0 {
		return true
	}
	for _, v := range vec {
		if v&1 == 0 {
			return false
		}
	}
	return true
}
