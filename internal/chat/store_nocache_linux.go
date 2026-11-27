//go:build linux

package chat

import (
	"os"

	"golang.org/x/sys/unix"
)

func setNoCache(f *os.File) {
	_ = unix.Fadvise(int(f.Fd()), 0, 0, unix.FADV_SEQUENTIAL)
}

func dropPageCache(f *os.File, written bool) {
	if written {
		_ = f.Sync()
	}
	_ = unix.Fadvise(int(f.Fd()), 0, 0, unix.FADV_DONTNEED)
}
