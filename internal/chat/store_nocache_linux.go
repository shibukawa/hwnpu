//go:build linux

package chat

import (
	"os"

	"golang.org/x/sys/unix"
)

func setNoCache(f *os.File) {
	_ = unix.Fadvise(int(f.Fd()), 0, 0, unix.FADV_SEQUENTIAL)
}

// dropPageCache drops f's pages from the file cache. Read pages are clean and go at once. Written
// ones are dirty and stay until they are on the disk, so they are written back and dropped on a
// copy of the descriptor in the background: the answer does not wait for the disk (145 MB on
// NVMe: fdatasync 2-28 ms, on a slower disk more), and f can be closed and renamed meanwhile.
func dropPageCache(f *os.File, written bool) {
	if !written {
		_ = unix.Fadvise(int(f.Fd()), 0, 0, unix.FADV_DONTNEED)
		return
	}
	fd, err := unix.Dup(int(f.Fd()))
	if err != nil {
		return
	}
	dropping.Add(1)
	go func() {
		defer dropping.Done()
		_ = unix.Fdatasync(fd)
		_ = unix.Fadvise(fd, 0, 0, unix.FADV_DONTNEED)
		unix.Close(fd)
	}()
}
