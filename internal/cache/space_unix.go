//go:build unix

package cache

import (
	"io/fs"
	"syscall"

	"golang.org/x/sys/unix"
)

func diskSpace(dir string) (free, total int64, err error) {
	var st unix.Statfs_t
	if err := unix.Statfs(dir, &st); err != nil {
		return 0, 0, err
	}
	return int64(st.Bavail) * int64(st.Bsize), int64(st.Blocks) * int64(st.Bsize), nil
}

// fileKey names a file whatever link it is read by.
type fileKey struct{ dev, ino uint64 }

func keyOf(fi fs.FileInfo) (fileKey, bool) {
	st, ok := fi.Sys().(*syscall.Stat_t)
	if !ok || st.Nlink < 2 {
		return fileKey{}, false // one link: never met again
	}
	return fileKey{uint64(st.Dev), uint64(st.Ino)}, true
}
