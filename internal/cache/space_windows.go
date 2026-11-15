//go:build windows

package cache

import (
	"io/fs"

	"golang.org/x/sys/windows"
)

func diskSpace(dir string) (free, total int64, err error) {
	p, err := windows.UTF16PtrFromString(dir)
	if err != nil {
		return 0, 0, err
	}
	var avail, size, all uint64
	if err := windows.GetDiskFreeSpaceEx(p, &avail, &size, &all); err != nil {
		return 0, 0, err
	}
	return int64(avail), int64(size), nil
}

// fileKey names a file whatever link it is read by (not told on Windows: each link counts).
type fileKey struct{}

func keyOf(fs.FileInfo) (fileKey, bool) { return fileKey{}, false }
