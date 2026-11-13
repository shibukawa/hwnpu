//go:build windows

package cache

import (
	"errors"
	"os"

	"golang.org/x/sys/windows"
)

// tryLock locks the first byte, far past what the holder writes, so that the holder's line stays
// readable.
func tryLock(f *os.File) (bool, error) {
	ol := new(windows.Overlapped)
	ol.OffsetHigh = 0x7fffffff
	err := windows.LockFileEx(windows.Handle(f.Fd()), windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, ol)
	if errors.Is(err, windows.ERROR_LOCK_VIOLATION) || errors.Is(err, windows.ERROR_IO_PENDING) {
		return false, nil
	}
	return err == nil, err
}

func unlock(f *os.File) {
	ol := new(windows.Overlapped)
	ol.OffsetHigh = 0x7fffffff
	windows.UnlockFileEx(windows.Handle(f.Fd()), 0, 1, 0, ol)
}
