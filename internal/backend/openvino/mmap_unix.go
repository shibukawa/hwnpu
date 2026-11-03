//go:build unix

package openvino

import (
	"os"
	"syscall"
)

const supported = true

// shmDir is where the shared buffers' files go: memory backed where there is such a place.
func shmDir() string {
	if fi, err := os.Stat("/dev/shm"); err == nil && fi.IsDir() {
		return "/dev/shm"
	}
	return os.TempDir()
}

// mapFile makes a file of size bytes at path and maps it shared.
func mapFile(path string, size int) ([]byte, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	n := max(size, 1)
	if err := f.Truncate(int64(n)); err != nil {
		os.Remove(path)
		return nil, err
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, n, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		os.Remove(path)
		return nil, err
	}
	return b[:size], nil
}

func unmap(b []byte) error { return syscall.Munmap(b[:cap(b)]) }
