package iron

import (
	"os"
	"syscall"
)

// openDirect opens path for reads that bypass the page cache (O_DIRECT: directReads).
func openDirect(path string) (*os.File, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_DIRECT|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	return os.NewFile(uintptr(fd), path), nil
}
