//go:build unix

package mapped

import (
	"fmt"
	"os"
	"syscall"
)

// Open maps the file at path.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close() // the mapping outlives the descriptor
	fi, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if fi.Size() == 0 {
		return &File{b: []byte{}}, nil
	}
	if int64(int(fi.Size())) != fi.Size() {
		return nil, fmt.Errorf("mapped: %s is too large to map", path)
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(fi.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mapped: mapping %s: %w", path, err)
	}
	return &File{b: b, mapped: true}, nil
}

// Anon is n bytes of memory of their own, outside Go's heap, to be written: Close gives them back
// to the system at once, where memory the garbage collector frees may stay with the process for
// minutes (a copy of a model of hundreds of MB that a library takes its own copy of).
func Anon(n int) (*File, error) {
	if n == 0 {
		return &File{b: []byte{}}, nil
	}
	b, err := syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_PRIVATE|syscall.MAP_ANON)
	if err != nil {
		return nil, fmt.Errorf("mapped: %d bytes: %w", n, err)
	}
	return &File{b: b, mapped: true}, nil
}

func unmap(b []byte) error { return syscall.Munmap(b) }
