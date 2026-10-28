//go:build darwin || linux

// Package mem allocates page aligned memory outside the Go heap, so that native frameworks
// may keep pointers to it.
package mem

import (
	"errors"
	"syscall"
)

// Alloc maps size bytes of zeroed anonymous memory, rounded up to the page size.
// The returned slice has length size.
func Alloc(size int) ([]byte, error) {
	if size <= 0 {
		return nil, errors.New("mem: size must be positive")
	}
	page := syscall.Getpagesize()
	n := (size + page - 1) / page * page
	b, err := syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		return nil, err
	}
	return b[:size:n], nil
}

// Free unmaps memory returned by Alloc.
func Free(b []byte) error {
	if b == nil {
		return nil
	}
	return syscall.Munmap(b[:cap(b)])
}
