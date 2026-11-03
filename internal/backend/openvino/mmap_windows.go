//go:build windows

package openvino

import (
	"os"
	"sync"
	"syscall"
	"unsafe"
)

const supported = true

func shmDir() string { return os.TempDir() }

// views are the mapping handles of the mapped buffers, by their first byte.
var views sync.Map

// mapFile makes a file of size bytes at path and maps it shared.
func mapFile(path string, size int) ([]byte, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o600)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	n := max(size, 1)
	if err := f.Truncate(int64(n)); err != nil {
		return nil, err
	}
	h, err := syscall.CreateFileMapping(syscall.Handle(f.Fd()), nil, syscall.PAGE_READWRITE, uint32(uint64(n)>>32), uint32(n), nil)
	if err != nil {
		return nil, os.NewSyscallError("CreateFileMapping", err)
	}
	addr, err := syscall.MapViewOfFile(h, syscall.FILE_MAP_WRITE, 0, 0, uintptr(n))
	if err != nil {
		syscall.CloseHandle(h)
		return nil, os.NewSyscallError("MapViewOfFile", err)
	}
	// the view is outside Go's heap; the copy through a pointer keeps vet's unsafeptr check quiet
	b := unsafe.Slice((*byte)(*(*unsafe.Pointer)(unsafe.Pointer(&addr))), n)
	views.Store(addr, h)
	return b[:size], nil
}

func unmap(b []byte) error {
	addr := uintptr(unsafe.Pointer(unsafe.SliceData(b[:cap(b)])))
	err := syscall.UnmapViewOfFile(addr)
	if h, ok := views.LoadAndDelete(addr); ok {
		syscall.CloseHandle(h.(syscall.Handle))
	}
	return err
}
