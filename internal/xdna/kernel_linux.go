package xdna

import (
	"sync"
	"syscall"
	"unsafe"
)

// OpenNode opens an amdxdna device node such as /dev/accel/accel0.
func OpenNode(path string) (Driver, error) {
	fd, err := syscall.Open(path, syscall.O_RDWR|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	return &node{fd: fd}, nil
}

type node struct {
	fd int

	mu      sync.Mutex
	aligned map[*byte]int // the mappings of MmapAligned, by their first byte: their lengths
}

func (n *node) Ioctl(req uintptr, arg unsafe.Pointer) error {
	for {
		_, _, e := syscall.Syscall(syscall.SYS_IOCTL, uintptr(n.fd), req, uintptr(arg))
		switch e {
		case 0:
			return nil
		case syscall.EINTR:
			continue // EAGAIN is a real answer from amdxdna (e.g. no device heap yet)
		}
		return e
	}
}

// Mmap maps a buffer object. Mapping the device heap can take long enough for a signal (the Go
// runtime's preemption) to interrupt it: hwnpu-chat failed to start with "map device heap:
// interrupted system call" right after another process let the NPU go. It is tried again then.
func (n *node) Mmap(offset uint64, length int) ([]byte, error) {
	for {
		b, err := syscall.Mmap(n.fd, int64(offset), length, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
		if err != syscall.EINTR {
			return b, err
		}
	}
}

// MmapAligned maps a buffer object at an address that is a multiple of align (a power of two):
// it reserves align more address space than the mapping takes, maps the buffer object at the
// multiple of align in it, and gives back the rest. The kernel picks such an address for the
// device heap by itself only while the address space has room where it looks first.
func (n *node) MmapAligned(offset uint64, length, align int) ([]byte, error) {
	total := uintptr(length + align)
	base, _, e := syscall.Syscall6(syscall.SYS_MMAP, 0, total, syscall.PROT_NONE,
		syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS|syscall.MAP_NORESERVE, ^uintptr(0), 0)
	if e != 0 {
		return nil, e
	}
	addr := (base + uintptr(align) - 1) &^ (uintptr(align) - 1)
	for {
		_, _, e = syscall.Syscall6(syscall.SYS_MMAP, addr, uintptr(length), syscall.PROT_READ|syscall.PROT_WRITE,
			syscall.MAP_SHARED|syscall.MAP_FIXED, uintptr(n.fd), uintptr(offset))
		if e != syscall.EINTR {
			break
		}
	}
	if e != 0 {
		syscall.Syscall(syscall.SYS_MUNMAP, base, total, 0)
		return nil, e
	}
	if addr > base {
		syscall.Syscall(syscall.SYS_MUNMAP, base, addr-base, 0)
	}
	if end, top := addr+uintptr(length), base+total; top > end {
		syscall.Syscall(syscall.SYS_MUNMAP, end, top-end, 0)
	}
	// the address of memory the kernel mapped, outside Go's heap
	b := unsafe.Slice(*(**byte)(unsafe.Pointer(&addr)), length)
	n.mu.Lock()
	if n.aligned == nil {
		n.aligned = map[*byte]int{}
	}
	n.aligned[&b[0]] = length
	n.mu.Unlock()
	return b, nil
}

func (n *node) Munmap(b []byte) error {
	if len(b) > 0 {
		n.mu.Lock()
		length, ok := n.aligned[&b[0]]
		delete(n.aligned, &b[0])
		n.mu.Unlock()
		if ok { // not one of syscall.Mmap's, which syscall.Munmap knows
			if _, _, e := syscall.Syscall(syscall.SYS_MUNMAP, uintptr(unsafe.Pointer(&b[0])), uintptr(length), 0); e != 0 {
				return e
			}
			return nil
		}
	}
	return syscall.Munmap(b)
}
func (n *node) Close() error { return syscall.Close(n.fd) }
