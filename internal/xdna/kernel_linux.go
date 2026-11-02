package xdna

import (
	"syscall"
	"unsafe"
)

// OpenNode opens an amdxdna device node such as /dev/accel/accel0. Opening binds the process
// address space to the IOMMU (SVA), which fails with EINTR when a signal arrives, as Go's
// preemption signal does; the open is retried like any other interrupted call.
func OpenNode(path string) (Driver, error) {
	for {
		fd, err := syscall.Open(path, syscall.O_RDWR|syscall.O_CLOEXEC, 0)
		if err == syscall.EINTR {
			continue
		}
		if err != nil {
			return nil, err
		}
		return &node{fd: fd}, nil
	}
}

type node struct{ fd int }

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

func (n *node) Mmap(offset uint64, length int) ([]byte, error) {
	return syscall.Mmap(n.fd, int64(offset), length, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
}

func (n *node) Munmap(b []byte) error { return syscall.Munmap(b) }
func (n *node) Close() error          { return syscall.Close(n.fd) }
