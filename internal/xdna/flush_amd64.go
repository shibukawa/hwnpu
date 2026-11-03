package xdna

import "unsafe"

//go:noescape
func clflushRange(p unsafe.Pointer, n uintptr, line uintptr)

// cacheLine is the CLFLUSH line size. 64 bytes on every x86-64 CPU with an AMD NPU.
const cacheLine = 64

// FlushCPU writes back and invalidates the CPU cache lines of b, so the device sees what the
// CPU wrote and the CPU then reads what the device wrote. XRT does this in user space on x86
// (pcidev.cpp clflush_data) instead of the SYNC_BO ioctl. SYNC_BO is not enough on Linux 7.0:
// it flushes through the kernel's view of the buffer object, which a host buffer object
// lacks until the driver pins its pages for a command, so the first input written to a new
// buffer reached the device as zeros (seen on a Ryzen AI 9 365).
func FlushCPU(b []byte) bool {
	if len(b) > 0 {
		clflushRange(unsafe.Pointer(&b[0]), uintptr(len(b)), cacheLine)
	}
	return true
}
