//go:build unix

package safetensors

import (
	"fmt"
	"syscall"
	"unsafe"
)

// Map returns a tensor's raw data in the file mapped into memory (read only), without reading it:
// the system reads the pages as they are touched and may drop them again, so a checkpoint larger
// than what is free can be run from (gpt-oss's experts). The slice stays valid until Close.
func (f *File) Map(name string) ([]byte, error) {
	t, ok := f.tensors[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: no tensor %q", name)
	}
	if f.mapped == nil {
		fi, err := f.f.Stat()
		if err != nil {
			return nil, err
		}
		b, err := syscall.Mmap(int(f.f.Fd()), 0, int(fi.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
		if err != nil {
			return nil, fmt.Errorf("safetensors: mapping %s: %w", f.f.Name(), err)
		}
		f.mapped = b
	}
	return f.mapped[f.dataBase+t.begin : f.dataBase+t.end : f.dataBase+t.end], nil
}

func (f *File) unmap() {
	if f.mapped != nil {
		syscall.Munmap(f.mapped)
		f.mapped = nil
	}
}

// MappedAt is the file offset of b when b is in the file's mapping (Map).
func (f *File) MappedAt(b []byte) (int64, bool) {
	if f.mapped == nil || len(b) == 0 {
		return 0, false
	}
	base := uintptr(unsafe.Pointer(unsafe.SliceData(f.mapped)))
	p := uintptr(unsafe.Pointer(unsafe.SliceData(b)))
	if p < base || p+uintptr(len(b)) > base+uintptr(len(f.mapped)) {
		return 0, false
	}
	return int64(p - base), true
}
