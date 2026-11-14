//go:build darwin

package npuinfo

import (
	"errors"
	"fmt"
	"sync"
	"syscall"
	"unsafe"
)

// Calling C without cgo, as golang.org/x/sys/unix does on macOS: libSystem's dlopen and dlsym
// are imported by the linker (cgo_import_dynamic, with trampolines in dl_darwin_*.s) and
// called through syscall.syscall6, which runs them on the system stack. Everything else
// (CoreFoundation, libIOReport, the proc_* calls) is looked up with dlsym at run time, so a
// macOS without one of them gives an error here instead of a binary that does not start.

//go:cgo_import_dynamic npuinfo_dlopen dlopen "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic npuinfo_dlsym dlsym "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic npuinfo_dlerror dlerror "/usr/lib/libSystem.B.dylib"

var dlopenTrampolineAddr, dlsymTrampolineAddr, dlerrorTrampolineAddr uintptr

//go:linkname syscall_syscall6 syscall.syscall6
func syscall_syscall6(fn, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err syscall.Errno)

// ccall calls the C function at fn with up to six integer or pointer arguments and returns
// its integer or pointer result. Go memory passed by pointer must stay reachable until it
// returns (runtime.KeepAlive); the goroutine does not move its stack during the call.
func ccall(fn uintptr, args ...uintptr) uintptr {
	var a [6]uintptr
	if len(args) > len(a) {
		panic("npuinfo: ccall with more than 6 arguments")
	}
	copy(a[:], args)
	r, _, _ := syscall_syscall6(fn, a[0], a[1], a[2], a[3], a[4], a[5])
	return r
}

// cstring returns s as a NUL-terminated C string in Go memory.
func cstring(s string) *byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return &b[0]
}

// gostringN reads a NUL-terminated string from b.
func gostringN(b []byte) string {
	for i, c := range b {
		if c == 0 {
			return string(b[:i])
		}
	}
	return string(b)
}

func dlerror() string {
	p := ccall(dlerrorTrampolineAddr)
	if p == 0 {
		return ""
	}
	// p is C memory, outside the Go heap; reinterpreting the word (not converting the
	// uintptr) keeps vet's unsafe.Pointer check quiet. dlerror's messages are short.
	b := *(**[256]byte)(unsafe.Pointer(&p))
	return gostringN(b[:])
}

const rtldLazy, rtldLocal = 0x1, 0x4

// dylib is a library opened with dlopen and the symbols looked up in it.
type dylib struct {
	path string
	h    uintptr
}

func dlopen(path string) (*dylib, error) {
	p := cstring(path)
	h := ccall(dlopenTrampolineAddr, uintptr(unsafe.Pointer(p)), rtldLazy|rtldLocal)
	if h == 0 {
		return nil, fmt.Errorf("dlopen %s: %s", path, dlerror())
	}
	return &dylib{path: path, h: h}, nil
}

func (l *dylib) sym(name string) (uintptr, error) {
	p := cstring(name)
	f := ccall(dlsymTrampolineAddr, l.h, uintptr(unsafe.Pointer(p)))
	if f == 0 {
		return 0, fmt.Errorf("dlsym %s in %s: %s", name, l.path, dlerror())
	}
	return f, nil
}

// syms fills the uintptr fields pointed to by ptrs with the symbols of the same index in
// names, and returns the first error.
func (l *dylib) syms(names []string, ptrs ...*uintptr) error {
	if len(names) != len(ptrs) {
		return errors.New("npuinfo: syms: names and pointers differ in length")
	}
	for i, n := range names {
		f, err := l.sym(n)
		if err != nil {
			return err
		}
		*ptrs[i] = f
	}
	return nil
}

// libSystem is opened once; it is always there.
var libSystem = sync.OnceValues(func() (*dylib, error) { return dlopen("/usr/lib/libSystem.B.dylib") })
