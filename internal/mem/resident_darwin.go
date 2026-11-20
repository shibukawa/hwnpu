package mem

import (
	"os"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

var mincore struct {
	once sync.Once
	fn   func(addr unsafe.Pointer, n uintptr, vec *byte) int32
}

// Resident is how many of b's bytes (a mapping's, from whole pages) are in memory now: read
// from the disk already, wired or not (mincore). ok is false where it cannot be told.
func Resident(b []byte) (bytes int64, ok bool) {
	mincore.once.Do(func() {
		lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err != nil {
			return
		}
		if p, err := purego.Dlsym(lib, "mincore"); err == nil && p != 0 {
			purego.RegisterFunc(&mincore.fn, p)
		}
	})
	if mincore.fn == nil || len(b) == 0 {
		return 0, mincore.fn != nil
	}
	page := os.Getpagesize()
	base := unsafe.Pointer(unsafe.SliceData(b))
	off := int(uintptr(base) & uintptr(page-1)) // back to the page's start
	n := uintptr(off + len(b))
	vec := make([]byte, (int(n)+page-1)/page)
	if mincore.fn(unsafe.Add(base, -off), n, &vec[0]) != 0 {
		return 0, false
	}
	for _, v := range vec {
		if v&1 != 0 { // MINCORE_INCORE
			bytes += int64(page)
		}
	}
	return min(bytes, int64(len(b))), true
}
