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

// Probe asks whether b's pages are in memory by looking at k of them, spread inside it, a page a
// call (all of them where b is at most 4 pages): for ranges read and dropped whole, which are in
// memory or not as a whole. Resident walks every page, which for a mapped file costs about 0.25
// µs a page (16 µs for 1 MB of 16 KiB pages on an M3): asked for each of a generated token's
// experts, that was more time than their products took. ok is false where it cannot be told.
func Probe(b []byte, k int) (in, ok bool) {
	if len(b) == 0 {
		return true, true
	}
	page := os.Getpagesize()
	base := unsafe.Pointer(unsafe.SliceData(b))
	off := int(uintptr(base) & uintptr(page-1))
	pages := (off + len(b) + page - 1) / page
	if pages <= 4 || k <= 0 {
		n, ok := Resident(b)
		return ok && n >= int64(len(b)), ok
	}
	if _, ok := Resident(b[:1]); !ok { // (binds mincore)
		return false, false
	}
	var vec [1]byte
	first := unsafe.Add(base, -off)
	for i := range k {
		p := (2*i + 1) * pages / (2 * k)
		if mincore.fn(unsafe.Add(first, p*page), uintptr(page), &vec[0]) != 0 {
			return false, false
		}
		if vec[0]&1 == 0 {
			return false, true
		}
	}
	return true, true
}
