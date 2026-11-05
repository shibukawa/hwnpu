package cache

import (
	"os"

	"github.com/ebitengine/purego"
)

var cloneFn func(src, dst string, flags uint32) int32

var canClone = func() bool {
	lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return false
	}
	sym, err := purego.Dlsym(lib, "clonefile")
	if err != nil || sym == 0 {
		return false
	}
	purego.RegisterFunc(&cloneFn, sym)
	return true
}()

// clonefile is clonefile(2): dst, which must not exist, becomes a clone of src.
func clonefile(src, dst string) bool { return cloneFn(src, dst, 0) == 0 }

// sharedPrefix would say how much of two files' start lies on the same blocks; APFS has no cheap
// way to ask, so Clone compares their bytes again.
func sharedPrefix(a, b string, limit int64) int64 { return 0 }

// rangeBlock is 0: macOS clones whole files only (clonefile), so ClonePrefix does nothing.
func rangeBlock(f *os.File) int64 { return 0 }

func cloneRange(src, dst *os.File, off, n int64) bool { return false }
