package cache

import "github.com/ebitengine/purego"

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
