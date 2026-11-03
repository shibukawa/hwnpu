//go:build !windows

package amdxrt

// preloadXRT does nothing outside Windows: the shim finds libxrt_coreutil through its rpath
// or the loader's search path (LD_LIBRARY_PATH for an unpacked XRT).
func preloadXRT() {}
