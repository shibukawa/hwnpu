package daemon

import (
	"fmt"
	"net"
	"os"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"
)

// activated returns the listeners launchd holds for this process under the socket name (the
// plist's Sockets key), nil when launchd did not start it.
func activated(name string) ([]net.Listener, error) {
	lib, err := purego.Dlopen("/usr/lib/libSystem.B.dylib", purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, nil
	}
	var activate func(name string, fds *unsafe.Pointer, cnt *uintptr) int32
	var free func(p unsafe.Pointer)
	for sym, fn := range map[string]any{"launch_activate_socket": &activate, "free": &free} {
		p, err := purego.Dlsym(lib, sym)
		if err != nil || p == 0 {
			return nil, nil
		}
		purego.RegisterFunc(fn, p)
	}
	var fds unsafe.Pointer
	var cnt uintptr
	switch rc := activate(name, &fds, &cnt); syscall.Errno(rc) {
	case 0:
	case syscall.ESRCH, syscall.ENOENT: // not started by launchd, or without that socket
		return nil, nil
	default:
		return nil, fmt.Errorf("launch_activate_socket %s: %v", name, syscall.Errno(rc))
	}
	defer free(fds)
	var out []net.Listener
	for _, fd := range unsafe.Slice((*int32)(fds), cnt) {
		f := os.NewFile(uintptr(fd), name)
		ln, err := net.FileListener(f)
		f.Close() // FileListener dups it
		if err != nil {
			for _, l := range out {
				l.Close()
			}
			return nil, err
		}
		out = append(out, ln)
	}
	return out, nil
}
