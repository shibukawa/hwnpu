// Package dl loads shared libraries and binds their functions with purego.
// It turns purego's panics into errors and reports every path it tried.
package dl

import (
	"errors"
	"fmt"
	"strings"

	"github.com/ebitengine/purego"
)

// ErrNotFound is wrapped by Open when no candidate could be loaded.
var ErrNotFound = errors.New("shared library not found")

// Library is a loaded shared library.
type Library struct {
	Path   string
	handle uintptr
}

// Open loads the first candidate that succeeds. Empty candidates are skipped.
func Open(candidates ...string) (*Library, error) {
	var tried []string
	for _, c := range candidates {
		if c == "" {
			continue
		}
		h, err := open(c)
		if err == nil {
			return &Library{Path: c, handle: h}, nil
		}
		tried = append(tried, fmt.Sprintf("%s (%v)", c, err))
	}
	return nil, fmt.Errorf("%w; tried: %s", ErrNotFound, strings.Join(tried, "; "))
}

// Sym returns the address of a symbol.
func (l *Library) Sym(name string) (uintptr, error) {
	return sym(l.handle, name)
}

// Bind sets fptr (a pointer to a func variable) to call the named symbol.
func (l *Library) Bind(fptr any, name string) error {
	addr, err := l.Sym(name)
	if err != nil {
		return fmt.Errorf("%s: %w", name, err)
	}
	return BindAddr(fptr, addr)
}

// BindAddr sets fptr to call the function at addr, e.g. an entry of a driver dispatch table.
func BindAddr(fptr any, addr uintptr) (err error) {
	if addr == 0 {
		return errors.New("nil function pointer")
	}
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("bind: %v", r)
		}
	}()
	purego.RegisterFunc(fptr, addr)
	return nil
}
