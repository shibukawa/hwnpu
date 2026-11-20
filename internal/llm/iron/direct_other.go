//go:build !linux

package iron

import (
	"errors"
	"os"
)

// openDirect: no O_DIRECT here (the experts are read from the page cache).
func openDirect(path string) (*os.File, error) {
	return nil, errors.New("no O_DIRECT on this system")
}
