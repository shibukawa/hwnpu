//go:build !linux

package iron

import (
	"errors"
	"os"
)

// openDirect has no O_DIRECT off Linux (the AMD NPU's only OS): the experts are read from the
// page cache.
func openDirect(path string) (*os.File, error) {
	return nil, errors.New("no O_DIRECT on this OS")
}
