//go:build !linux && !darwin

package console

import (
	"errors"
	"os"
)

// makeRaw is not implemented here (Windows): the editor reads lines as they come.
func makeRaw(int) (func(), error) { return nil, errors.New("console: no raw mode on this system") }

func columns(*os.File) int { return 80 }
