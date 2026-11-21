//go:build !darwin

package modelscmd

import (
	"errors"

	"github.com/shibukawa/hwnpu/cmd/hwnpu-dev/devopt"
)

func (PrebuildCmd) Run(*devopt.Defs) error {
	return errors.New("Core AI's programs are compiled on macOS alone, with Xcode 27's Metal Toolchain (coreai-build): run hwnpu-dev models prebuild on a Mac. The generic exports are made here too: hwnpu-dev models export MODEL/coreai")
}
