//go:build !darwin

package modelscmd

import (
	"context"
	"errors"
)

var errNoCompiler = errors.New("Core AI's programs are compiled for coreai-ARCH on macOS alone, with Xcode 27's Metal Toolchain (coreai-build): run it on a Mac. The generic exports are made here too: hwnpu-dev models package MODEL/coreai")

// canCompile is why this OS cannot compile for coreai-ARCH.
func canCompile() error { return errNoCompiler }

func compile(context.Context, []string, []string, string, compileFlags) error { return errNoCompiler }
