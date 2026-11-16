//go:build !darwin

// Command hwnpu-dev prebuild compiles Core AI programs ahead of time: macOS alone (see
// main_darwin.go).
package prebuild

import (
	"fmt"
	"os"
)

func Main() {
	fmt.Fprintln(os.Stderr, "hwnpu-dev prebuild: Core AI programs are compiled on macOS alone, with Xcode 27's Metal Toolchain")
	os.Exit(1)
}
