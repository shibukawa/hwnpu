//go:build !unix

package console

import "os"

// endInterrupted ends the process with the exit status of a command stopped by Ctrl-C.
func endInterrupted() { os.Exit(130) }
