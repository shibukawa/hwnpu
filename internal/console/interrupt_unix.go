//go:build unix

package console

import (
	"os"
	"os/signal"
	"syscall"
	"time"
)

// endInterrupted ends the process by SIGINT, as one that does not catch the signal ends: a
// shell then sees a command killed by Ctrl-C, and stops the loop or the script that ran it (an
// exit status of 1 let a for loop go on to its next command).
func endInterrupted() {
	signal.Reset(os.Interrupt)
	syscall.Kill(syscall.Getpid(), syscall.SIGINT)
	time.Sleep(time.Second) // the signal ends the process before this
	os.Exit(130)
}
