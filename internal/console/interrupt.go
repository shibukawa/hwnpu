package console

import (
	"context"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"time"
)

// InterruptGrace is how long a command has, after Ctrl-C ends its context, to return by itself
// (what watches the context: a download, a child process, a file half written to remove) before
// the process is ended.
const InterruptGrace = 100 * time.Millisecond

// interrupted is whether Ctrl-C has ended the context of Interrupt, until its stop.
var interrupted atomic.Bool

// grace is InterruptGrace, longer in a test of the second Ctrl-C.
var grace = InterruptGrace

// Interrupt is the context of a command that Ctrl-C ends, and what stops watching for it (which
// ends the context too, as signal.NotifyContext's stop does: a command that goes on to take
// Ctrl-C by itself, as a chat does to stop an answer, calls it once its model is loaded).
//
// The context alone ends a command late: a command on the NPU cannot be taken back, so the
// device's wait returns at once but closing its run waits for the command (up to 2.7 s, a stage
// of the VAE at 1024 x 1024), allocating and reading a graph's pinned buffers are single calls
// that see no context, and freeing them takes as long again (Qwen-Image: up to 2.3 s from Ctrl-C
// to the prompt). So the process is ended InterruptGrace after Ctrl-C, or at once on a second
// one, as the signal ends a process that does not catch it: the kernel takes back the NPU's
// contexts and buffers of a killed process (within 0.35 s, and the next run draws the same
// picture).
func Interrupt() (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithCancel(context.Background())
	sig := make(chan os.Signal, 2)
	signal.Notify(sig, os.Interrupt)
	stopped := make(chan struct{})
	var once sync.Once
	stop := func() {
		once.Do(func() {
			signal.Stop(sig)
			interrupted.Store(false)
			close(stopped)
			cancel()
		})
	}
	go func() {
		select {
		case <-sig:
		case <-stopped:
			return
		}
		interrupted.Store(true)
		cancel()
		select {
		case <-sig: // a second Ctrl-C
		case <-time.After(grace):
		case <-stopped:
			return
		}
		if interrupted.Load() {
			end()
		}
	}()
	return ctx, stop
}

// EndInterrupted ends the process as the signal does when err is what a command returned after
// Ctrl-C ended the context of Interrupt: the error is then the cancellation at whatever the
// command was doing ("step 1: blocks 0..3: command 0: context canceled"), not a failure to
// report. Otherwise it returns err.
func EndInterrupted(err error) error {
	if err != nil && interrupted.Load() {
		end()
	}
	return err
}

// end ends the process as the interrupt signal does (endInterrupted), after a new line on a
// terminal: the cursor is after the "^C" the terminal shows, often on a line of progress, and
// what comes next (go run's "signal: interrupt", the prompt of a shell that adds no new line
// itself) would follow on that line.
func end() {
	if Terminal(os.Stderr) {
		os.Stderr.WriteString("\n")
	}
	endInterrupted()
}
