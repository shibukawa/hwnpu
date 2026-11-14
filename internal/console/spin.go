package console

import (
	"fmt"
	"io"
	"os"
	"time"
)

// Live reports whether w shows a terminal, where a line can be redrawn: a terminal file, or a
// writer of Log over one.
func Live(w io.Writer) bool {
	switch v := w.(type) {
	case *os.File:
		return Terminal(v)
	case interface{ Terminal() bool }:
		return v.Terminal()
	}
	return false
}

// Terminal says that a writer of Log writes to a terminal (Log returns one only for a terminal).
func (logWriter) Terminal() bool { return true }

// spinFrames are the frames of the spinner.
var spinFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

// Spin writes label and, on a terminal, a spinner and the seconds since it started after it, redrawn
// ten times a second, until stop: then the line holds the label alone, the cursor after it, for
// the caller to finish. Elsewhere it writes the label once.
func Spin(w io.Writer, label string) (stop func()) {
	if w == nil {
		return func() {}
	}
	if !Live(w) {
		io.WriteString(w, label)
		return func() {}
	}
	start := time.Now()
	line := pushLine()
	draw := func(i int) {
		line.draw(func() {
			io.WriteString(w, oneLine(w, fmt.Sprintf("\r%s %s %ds", label, spinFrames[i%len(spinFrames)], int(time.Since(start).Seconds()))))
		})
	}
	draw(0)
	done, exited := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(exited)
		t := time.NewTicker(100 * time.Millisecond)
		defer t.Stop()
		for i := 1; ; i++ {
			select {
			case <-done:
				return
			case <-t.C:
				draw(i)
			}
		}
	}()
	return func() {
		close(done)
		<-exited
		line.end(func() { fmt.Fprintf(w, "\r%s\x1b[K", label) })
	}
}
