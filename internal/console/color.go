// Package console is what hwnpu-chat and hwnpu-image need of the terminal: logs in
// gray, with whether a model is all on the NPU in green or red (Log), a spinner (Spin) and a line
// of progress with a bar (Bar) for downloads, conversions, compiles and steps, and a line editor
// with a history kept across runs (Editor). None of them colors, redraws or edits unless it talks
// to a terminal; NO_COLOR turns the colors off.
package console

import (
	"fmt"
	"io"
	"os"
	"strings"
)

// The escape sequences of the colors.
const (
	gray  = "\x1b[90m"
	green = "\x1b[32m"
	red   = "\x1b[31m"
	reset = "\x1b[0m"
)

// Terminal reports whether f is a terminal.
func Terminal(f *os.File) bool {
	info, err := f.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

// Colors reports whether to color what goes to f: a terminal, with NO_COLOR unset and TERM not
// dumb.
func Colors(f *os.File) bool {
	return Terminal(f) && os.Getenv("NO_COLOR") == "" && os.Getenv("TERM") != "dumb"
}

// Log returns a writer for the logs of loading and serving a model (chat.Setup.Log and the like)
// that writes them to f in gray, with "all on the NPU" in green and "NOT all on the NPU" in red;
// f itself when it does not take colors.
func Log(f *os.File) io.Writer {
	if !Colors(f) {
		return f
	}
	return logWriter{f}
}

type logWriter struct{ w io.Writer }

func (l logWriter) Write(p []byte) (int, error) {
	if _, err := io.WriteString(l.w, gray+Highlight(string(p))+reset); err != nil {
		return 0, err
	}
	return len(p), nil
}

// Highlight colors the placements in a gray log line: "all on the NPU" green, "NOT all on the
// NPU" red, gray again after each.
func Highlight(s string) string {
	const not, all = "NOT all on the NPU", "all on the NPU"
	parts := strings.Split(s, not)
	for i, p := range parts {
		parts[i] = strings.ReplaceAll(p, all, green+all+gray)
	}
	return strings.Join(parts, red+not+gray)
}

// Notef writes a note (help, a change of setting) to standard error, in gray on a terminal.
func Notef(format string, args ...any) {
	s := fmt.Sprintf(format, args...)
	if Colors(os.Stderr) {
		s = gray + strings.TrimSuffix(s, "\n") + reset + s[len(strings.TrimSuffix(s, "\n")):]
	}
	io.WriteString(os.Stderr, s)
}
