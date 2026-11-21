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

	// the .env files are read before any variable is (docs/cache-sharing-design.md 7.4)
	_ "github.com/shibukawa/hwnpu/internal/envfile"
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
//
// What goes to standard error is kept in the run's file too (RecordRun).
func Log(f *os.File) io.Writer {
	if !Colors(f) {
		if f == os.Stderr {
			return teeRun{f}
		}
		return f
	}
	return logWriter{f}
}

type logWriter struct{ w *os.File }

func (l logWriter) Write(p []byte) (int, error) {
	if l.w == os.Stderr {
		recordLog(p)
	}
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

// Notef writes a note (help, a change of setting) to standard error, in gray on a terminal. While a
// bar or spinner is drawn it takes the bar's line, which the bar draws again on the next (a note
// written in the middle of "[2/20] dit_time.aimodel ⠙ ..." was left after the bar, 2026-11-21).
func Notef(format string, args ...any) {
	s := fmt.Sprintf(format, args...)
	recordLog([]byte(s))
	if Colors(os.Stderr) {
		s = gray + strings.TrimSuffix(s, "\n") + reset + s[len(strings.TrimSuffix(s, "\n")):]
	}
	liveLines.Lock()
	defer liveLines.Unlock()
	if len(liveLines.stack) > 0 && Terminal(os.Stderr) {
		s = "\r\x1b[K" + s
	}
	io.WriteString(os.Stderr, s)
}

// Columns is the width of the terminal f, 80 when it cannot be read.
func Columns(f *os.File) int { return columns(f) }

// Size is the columns and rows of the terminal f, 0 and 0 when they cannot be read.
func Size(f *os.File) (cols, rows int) { return size(f) }
