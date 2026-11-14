package console

import (
	"fmt"
	"io"
	"os"
	"strings"
	"sync"
	"time"
)

// Bar is a line of progress: on a terminal, the label, a spinner, a bar, the share done, the
// seconds since it started and a detail, redrawn ten times a second until Done, which leaves the
// label and its result. Elsewhere it writes only the line Done gives. A bar or spinner started
// while one is drawn (a program loaded during "decoding") takes the line over until it is done,
// then the one before goes on drawing on the next line: two redrawing the same line in turn flicker.
type Bar struct {
	w     io.Writer
	label string
	live  bool
	start time.Time
	line  *liveLine

	mu     sync.Mutex
	frac   float64 // < 0: not known (no bar, the spinner alone)
	detail string

	done, exited chan struct{}
}

// barWidth is the cells of the bar.
const barWidth = 24

// NewBar starts a line of progress on w for label, with no share known yet.
func NewBar(w io.Writer, label string) *Bar {
	b := &Bar{w: w, label: label, live: w != nil && Live(w), start: time.Now(), frac: -1}
	if !b.live {
		return b
	}
	b.done, b.exited = make(chan struct{}), make(chan struct{})
	b.line = pushLine()
	b.draw(0)
	go func() {
		defer close(b.exited)
		t := time.NewTicker(100 * time.Millisecond)
		defer t.Stop()
		for i := 1; ; i++ {
			select {
			case <-b.done:
				return
			case <-t.C:
				b.draw(i)
			}
		}
	}()
	return b
}

// Set says how far it is (0 to 1; negative when not known) and what to show after the bar.
func (b *Bar) Set(frac float64, detail string) {
	b.mu.Lock()
	b.frac, b.detail = min(frac, 1), detail
	b.mu.Unlock()
}

// Elapsed is the time since the bar started.
func (b *Bar) Elapsed() time.Duration { return time.Since(b.start) }

// Done ends the line with result after the label (and a newline).
func (b *Bar) Done(result string) {
	if b.w == nil {
		return
	}
	if b.live {
		close(b.done)
		<-b.exited
		b.line.end(func() { fmt.Fprintf(b.w, "\r%s %s\x1b[K\n", b.label, result) })
		return
	}
	fmt.Fprintf(b.w, "%s %s\n", b.label, result)
}

func (b *Bar) draw(i int) {
	b.mu.Lock()
	frac, detail := b.frac, b.detail
	b.mu.Unlock()
	var s strings.Builder
	fmt.Fprintf(&s, " %s", spinFrames[i%len(spinFrames)])
	if frac >= 0 {
		full := int(frac * barWidth)
		fmt.Fprintf(&s, " ▕%s%s▏ %3.0f%%", strings.Repeat("█", full), strings.Repeat("░", barWidth-full), frac*100)
	}
	fmt.Fprintf(&s, " %s", Duration(time.Since(b.start)))
	if detail != "" {
		s.WriteString("  " + detail)
	}
	label := b.label
	if n := columnsOf(b.w); n > 0 {
		// a narrow terminal shortens the label first: the bar and how far it is are what to see
		label = shorten(label, max(n-1-columnsIn(s.String()), 16))
	}
	b.line.draw(func() { io.WriteString(b.w, oneLine(b.w, "\r"+label+s.String())) })
}

// liveLines are the bars and spinners being drawn, the last started last: only it draws, so
// that one started inside another does not redraw the same line in turn with it.
var liveLines struct {
	sync.Mutex
	stack []*liveLine
}

// liveLine is a place in liveLines.
type liveLine struct{ _ byte } // not of size 0, whose pointers may all be equal

func pushLine() *liveLine {
	l := new(liveLine)
	liveLines.Lock()
	liveLines.stack = append(liveLines.stack, l)
	liveLines.Unlock()
	return l
}

// draw runs write when l is the last started, the writes of all lines one at a time.
func (l *liveLine) draw(write func()) {
	liveLines.Lock()
	defer liveLines.Unlock()
	if n := len(liveLines.stack); n > 0 && liveLines.stack[n-1] == l {
		write()
	}
}

// end takes l out of liveLines and runs write, the last of the line.
func (l *liveLine) end(write func()) {
	liveLines.Lock()
	defer liveLines.Unlock()
	for i, x := range liveLines.stack {
		if x == l {
			liveLines.stack = append(liveLines.stack[:i], liveLines.stack[i+1:]...)
			break
		}
	}
	write()
}

// columnsIn is the columns s takes.
func columnsIn(s string) int {
	n := 0
	for _, r := range s {
		n += cellWidth(r)
	}
	return n
}

// shorten keeps the start of s and an ellipsis when it takes more than n columns.
func shorten(s string, n int) string {
	if columnsIn(s) <= n {
		return s
	}
	return cut(s, n-1) + "…"
}

// oneLine makes a line that \r starts to redraw in place: cut to the terminal's width, as one
// that wraps leaves its first rows behind at each redraw (\r goes back to the start of the last
// row only), and written with the terminal's wrapping off (DECAWM), so that a line still wider
// than the terminal thinks (wide characters, a window made narrower) is clipped, not wrapped. It
// ends with erasing what is left of the row.
func oneLine(w io.Writer, line string) string {
	if n := columnsOf(w); n > 0 {
		line = cut(line, n-1)
	}
	return "\x1b[?7l" + line + "\x1b[K\x1b[?7h"
}

// cut keeps of s (a leading \r aside) what fits in n columns.
func cut(s string, n int) string {
	used := 0
	for i, r := range s {
		if r == '\r' {
			continue
		}
		if used += cellWidth(r); used > n {
			return s[:i]
		}
	}
	return s
}

// columnsOf is the width of the terminal w writes to, 0 when not known.
func columnsOf(w io.Writer) int {
	if l, ok := w.(logWriter); ok {
		w = l.w
	}
	if f, ok := w.(*os.File); ok {
		return columns(f)
	}
	return 0
}

// Duration is d for a person: 0.4s, 12s, 3m05s, 1h02m.
func Duration(d time.Duration) string {
	switch {
	case d < 10*time.Second:
		return fmt.Sprintf("%.1fs", d.Seconds())
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm%02ds", int(d.Minutes()), int(d.Seconds())%60)
	}
	return fmt.Sprintf("%dh%02dm", int(d.Hours()), int(d.Minutes())%60)
}
