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
	load   *loadState // of NewLoadBar: the bar is estimated while frac is not known
	steps  *steps     // of NewTransferBar on no terminal: lines as it goes

	done, exited chan struct{}
}

// barWidth is the cells of the bar.
const barWidth = 24

// NewBar starts a line of progress on w for label, with no share known yet.
func NewBar(w io.Writer, label string) *Bar { return newBar(w, label, nil) }

// NewTransferBar is NewBar for a download or copy, which can take minutes: on no terminal (a log
// file, a pipe to tee) it also writes the label when it starts and a line at each tenth done (each
// stepEvery while the share is not known), where NewBar writes nothing until Done.
func NewTransferBar(w io.Writer, label string) *Bar {
	b := newBar(w, label, nil)
	if w != nil && !b.live {
		b.steps = &steps{last: time.Now()}
		fmt.Fprintf(w, "%s ...\n", label)
	}
	return b
}

// steps is what a NewTransferBar on no terminal has written.
type steps struct {
	tenth int       // the tenths done written
	last  time.Time // when a line was last written
}

// stepEvery is how often a NewTransferBar on no terminal writes a line while the share is not known.
var stepEvery = 30 * time.Second

func newBar(w io.Writer, label string, load *loadState) *Bar {
	b := &Bar{w: w, label: label, live: w != nil && Live(w), start: time.Now(), frac: -1, load: load}
	if load != nil {
		load.start, load.curStart = b.start, b.start
		pushLoad(b)
	}
	if !b.live {
		return b
	}
	if d, ok := w.(interface{ Direct() io.Writer }); ok {
		b.w = d.Direct() // the bar writes holding the live lines' lock (Log's own writes take it)
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
	var line string
	if st := b.steps; st != nil {
		now := time.Now()
		if t := int(b.frac * 10); b.frac >= 0 && t > st.tenth && t < 10 {
			st.tenth, st.last = t, now
			line = fmt.Sprintf("%s %d%% %s %s", b.label, t*10, Duration(now.Sub(b.start)), detail)
		} else if b.frac < 0 && now.Sub(st.last) >= stepEvery {
			st.last = now
			line = fmt.Sprintf("%s %s %s", b.label, Duration(now.Sub(b.start)), detail)
		}
	}
	b.mu.Unlock()
	if line != "" {
		fmt.Fprintln(b.w, strings.TrimRight(line, " "))
	}
}

// Elapsed is the time since the bar started.
func (b *Bar) Elapsed() time.Duration { return time.Since(b.start) }

// Done ends the line with result after the label (and a newline).
func (b *Bar) Done(result string) {
	if b.load != nil {
		popLoad(b, true)
	}
	b.end(result)
}

func (b *Bar) end(result string) {
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
	if frac < 0 && b.load != nil {
		if frac = b.load.frac(time.Now()); frac >= 0 && detail == "" {
			detail = "(estimated)"
		}
	}
	b.mu.Unlock()
	line := barLine(b.label, spinFrames[i%len(spinFrames)], frac, time.Since(b.start), detail, columnsOf(b.w))
	b.line.draw(func() { io.WriteString(b.w, oneLine(b.w, "\r"+line)) })
}

// barLine is a bar's line: the label, the spinner, the bar, the share done, the time and the
// detail. On a terminal of cols columns (0: not known) too narrow for all of it, the label is
// shortened first (to 16 columns), then the bar (to nothing): the share, the time and the detail
// (a download's bytes so far and in all) are what is left to the end, as the line is cut at the
// terminal's width (a download in a narrow window showed the bar alone, its sizes cut off,
// 2026-11-21).
func barLine(label, spin string, frac float64, elapsed time.Duration, detail string, cols int) string {
	tail := " " + Duration(elapsed)
	if detail != "" {
		tail += "  " + detail
	}
	pct := ""
	if frac >= 0 {
		pct = fmt.Sprintf(" %3.0f%%", frac*100)
	}
	width := barWidth
	if cols > 0 {
		const minLabel = 16
		// " ⠋" + " ▕" + bar + "▏" + pct + tail
		fixed := 2 + columnsIn(pct) + columnsIn(tail)
		if frac >= 0 {
			fixed += 3
		}
		if room := cols - 1 - minLabel - fixed; room < width {
			width = max(room, 0)
		}
		if frac >= 0 && width < 4 {
			width = 0
		}
	}
	var s strings.Builder
	fmt.Fprintf(&s, " %s", spin)
	if frac >= 0 && width > 0 {
		full := int(frac * float64(width))
		fmt.Fprintf(&s, " ▕%s%s▏", strings.Repeat("█", full), strings.Repeat("░", width-full))
	}
	s.WriteString(pct + tail)
	if cols > 0 {
		label = shorten(label, max(cols-1-columnsIn(s.String()), 16))
	}
	return label + s.String()
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

// Above writes s (whole lines) to w, a terminal, above the line a bar or spinner is drawing:
// that line is cleared first and drawn again below at its next redraw, so that a line of log
// written while a bar goes on is not run into the bar's.
func Above(w io.Writer, s string) (int, error) {
	liveLines.Lock()
	defer liveLines.Unlock()
	if len(liveLines.stack) > 0 {
		s = "\r\x1b[K" + s
	}
	return io.WriteString(w, s)
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
	switch l := w.(type) {
	case logWriter:
		w = l.w
	case teeRun:
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
