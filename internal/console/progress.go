package console

import (
	"fmt"
	"io"
	"strings"
	"sync"
	"time"
)

// Bar is a line of progress: on a terminal, the label, a spinner, a bar, the share done, the
// seconds since it started and a detail, redrawn ten times a second until Done, which leaves the
// label and its result. Elsewhere it writes only the line Done gives.
type Bar struct {
	w     io.Writer
	label string
	live  bool
	start time.Time

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
		fmt.Fprintf(b.w, "\r%s %s\x1b[K\n", b.label, result)
		return
	}
	fmt.Fprintf(b.w, "%s %s\n", b.label, result)
}

func (b *Bar) draw(i int) {
	b.mu.Lock()
	frac, detail := b.frac, b.detail
	b.mu.Unlock()
	var s strings.Builder
	fmt.Fprintf(&s, "\r%s %s", b.label, spinFrames[i%len(spinFrames)])
	if frac >= 0 {
		full := int(frac * barWidth)
		fmt.Fprintf(&s, " ▕%s%s▏ %3.0f%%", strings.Repeat("█", full), strings.Repeat("░", barWidth-full), frac*100)
	}
	fmt.Fprintf(&s, " %s", Duration(time.Since(b.start)))
	if detail != "" {
		s.WriteString("  " + detail)
	}
	s.WriteString("\x1b[K")
	io.WriteString(b.w, s.String())
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
