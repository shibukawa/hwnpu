package console

import (
	"bytes"
	"testing"
	"time"
)

// TestBarNotLive writes only the line Done gives when the writer is no terminal: no redraws, no
// escape sequences.
func TestBarNotLive(t *testing.T) {
	var b bytes.Buffer
	bar := NewBar(&b, "downloading x")
	bar.Set(0.5, "1 / 2 GB")
	bar.Done("2 GB in 3s")
	if got, want := b.String(), "downloading x 2 GB in 3s\n"; got != want {
		t.Errorf("%q, want %q", got, want)
	}
}

// liveBuffer is a writer that says it is a terminal.
type liveBuffer struct{ bytes.Buffer }

func (*liveBuffer) Terminal() bool { return true }

// TestBarLive redraws the line with the bar and leaves the label and the result.
func TestBarLive(t *testing.T) {
	var b liveBuffer
	bar := NewBar(&b, "denoising")
	bar.Set(0.5, "step 5/8")
	time.Sleep(250 * time.Millisecond)
	bar.Done("8 steps")
	s := b.String()
	for _, want := range []string{"▕████████████░░░░░░░░░░░░▏  50%", "step 5/8", "\rdenoising 8 steps\x1b[K\n"} {
		if !bytes.Contains([]byte(s), []byte(want)) {
			t.Errorf("no %q in %q", want, s)
		}
	}
}

func TestDuration(t *testing.T) {
	for d, want := range map[time.Duration]string{
		400 * time.Millisecond: "0.4s", 12 * time.Second: "12s", 185 * time.Second: "3m05s", 62 * time.Minute: "1h02m",
	} {
		if got := Duration(d); got != want {
			t.Errorf("%v: %s, want %s", d, got, want)
		}
	}
}
