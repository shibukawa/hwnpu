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

// TestCut keeps what fits in the columns, wide characters as two.
func TestCut(t *testing.T) {
	for _, c := range []struct {
		s    string
		n    int
		want string
	}{
		{"\rdownloading x", 5, "\rdownl"},
		{"\rabc", 10, "\rabc"},
		{"\r書き出し 12%", 5, "\r書き"},
		{"\rx ▕██░░▏ 50%", 6, "\rx ▕██░"},
	} {
		if got := cut(c.s, c.n); got != c.want {
			t.Errorf("cut(%q, %d) = %q, want %q", c.s, c.n, got, c.want)
		}
	}
}

// TestOneLine turns the terminal's wrapping off for the redraw and on again.
func TestOneLine(t *testing.T) {
	if got, want := oneLine(&liveBuffer{}, "\rx"), "\x1b[?7l\rx\x1b[K\x1b[?7h"; got != want {
		t.Errorf("%q, want %q", got, want)
	}
}

func TestShorten(t *testing.T) {
	if got, want := shorten("downloading model.safetensors", 12), "downloading…"; got != want {
		t.Errorf("%q, want %q", got, want)
	}
	if got := shorten("abc", 3); got != "abc" {
		t.Errorf("%q", got)
	}
}
