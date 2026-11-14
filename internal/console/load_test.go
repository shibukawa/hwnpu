package console

import (
	"strings"
	"testing"
	"time"
)

// forgetLoads empties what the loads before told.
func forgetLoads(t *testing.T) {
	t.Helper()
	reset := func() {
		loads.Lock()
		loads.samples, loads.bars = nil, nil
		loads.Unlock()
	}
	reset()
	t.Cleanup(reset)
}

// TestPseudo goes with the time, then ever slower, and never shows a load as done.
func TestPseudo(t *testing.T) {
	last := -1.0
	for x := 0.0; x < 50; x += 0.01 {
		f := pseudo(x)
		if f < last || f >= 1 {
			t.Fatalf("pseudo(%.2f) = %v after %v", x, f, last)
		}
		last = f
	}
	for x, want := range map[float64]float64{0.35: 0.35, 0.8: 0.8, 1: 0.9, 2.6: 0.98} {
		if got := pseudo(x); got < want-1e-9 || got > want+1e-9 {
			t.Errorf("pseudo(%v) = %v, want %v", x, got, want)
		}
	}
}

// TestEstimate takes the slow load nearest in size, in proportion, and no load from the cache.
func TestEstimate(t *testing.T) {
	forgetLoads(t)
	if got := estimate(100 << 20); got != 0 {
		t.Errorf("%v with no load before", got)
	}
	observe(500<<20, 100*time.Millisecond) // from the compiler's cache: tells nothing
	if got := estimate(100 << 20); got != 0 {
		t.Errorf("%v after a load from the cache", got)
	}
	observe(500<<20, 50*time.Second)
	observe(10<<20, 4*time.Second)
	for bytes, want := range map[int64]time.Duration{
		250 << 20: 25 * time.Second, // half of the 500 MiB one
		20 << 20:  8 * time.Second,  // twice the 10 MiB one
		1 << 10:   400 * time.Millisecond,
	} {
		if got := estimate(bytes); got != want {
			t.Errorf("estimate(%d) = %v, want %v", bytes, got, want)
		}
	}
	observe(500<<20, 80*time.Second) // the last of a size is the one taken
	if got, want := estimate(500<<20), 80*time.Second; got != want {
		t.Errorf("%v, want %v", got, want)
	}
}

// TestLoadFrac is the bytes loaded and, of the part being loaded, the share of its time.
func TestLoadFrac(t *testing.T) {
	t0 := time.Now()
	l := &loadState{total: 1000, cur: 1000, curStart: t0}
	if got := l.frac(t0.Add(time.Minute)); got >= 0 {
		t.Errorf("%v with no estimate", got)
	}
	l.est = 10 * time.Second
	if got := l.frac(t0.Add(5 * time.Second)); got != 0.5 {
		t.Errorf("%v at half the estimate", got)
	}
	if got := l.frac(t0.Add(time.Hour)); got != 0.99 {
		t.Errorf("%v long after the estimate", got)
	}
	// 400 of 1000 loaded, the next 200 at half its time
	l = &loadState{total: 1000, done: 400, parts: true, cur: 200, curStart: t0, est: 10 * time.Second}
	if got := l.frac(t0.Add(5 * time.Second)); got != 0.5 {
		t.Errorf("%v with parts", got)
	}
	// between parts, and with a part that has no estimate: the bytes loaded
	l.cur, l.est = 0, 0
	if got := l.frac(t0); got != 0.4 {
		t.Errorf("%v between parts", got)
	}
}

// TestLoadBar is a spinner for the first program and an estimated bar for the next ones.
func TestLoadBar(t *testing.T) {
	forgetLoads(t)
	var b liveBuffer
	bar := NewLoadBar(&b, "  [1/2] text", 100<<20)
	time.Sleep(150 * time.Millisecond)
	bar.Done("loaded")
	if s := b.String(); strings.Contains(s, "▕") {
		t.Errorf("a bar with no load before: %q", s)
	}
	observe(100<<20, 2*time.Second) // as if the first had been compiled
	b.Reset()
	bar = NewLoadBar(&b, "  [2/2] dit", 100<<20)
	time.Sleep(250 * time.Millisecond)
	bar.Done("loaded")
	s := b.String()
	for _, want := range []string{"▕██", "(estimated)", "\r  [2/2] dit loaded\x1b[K\n"} {
		if !strings.Contains(s, want) {
			t.Errorf("no %q in %q", want, s)
		}
	}
}

// TestLoadingPart counts the parts in the load bar going on, and their time for the next loads.
func TestLoadingPart(t *testing.T) {
	forgetLoads(t)
	var b liveBuffer
	bar := NewLoadBar(&b, "  [1/1] model", 400)
	done := LoadingPart(100)
	done(true)
	LoadingPart(100)(false) // failed: not counted
	if got := bar.load.frac(time.Now()); got != 0.25 {
		t.Errorf("%v after 100 of 400", got)
	}
	bar.Failed("failed")
	loads.Lock()
	n, bars := len(loads.samples), len(loads.bars)
	loads.Unlock()
	if n != 0 || bars != 0 {
		t.Errorf("%d samples and %d bars after a failed load of fast parts", n, bars)
	}
	LoadingPart(100)(true) // no bar going on: nothing to count
}
