package llm

import (
	"context"
	"testing"
)

// series gives wired memory in MiB, look by look (the last one from then on), and counts the
// looks; a negative value is one that cannot be read.
func series(mib ...int64) (func() (int64, bool), *int) {
	looks := 0
	return func() (int64, bool) {
		v := mib[min(looks, len(mib)-1)]
		looks++
		return v << 20, v >= 0
	}, &looks
}

// After the lock is taken, the wait for the Neural Engine ends when no more than the base is on
// it, when what is left stops falling, or after its longest; it is no wait where nothing is over
// the base or nothing can be read.
func TestSettle(t *testing.T) {
	ctx := context.Background()
	for _, c := range []struct {
		name  string
		base  int64 // MiB
		mib   []int64
		waits int
		left  int64 // MiB
		told  int64 // MiB over the base
		ok    bool
	}{
		{"empty", 0, []int64{30}, 0, 30, 0, true},
		{"at the limit", 0, []int64{128}, 0, 128, 0, true},
		{"not readable", 0, []int64{-1}, 0, 0, 0, false},
		{"freed in three looks", 0, []int64{1500, 1500, 900, 20}, 3, 20, 0, true},
		{"flat for a while, then freed", 0, append(repeat(1500, 70), 0), 70, 0, 1500, true},
		{"becomes unreadable", 0, []int64{1500, 1500, -1}, 2, 1500, 0, false},
		{"in use by another program", 0, []int64{600}, npuStall, 600, 600, true},
		{"another program's, known from the process before", 600, []int64{650}, 0, 650, 0, true},
		{"the process before's over another program's", 600, []int64{2100, 2100, 640}, 2, 640, 0, true},
		{"falls, then in use", 0, []int64{1500, 1000, 600}, 2 + npuStall, 600, 600, true},
		{"a small change is no fall", 0, []int64{1500, 1495, 1490}, npuStall, 1490, 1490, true},
		{"small changes add up to a fall", 0, []int64{1500, 1490, 1480}, 2 + npuStall, 1480, 1480, true},
	} {
		wired, looks := series(c.mib...)
		var told int64
		left, waits, ok := settle(ctx, wired, c.base<<20, 0, func(over int64) { told = over >> 20 })
		if waits != c.waits || left>>20 != c.left || told != c.told || ok != c.ok {
			t.Errorf("%s: waited %d looks, %d MiB left, told of %d MiB, ok %v; want %d, %d, %d, %v", c.name, waits, left>>20, told, ok, c.waits, c.left, c.told, c.ok)
		} else if *looks != c.waits+1 {
			t.Errorf("%s: %d looks for %d waits", c.name, *looks, c.waits)
		}
	}
	// falling slowly all the way: the longest wait
	mib := make([]int64, npuSettle+2)
	for i := range mib {
		mib[i] = int64(20000 - 20*i)
	}
	wired, _ := series(mib...)
	if _, waits, _ := settle(ctx, wired, 0, 0, nil); waits != npuSettle {
		t.Errorf("falling without end: waited %d looks, want %d", waits, npuSettle)
	}
	// a context that ends, ends the wait
	done, cancel := context.WithCancel(ctx)
	cancel()
	wired, _ = series(1500)
	if _, waits, _ := settle(done, wired, 0, 0, nil); waits != 0 {
		t.Errorf("a cancelled context: waited %d looks", waits)
	}
}

func repeat(v int64, n int) []int64 {
	out := make([]int64, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func TestLockBase(t *testing.T) {
	for content, want := range map[string]int64{
		"":                                     0,
		"process 12 (bonsaiai-chat)\n":         0,
		"process 12 (bonsaiai-chat)\nbase 0\n": 0,
		"process 12 (bonsaiai-image)\nbase 629145600\n": 629145600,
		"process 12 (bonsaiai-image)\nbase -5\n":        0,
		"process 12 (bonsaiai-image)\nbase x\n":         0,
		// what a shorter content leaves of a longer one before it
		"process 7 (bonsaiai-chat)\nbase 12\n145600\n": 12,
	} {
		if got := lockBase([]byte(content)); got != want {
			t.Errorf("%q: base %d, want %d", content, got, want)
		}
	}
}
