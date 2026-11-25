package experts

import (
	"fmt"
	"os"
	"slices"
	"sync"
	"time"
)

// runTimes keeps, for BONSAIAI_DEV's moe.time, how long a row's layer of experts took (one row:
// a generated token) by what its experts were: all wired, some not wired but every page of
// them in memory (the page cache kept them, or the prefetcher brought them in), or some with
// pages out of memory when the row came, guessed by the prefetcher (not in yet) or not.
type runTimes struct {
	mu  sync.Mutex
	got [4][]time.Duration
	// the prefetcher's guesses of the next layer, in those times
	guesses int
	guessed time.Duration
}

var runKinds = [4]string{"all wired", "unwired ones in memory", "guessed ones not in memory yet", "ones not guessed out of memory"}

// kind is what row idx of layer l runs on: an index of runKinds. Called before the prefetcher
// is told of the row (it drops the layer's guess) and before the row's experts are read
// (reader.need); the pages in memory are Host.cold's sample.
func (h *Host) kind(l int, idx []int) int {
	pl := h.place
	if pl == nil {
		return 0
	}
	E := h.shape.Experts
	k := 0
	for _, e := range idx {
		if e < 0 || e >= E || pl.on[l*E+e].Load() {
			continue
		}
		in := !h.cold(l, e)
		switch {
		case in:
			k = max(k, 1)
		case h.pre != nil && h.pre.guessed(l, e):
			k = max(k, 2)
		default:
			return 3
		}
	}
	return k
}

func (t *runTimes) add(kind int, d time.Duration) {
	t.mu.Lock()
	t.got[kind] = append(t.got[kind], d)
	t.mu.Unlock()
}

func (t *runTimes) guess(d time.Duration) {
	t.mu.Lock()
	t.guesses++
	t.guessed += d
	t.mu.Unlock()
}

func (t *runTimes) report() {
	t.mu.Lock()
	defer t.mu.Unlock()
	n, all := 0, time.Duration(0)
	for _, g := range t.got {
		n += len(g)
		for _, d := range g {
			all += d
		}
	}
	if n == 0 {
		return
	}
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1e3 }
	if t.guesses > 0 {
		fmt.Fprintf(os.Stderr, "experts: the next layer's guess %.3f ms a row's layer, in the times below\n", ms(t.guessed)/float64(t.guesses))
	}
	for k, g := range t.got {
		if len(g) == 0 {
			continue
		}
		slices.Sort(g)
		var sum time.Duration
		for _, d := range g {
			sum += d
		}
		q := func(f float64) time.Duration { return g[min(int(f*float64(len(g))), len(g)-1)] }
		fmt.Fprintf(os.Stderr, "experts: a row's layer, %s: %.1f%% of %d, mean %.2f ms (%.1f%% of the time), p10 %.2f, median %.2f, p90 %.2f, p99 %.2f\n",
			runKinds[k], 100*float64(len(g))/float64(n), n, ms(sum)/float64(len(g)), 100*float64(sum)/float64(all), ms(q(0.1)), ms(q(0.5)), ms(q(0.9)), ms(q(0.99)))
	}
}
