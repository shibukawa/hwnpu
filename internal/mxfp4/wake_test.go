package mxfp4

import (
	"runtime"
	"slices"
	"sync/atomic"
	"testing"
	"time"
)

// TestWake times how long after a Run starts each of its workers starts (HWNPU_DEV=mxfp4.
// bench=wake): the workers sleep between Runs, and the scheduler wakes them one after another.
func TestWake(t *testing.T) {
	if v, _ := Opt("bench"); v != "wake" {
		t.Skip("HWNPU_DEV=mxfp4.bench=wake")
	}
	n := runtime.GOMAXPROCS(0)
	for _, gap := range []time.Duration{0, time.Millisecond, 1500 * time.Microsecond, 2 * time.Millisecond, 3 * time.Millisecond, 4 * time.Millisecond, 8 * time.Millisecond} {
		const runs = 200
		at := make([][]time.Duration, n) // by the order the workers started
		for range runs {
			if gap > 0 {
				time.Sleep(gap)
			}
			var started atomic.Int32
			offs := make([]time.Duration, n)
			start := time.Now()
			Parallel(n, func(i int) {
				offs[i] = time.Since(start)
				started.Add(1)
				// hold this worker until all have started, so that each takes one
				for started.Load() < int32(n) && time.Since(start) < 20*time.Millisecond {
				}
			})
			slices.Sort(offs)
			for k, d := range offs {
				at[k] = append(at[k], d)
			}
		}
		line := ""
		for k := range at {
			slices.Sort(at[k])
			line += " " + at[k][runs/2].Round(time.Microsecond).String()
		}
		last := at[n-1]
		t.Logf("gap %v: the workers' starts, medians:%s; the last one p90 %v", gap, line, last[runs*9/10].Round(time.Microsecond))
	}
}
