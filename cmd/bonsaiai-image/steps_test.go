package main

import (
	"testing"
	"time"
)

func TestStepTimesSummary(t *testing.T) {
	s := time.Second
	for _, c := range []struct {
		steps stepTimes
		want  string
	}{
		{stepTimes{5 * s, 5 * s, 6 * s, 5 * s}, "4 steps in 21s (5.2 s a step)"},
		{stepTimes{299 * s, 4 * s, 3 * s, 4 * s}, "4 steps in 5m10s (3.7 s a step; the first 4m59s, with loading and compiling)"},
		{stepTimes{9 * s}, "1 steps in 9.0s (9.0 s a step)"},
	} {
		var total time.Duration
		for _, d := range c.steps {
			total += d
		}
		if got := c.steps.summary(total, aStep); got != c.want {
			t.Errorf("%v: %q, want %q", c.steps, got, c.want)
		}
	}
}
