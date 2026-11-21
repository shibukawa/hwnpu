package main

import (
	"fmt"
	"time"

	"github.com/shibukawa/hwnpu/internal/console"
)

// stepTimes are the denoising steps' times, for the line that ends the bar. The first step of a
// run whose programs were not in Core AI's cache yet holds their compiles and the first calls (on
// an M3, 5m25s for 8 steps of 3.7 s, 2026-11-21): the time a step is then the other steps'.
type stepTimes []time.Duration

// slowFirst is how much longer than the others' mean the first step has to take to be told apart.
const slowFirst = 5 * time.Second

// summary is "8 steps in 42s (5.3 s a step)", or, when the first step took much longer than the
// others, "8 steps in 5m25s (3.7 s a step; the first 4m59s, with loading and compiling)". per
// writes a step's time ("5.3 s a step", and with several pictures each one's share).
func (s stepTimes) summary(total time.Duration, per func(time.Duration) string) string {
	n := len(s)
	if n == 0 {
		return fmt.Sprintf("0 steps in %s", console.Duration(total))
	}
	if n >= 2 {
		var rest time.Duration
		for _, d := range s[1:] {
			rest += d
		}
		mean := rest / time.Duration(n-1)
		if s[0] > 2*mean+slowFirst {
			return fmt.Sprintf("%d steps in %s (%s; the first %s, with loading and compiling)", n, console.Duration(total), per(mean), console.Duration(s[0]))
		}
	}
	return fmt.Sprintf("%d steps in %s (%s)", n, console.Duration(total), per(total/time.Duration(n)))
}

// aStep is "5.3 s a step".
func aStep(d time.Duration) string { return fmt.Sprintf("%.1f s a step", d.Seconds()) }
