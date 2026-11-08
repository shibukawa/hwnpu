package decode

import (
	"fmt"
	"io"
	"os"
	"sort"
	"sync"
	"time"
)

// profile adds up where the steps of programs split at their attention spend their time, by
// tokens per step: each part's run (with its syncs) and the attention over the session's cache
// between them. It is on when HWNPU_PROFILE is set, and WriteProfile prints it.
var profile = struct {
	sync.Mutex
	on    bool
	steps map[int]*profSteps
}{on: os.Getenv("HWNPU_PROFILE") != "", steps: map[int]*profSteps{}}

type profSteps struct {
	n      int
	parts  []time.Duration
	attend []time.Duration // after part i
	rows   time.Duration   // decoding the logits rows
}

func (p *profSteps) part(i int, d time.Duration) {
	for len(p.parts) <= i {
		p.parts = append(p.parts, 0)
		p.attend = append(p.attend, 0)
	}
	p.parts[i] += d
}

func profStep(q int) *profSteps {
	if !profile.on {
		return nil
	}
	profile.Lock()
	defer profile.Unlock()
	p := profile.steps[q]
	if p == nil {
		p = &profSteps{}
		profile.steps[q] = p
	}
	p.n++
	return p
}

// WriteProfile prints the time per step of each part and attention, by tokens per step, when
// HWNPU_PROFILE is set; otherwise nothing.
func WriteProfile(w io.Writer) {
	if !profile.on {
		return
	}
	profile.Lock()
	defer profile.Unlock()
	qs := make([]int, 0, len(profile.steps))
	for q := range profile.steps {
		qs = append(qs, q)
	}
	sort.Ints(qs)
	ms := func(d time.Duration, n int) float64 { return float64(d.Microseconds()) / 1000 / float64(n) }
	for _, q := range qs {
		p := profile.steps[q]
		var total time.Duration
		fmt.Fprintf(w, "profile q=%d, %d steps, ms a step:", q, p.n)
		for i := range p.parts {
			fmt.Fprintf(w, " part%d %.2f", i, ms(p.parts[i], p.n))
			total += p.parts[i]
			if p.attend[i] > 0 {
				fmt.Fprintf(w, " attn %.2f", ms(p.attend[i], p.n))
				total += p.attend[i]
			}
		}
		fmt.Fprintf(w, " rows %.2f; total %.2f\n", ms(p.rows, p.n), ms(total+p.rows, p.n))
	}
}

func profRows(q int, t0 time.Time) {
	d := time.Since(t0)
	profile.Lock()
	defer profile.Unlock()
	if p := profile.steps[q]; p != nil {
		p.rows += d
	}
}
