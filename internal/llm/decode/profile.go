package decode

import (
	"fmt"
	"io"
	"sort"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/internal/console"
)

// profile adds up where the steps of programs split at their attention spend their time, by
// tokens per step and stages: each part's run (with its syncs) and the attention over the
// session's cache between them. Programs of as many tokens per step but other stages (the model's
// and its multi-token prediction layer's, both 8) are apart: added up together, the model's later
// parts looked a third as long as they are. It is on when HWNPU_PROFILE is set, and WriteProfile
// prints it.
var profile = struct {
	sync.Mutex
	on    bool
	steps map[profKey]*profSteps
}{on: console.Profile, steps: map[profKey]*profSteps{}}

// profKey tells programs apart: their tokens per step and stages (the attention layers and one).
type profKey struct{ q, stages int }

// profKeyOf is p's profKey.
func profKeyOf(p Program) profKey { return profKey{p.InShape["x"][1], len(p.Attend) + 1} }

// the steps of the whole run go to the command's last lines too (console.WriteProfile)
func init() { console.OnProfile(WriteProfile) }

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

func profStep(k profKey) *profSteps {
	if !profile.on {
		return nil
	}
	profile.Lock()
	defer profile.Unlock()
	p := profile.steps[k]
	if p == nil {
		p = &profSteps{}
		profile.steps[k] = p
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
	ks := make([]profKey, 0, len(profile.steps))
	perQ := map[int]int{}
	for k := range profile.steps {
		ks = append(ks, k)
		perQ[k.q]++
	}
	sort.Slice(ks, func(i, j int) bool { return ks[i].q < ks[j].q || ks[i].q == ks[j].q && ks[i].stages > ks[j].stages })
	ms := func(d time.Duration, n int) float64 { return float64(d.Microseconds()) / 1000 / float64(n) }
	for _, k := range ks {
		p := profile.steps[k]
		var total time.Duration
		name := fmt.Sprintf("q=%d", k.q)
		if perQ[k.q] > 1 {
			name += fmt.Sprintf(" (%d stages)", k.stages)
		}
		fmt.Fprintf(w, "profile %s, %d steps, ms a step:", name, p.n)
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

func profRows(k profKey, t0 time.Time) {
	d := time.Since(t0)
	profile.Lock()
	defer profile.Unlock()
	if p := profile.steps[k]; p != nil {
		p.rows += d
	}
}
