package chat

import (
	"fmt"
	"os"
	"strconv"
	"time"

	devenv "github.com/shibukawa/hwnpu/internal/dev"
)

// draftPolicy decides, call by call, whether checking the drafts of a prompt lookup pays, where
// the call that checks drafts costs more than a decode step (AMD's NPU: the version checking 2
// drafts takes 1.95 steps on Qwen3.5-0.8B and 1.68 on the 2B, docs/design.md 16.284). There a
// draft that is rarely right slows the generation down: checking every draft, an explanation in
// English ran 22 % slower than with none. It compares the tokens a call is expected to bring by
// its cost, as Basalt's DraftPolicy does (github.com/jesdga95/basalt, include/basalt/spec/draft_policy.hpp):
//
//	a step:   1 token for 1
//	a check:  1 + q + q^2 + ... (a term a draft) for ratio, q the share of the drafts that were
//	          right among those behind a match about as long
//
// The drafts it leaves unchecked are still compared with the tokens that come, so q keeps
// following the text at no cost. The cost is a constant and not the calls' times, so that the
// choices, and with them the call that computes each token, are the same from run to run.
//
// HWNPU_DEV=iron.lookup.cost is the cost of a check in steps; HWNPU_DEV=iron.lookup.trace=FILE adds
// a line a call, "KIND MATCH DRAFTS RIGHT MS" (step, check, leave, and left when the drafts left
// unchecked are known: tools/amd/draft_analyze.py reads it). Checking every draft, as before the
// policy, is gone (2026-11-27): prose ran slower than with no drafts (docs/design/amd-llm.md).
type draftPolicy struct {
	ok, bad [draftBuckets]float64 // drafts that were right, and drafts that ended a run, decayed
	open    []openDraft           // drafts left unchecked, waiting for the tokens they guessed
	ratio   float64               // a check's cost in steps
	trace   *os.File
}

const (
	draftBuckets = 6
	draftMargin  = 0.03 // a check must bring this much more than its cost
	draftPrior   = 4.0  // the prior's weight, in drafts
	draftDecay   = 0.99
	// draftCheckSteps is what a check costs in steps unless HWNPU_DEV=iron.lookup.cost says: between the
	// two models measured (1.95 and 1.68).
	draftCheckSteps = 1.8
)

// draftQ0 is each bucket's share of right drafts before any was seen.
var draftQ0 = [draftBuckets]float64{0.2, 0.35, 0.5, 0.65, 0.8, 0.9}

type openDraft struct {
	drafts []int32
	match  int
	at     int // the drafts compared so far
}

// draftBucket is the bucket of a match of n tokens: 1, 2, 3, 4-7, 8-15, 16 and more.
func draftBucket(n int) int {
	switch {
	case n <= 3:
		return max(n, 1) - 1
	case n < 8:
		return 3
	case n < 16:
		return 4
	}
	return 5
}

func newDraftPolicy() *draftPolicy {
	p := &draftPolicy{ratio: draftCheckSteps}
	if r, err := strconv.ParseFloat(devenv.Value("iron.lookup.cost"), 64); err == nil && r > 0 {
		p.ratio = r
	}
	if path := devenv.Value("iron.lookup.trace"); path != "" {
		p.trace, _ = os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	}
	return p
}

func (p *draftPolicy) q(b int) float64 {
	return (p.ok[b] + draftQ0[b]*draftPrior) / (p.ok[b] + p.bad[b] + draftPrior)
}

// take reports whether to check n drafts behind a match of match tokens.
func (p *draftPolicy) take(match, n int) bool {
	q := p.q(draftBucket(match))
	e, t := 1.0, 1.0
	for range n {
		t *= q
		e += t
	}
	return e > p.ratio*(1+draftMargin)
}

func (p *draftPolicy) count(b, right int, ended bool) {
	p.ok[b] = p.ok[b]*draftDecay + float64(right)
	p.bad[b] *= draftDecay
	if ended {
		p.bad[b]++
	}
}

// checked records a check: right of the n drafts behind a match of match tokens were kept, in d.
func (p *draftPolicy) checked(match, n, right int, d time.Duration) {
	p.count(draftBucket(match), right, right < n)
	p.log("check", match, n, right, d)
}

// stepped records a decode step of one token.
func (p *draftPolicy) stepped(d time.Duration) {
	p.log("step", 0, 0, 0, d)
}

// leave keeps drafts that are not checked, to compare them with the tokens that come.
func (p *draftPolicy) leave(match int, drafts []int32) {
	p.open = append(p.open, openDraft{drafts: drafts, match: match})
	p.log("leave", match, len(drafts), 0, 0)
}

// saw takes a token of the answer: the drafts left unchecked that guessed it go on, the others
// end.
func (p *draftPolicy) saw(tok int32) {
	keep := p.open[:0]
	for _, o := range p.open {
		if o.drafts[o.at] != tok {
			p.count(draftBucket(o.match), o.at, true)
			p.log("left", o.match, len(o.drafts), o.at, 0)
			continue
		}
		if o.at++; o.at == len(o.drafts) {
			p.count(draftBucket(o.match), o.at, false)
			p.log("left", o.match, len(o.drafts), o.at, 0)
			continue
		}
		keep = append(keep, o)
	}
	p.open = keep
}

func (p *draftPolicy) log(kind string, match, n, right int, d time.Duration) {
	if p.trace != nil {
		fmt.Fprintf(p.trace, "%s %d %d %d %.3f\n", kind, match, n, right, d.Seconds()*1e3)
	}
}

// lookupDraft is LookupDraft with the length of the match behind the drafts: how many of seq's
// last tokens (up to 32) the earlier occurrence repeats.
func lookupDraft(seq []int32, k int) ([]int32, int) {
	const maxN, minN, most = 4, 1, 32
	for n := maxN; n >= minN; n-- {
		if len(seq) <= n {
			continue
		}
		key := seq[len(seq)-n:]
	next:
		for i := len(seq) - n - 1; i >= 0; i-- {
			for j := range n {
				if seq[i+j] != key[j] {
					continue next
				}
			}
			m := n
			for m < most && i+n-1-m >= 0 && seq[i+n-1-m] == seq[len(seq)-1-m] {
				m++
			}
			return seq[i+n : min(i+n+k, len(seq))], m
		}
	}
	return nil, 0
}
