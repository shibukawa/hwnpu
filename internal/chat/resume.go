package chat

import "slices"

// markKind records why an in-memory mark was taken, so the mark pool can keep the marks that save
// the most prefill work instead of evicting oldest first.
type markKind int

const (
	markOther markKind = iota
	markTurn           // start of a chat turn (markTurns)
	markEnd            // end of a prompt
	markSig            // right before a learned divergence signature
	markNKinds
)

const (
	rsmSigLen  = 4 // tokens after a divergence point that identify a rewritten tail block
	rsmMaxSigs = 4 // most distinct signatures kept (LRU)
)

type rsmSig struct {
	lead int32   // token at the divergence point itself (preferred match; disambiguates <tag> vs </tag>)
	toks []int32 // rsmSigLen tokens after the divergence point (fallback when lead BPE-merges)
	used int64
}

// resumeTracker learns where agent-loop prompts repeatedly diverge from the cached prefix (such as
// a <system-reminder> block stripped and re-appended at the tail), predicts those cut points on
// later prompts so prefill marks right before them, and scores marks by expected saved tokens.
type resumeTracker struct {
	sigs  []rsmSig
	clock int64
	hits  [markNKinds]float64
	tries [markNKinds]float64
}

var rsmPrior = [markNKinds]float64{
	markOther: 0.2,
	markTurn:  0.5,
	markEnd:   0.5,
	markSig:   0.9,
}

// rate is the empirical resume hit rate of kind k, smoothed with a prior worth 2 observations.
func (r *resumeTracker) rate(k markKind) float64 {
	if k < 0 || k >= markNKinds {
		return rsmPrior[markOther]
	}
	return (r.hits[k] + 2*rsmPrior[k]) / (r.tries[k] + 2)
}

// value is expected prompt tokens saved by keeping a mark of kind k at position pos.
func (r *resumeTracker) value(k markKind, pos int) float64 {
	return r.rate(k) * float64(pos)
}

func (r *resumeTracker) recordHit(k markKind) {
	if k >= 0 && k < markNKinds {
		r.hits[k]++
		r.tries[k]++
	}
}

func (r *resumeTracker) recordInvalidated(k markKind) {
	if k >= 0 && k < markNKinds {
		r.tries[k]++
	}
}

// observeDivergence learns a signature when a new prompt diverges from cached tokens inside the
// previous prompt (0 < common < prevPromptLen). The token at common is kept as lead (to distinguish
// opening and closing XML tags like <system-reminder> vs </system-reminder>), while the rsmSigLen
// tokens after it provide a fallback if lead merges with preceding whitespace on a later turn.
func (r *resumeTracker) observeDivergence(cached []int32, prevPromptLen, common int) {
	if common <= 0 || common >= prevPromptLen || common+1+rsmSigLen > len(cached) {
		return
	}
	lead := cached[common]
	if lead < 0 {
		return // ignore soft media tokens
	}
	sig := cached[common+1 : common+1+rsmSigLen]
	for _, t := range sig {
		if t < 0 {
			return // ignore soft media tokens
		}
	}
	r.clock++
	for i := range r.sigs {
		if r.sigs[i].lead == lead && slices.Equal(r.sigs[i].toks, sig) {
			r.sigs[i].used = r.clock
			return
		}
	}
	if len(r.sigs) >= rsmMaxSigs {
		victim := 0
		for i := 1; i < len(r.sigs); i++ {
			if r.sigs[i].used < r.sigs[victim].used {
				victim = i
			}
		}
		r.sigs = slices.Delete(r.sigs, victim, victim+1)
	}
	r.sigs = append(r.sigs, rsmSig{lead: lead, toks: slices.Clone(sig), used: r.clock})
}

// predictPoints returns sorted, unique positions (> after, < len(prompt)) where prefill should cut
// and take a mark: one token before the last occurrence of each learned signature in prompt.
// Preferring a match whose preceding token also equals lead avoids matching closing XML tags
// (</system-reminder>) when learning an opening tag (<system-reminder>), while falling back to a
// lead-agnostic match handles BPE boundary merges with preceding whitespace.
func (r *resumeTracker) predictPoints(prompt []int32, after int) []int {
	if len(r.sigs) == 0 || len(prompt) <= rsmSigLen+1 {
		return nil
	}
	var pts []int
	for _, s := range r.sigs {
		matched := false
		for q := len(prompt) - rsmSigLen - 1; q-1 > after; q-- {
			if prompt[q-1] == s.lead && slices.Equal(prompt[q:q+rsmSigLen], s.toks) {
				pts = append(pts, q-1)
				matched = true
				break
			}
		}
		if matched {
			continue
		}
		// Fallback when lead merged with preceding text: take up to 2 trailing matches so an
		// opening+closing tag pair still marks the opening tag.
		found := 0
		for q := len(prompt) - rsmSigLen - 1; q-1 > after; q-- {
			if slices.Equal(prompt[q:q+rsmSigLen], s.toks) {
				pts = append(pts, q-1)
				found++
				if found >= 2 {
					break
				}
			}
		}
	}
	slices.Sort(pts)
	return slices.Compact(pts)
}

// addMark inserts m into marks, keeping at most limit entries. Evictions are cost-aware: the mark
// with the lowest expected saved tokens (hit rate * position) is evicted, and a new mark worth less
// than all kept marks is skipped.
func (r *resumeTracker) addMark(marks []mark, m mark, limit int) []mark {
	if limit <= 0 {
		return marks
	}
	pos := m.m.Pos()
	for i := range marks {
		if marks[i].m.Pos() == pos {
			if r.value(m.kind, pos) >= r.value(marks[i].kind, pos) {
				marks[i] = m
			}
			return marks
		}
	}
	if len(marks) >= limit {
		vNew := r.value(m.kind, pos)
		victim := 0
		vOld := r.value(marks[0].kind, marks[0].m.Pos())
		for i := 1; i < len(marks); i++ {
			if v := r.value(marks[i].kind, marks[i].m.Pos()); v < vOld {
				victim, vOld = i, v
			}
		}
		if vOld > vNew {
			return marks
		}
		marks = slices.Delete(marks, victim, victim+1)
	}
	return append(marks, m)
}
