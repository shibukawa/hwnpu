package chat

import (
	"container/heap"
	"math"
	"math/rand/v2"
	"sort"
)

// Sampling chooses the next token from the logits.
type Sampling struct {
	// Temperature 0 picks the most likely token.
	Temperature float64
	// TopK and TopP keep the K most likely tokens, then the fewest of them whose
	// probabilities add up to TopP. 0 turns either off.
	TopK int
	TopP float64
	// Repetition divides the logits of the tokens already generated in this answer (positive
	// logits; it multiplies negative ones), as Hugging Face's repetition_penalty does. 1 or 0
	// turns it off. The prompt is left out: an agent copies names from it.
	Repetition float64

	// softcap is the soft cap c the engine's session leaves to sampling (decode.Session.Softcap):
	// the logits are the program's y, the model's are c*tanh(y/c). It is monotonic, so the most
	// likely tokens are the same, and pick applies it to the values it uses alone.
	softcap float64
}

// capped is the logit after the soft cap.
func (s Sampling) capped(y float32) float64 {
	if s.softcap <= 0 {
		return float64(y)
	}
	return s.softcap * math.Tanh(float64(y)/s.softcap)
}

// uncapped is the logit before the soft cap whose capped value is l (capped's inverse): a value at
// or past the cap, which no logit reaches, is far past every logit.
func (s Sampling) uncapped(l float64) float32 {
	c := s.softcap
	switch {
	case c <= 0:
		return float32(l)
	case l >= c:
		return float32(math.MaxFloat32)
	case l <= -c:
		return -float32(math.MaxFloat32)
	}
	return float32(c * math.Atanh(l/c))
}

// pick chooses a token. It changes logits.
func (s Sampling) pick(logits []float32, seen []int32, rng *rand.Rand) int32 {
	if s.Repetition > 0 && s.Repetition != 1 {
		done := map[int32]bool{}
		for _, t := range seen {
			if done[t] || int(t) >= len(logits) {
				continue
			}
			done[t] = true
			if s.softcap <= 0 {
				if l := logits[t]; l > 0 {
					logits[t] = l / float32(s.Repetition)
				} else {
					logits[t] = l * float32(s.Repetition)
				}
				continue
			}
			// on the capped logit, as the model's are; written back before the cap, so that the
			// order with the others stays
			if l := s.capped(logits[t]); l > 0 {
				logits[t] = s.uncapped(l / s.Repetition)
			} else {
				logits[t] = s.uncapped(l * s.Repetition)
			}
		}
	}
	if s.Temperature <= 0 {
		best := 0
		for i, l := range logits {
			if l > logits[best] {
				best = i
			}
		}
		return int32(best)
	}
	idx := topK(logits, s.TopK)
	ps := make([]float64, len(idx))
	var sum float64
	top := s.capped(logits[idx[0]])
	for i, j := range idx {
		if s.softcap > 0 {
			ps[i] = math.Exp((s.capped(logits[j]) - top) / s.Temperature)
		} else {
			ps[i] = math.Exp(float64(logits[j]-logits[idx[0]]) / s.Temperature)
		}
		sum += ps[i]
	}
	if s.TopP > 0 && s.TopP < 1 {
		var acc float64
		for i, p := range ps {
			if acc += p; acc >= s.TopP*sum {
				ps, idx, sum = ps[:i+1], idx[:i+1], acc
				break
			}
		}
	}
	r := rng.Float64() * sum
	for i, p := range ps {
		if r -= p; r <= 0 {
			return int32(idx[i])
		}
	}
	return int32(idx[len(idx)-1])
}

// topK returns the indices of the k largest logits, largest first (all when k is 0).
func topK(logits []float32, k int) []int {
	if k <= 0 || k >= len(logits) {
		idx := make([]int, len(logits))
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
		return idx
	}
	h := &minHeap{l: logits}
	for i := range logits {
		if len(h.idx) < k {
			heap.Push(h, i)
		} else if logits[i] > logits[h.idx[0]] {
			h.idx[0] = i
			heap.Fix(h, 0)
		}
	}
	idx := h.idx
	sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
	return idx
}

type minHeap struct {
	l   []float32
	idx []int
}

func (h *minHeap) Len() int           { return len(h.idx) }
func (h *minHeap) Less(i, j int) bool { return h.l[h.idx[i]] < h.l[h.idx[j]] }
func (h *minHeap) Swap(i, j int)      { h.idx[i], h.idx[j] = h.idx[j], h.idx[i] }
func (h *minHeap) Push(x any)         { h.idx = append(h.idx, x.(int)) }
func (h *minHeap) Pop() any {
	x := h.idx[len(h.idx)-1]
	h.idx = h.idx[:len(h.idx)-1]
	return x
}
