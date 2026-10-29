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
			if l := logits[t]; l > 0 {
				logits[t] = l / float32(s.Repetition)
			} else {
				logits[t] = l * float32(s.Repetition)
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
	for i, j := range idx {
		ps[i] = math.Exp(float64(logits[j]-logits[idx[0]]) / s.Temperature)
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
