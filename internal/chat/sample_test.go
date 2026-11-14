package chat

import (
	"math"
	"math/rand/v2"
	"slices"
	"sort"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// Sampling the logits before a soft cap, with the cap left to pick (softcap), chooses as sampling
// the capped logits does: the same token at temperature 0, with the repetition penalty moving
// tokens across the most likely one, and nearly always the same with temperature, top-k and top-p
// (the values differ in their last bits only).
func TestPickSoftcap(t *testing.T) {
	const c = 30
	rng := rand.New(rand.NewPCG(1, 2))
	raw := make([]float32, 2000)
	for i := range raw {
		raw[i] = float32(rng.NormFloat64() * 25) // well past the cap: tanh flattens the top
	}
	top := slices.Index(raw, slices.Max(raw))
	seen := []int32{int32(top), 3, 17, 3, 900}
	capped := slices.Clone(raw)
	llm.Softcap(capped, c)
	for _, tc := range []struct {
		name string
		s    Sampling
		min  float64 // share of the seeds picking alike
	}{
		{"greedy", Sampling{Repetition: 1.3}, 1},
		{"greedy, strong penalty", Sampling{Repetition: 3}, 1},
		{"top-k 50", Sampling{Temperature: 0.8, TopK: 50, Repetition: 1.05}, 0.99},
		{"top-k and top-p", Sampling{Temperature: 1.5, TopK: 200, TopP: 0.9, Repetition: 1.2}, 0.99},
	} {
		same := 0
		const n = 500
		for seed := range uint64(n) {
			want := tc.s.pick(slices.Clone(capped), seen, rand.New(rand.NewPCG(seed, 7)))
			s := tc.s
			s.softcap = c
			got := s.pick(slices.Clone(raw), seen, rand.New(rand.NewPCG(seed, 7)))
			if got == want {
				same++
			}
		}
		if share := float64(same) / n; share < tc.min {
			t.Errorf("%s: %d of %d seeds pick alike, want %.0f%%", tc.name, same, n, 100*tc.min)
		}
	}
	// the penalty acts on the capped values: token 0 (raw 100, capped 29.9) penalised by 1.05 is
	// 28.5, under token 1 (raw 80, capped 29.6); on the raw logits it would stay ahead (95)
	two := []float32{100, 80, 0}
	if got := (Sampling{Repetition: 1.05, softcap: c}).pick(slices.Clone(two), []int32{0}, nil); got != 1 {
		t.Errorf("the penalised token's capped logit falls under the next one's: picked %d, want 1", got)
	}
	if got := (Sampling{Repetition: 1.05}).pick(slices.Clone(two), []int32{0}, nil); got != 0 {
		t.Errorf("without the cap the penalised token stays ahead: picked %d, want 0", got)
	}
}

// topK's pass over whole blocks keeps what a full sort keeps, with ties, NaNs and infinities.
func TestTopKBlocks(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 9))
	for _, n := range []int{50, 51, 300, 1000, 5001} {
		logits := make([]float32, n)
		for i := range logits {
			logits[i] = float32(rng.NormFloat64())
		}
		if n > 100 { // past the first k, which topK takes as they are
			logits[n-2] = float32(math.NaN())
		}
		logits[n/3] = float32(math.Inf(-1))
		logits[n-1] = logits[n/4] // a tie, at the end
		for _, k := range []int{1, 7, 50} {
			got := topK(logits, k)
			want := topK(logits, 0)
			// the full sort puts the NaN anywhere: take the k largest numbers
			var top []float32
			for _, i := range want {
				if l := logits[i]; l == l {
					top = append(top, l)
				}
			}
			sort.Slice(top, func(a, b int) bool { return top[a] > top[b] })
			if len(got) != k {
				t.Fatalf("n %d k %d: %d indices", n, k, len(got))
			}
			for j, i := range got {
				if logits[i] != top[j] {
					t.Fatalf("n %d k %d: place %d is %v, want %v", n, k, j, logits[i], top[j])
				}
			}
		}
	}
}
