package chat

import (
	"math/rand/v2"
	"slices"
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
