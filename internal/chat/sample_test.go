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
		{"top-p only", Sampling{Temperature: 0.8, TopP: 0.9, Repetition: 1.05}, 0.99},
		{"temp only", Sampling{Temperature: 0.8, Repetition: 1.05}, 0.99},
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

// benchPick times a pick over n logits (Qwen3.5's vocabulary: 248320), copied into one array as
// the engine does.
func benchPick(b *testing.B, s Sampling, n int) {
	r := rand.New(rand.NewPCG(1, 2))
	logits := make([]float32, n)
	for i := range logits {
		logits[i] = float32(r.NormFloat64() * 3)
	}
	seen := []int32{1, 2, 3, 400, 5000}
	var buf []float32
	for b.Loop() {
		buf = append(buf[:0], logits...)
		s.pick(buf, seen, r)
	}
}

func BenchmarkPickGreedy(b *testing.B) { benchPick(b, Sampling{}, 248320) }
func BenchmarkPickTopK(b *testing.B) {
	benchPick(b, Sampling{Temperature: 0.7, TopK: 20, TopP: 0.8}, 248320)
}
func BenchmarkPickTopKSoftcap(b *testing.B) {
	benchPick(b, Sampling{Temperature: 0.7, TopK: 20, TopP: 0.8, softcap: 30}, 248320)
}
func BenchmarkPickTopP(b *testing.B) {
	benchPick(b, Sampling{Temperature: 0.7, TopP: 0.9}, 248320)
}
func BenchmarkPickTopPSoftcap(b *testing.B) {
	benchPick(b, Sampling{Temperature: 0.7, TopP: 0.9, softcap: 30}, 248320)
}
func BenchmarkPickTempOnly(b *testing.B) {
	benchPick(b, Sampling{Temperature: 0.8}, 248320)
}
func BenchmarkPickTempOnlySoftcap(b *testing.B) {
	benchPick(b, Sampling{Temperature: 0.8, softcap: 30}, 248320)
}

// TestArgmax checks argmax (blocks passed over by their largest) against a plain loop: the first
// largest, with ties.
func TestArgmax(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, n := range []int{1, 5, 255, 256, 257, 1000, 248320} {
		for range 20 {
			l := make([]float32, n)
			for i := range l {
				l[i] = float32(r.IntN(50)) // ties
			}
			want := 0
			for i, v := range l {
				if v > l[want] {
					want = i
				}
			}
			if got := int(argmax(l)); got != want {
				t.Fatalf("n %d: %d, want %d", n, got, want)
			}
		}
	}
}

// TestArgmaxNaN checks that NaN (including at index 0) never beats finite logits.
func TestArgmaxNaN(t *testing.T) {
	nan := float32(math.NaN())
	cases := []struct {
		name   string
		logits []float32
		want   int32
	}{
		{"nan at 0", []float32{nan, -100, -50, nan}, 2},
		{"nan in first block, max in second", append([]float32{nan}, make([]float32, 300)...), 1},
		{"all nan", []float32{nan, nan, nan}, 0},
		{"all -inf", []float32{float32(math.Inf(-1)), float32(math.Inf(-1))}, 0},
		{"+inf wins", []float32{10, float32(math.Inf(1)), 20}, 1},
	}
	cases[1].logits[280] = 5.0
	cases[1].want = 280
	for _, tc := range cases {
		if got := argmax(tc.logits); got != tc.want {
			t.Errorf("%s: got %d, want %d", tc.name, got, tc.want)
		}
	}
}

// TestPickHistogramEdgeCases checks NaN, +Inf, -Inf, ties, and tiny temperatures for TopK=0 and TopK>0.
func TestPickHistogramEdgeCases(t *testing.T) {
	nan := float32(math.NaN())
	posInf := float32(math.Inf(1))
	negInf := float32(math.Inf(-1))
	rng := rand.New(rand.NewPCG(11, 22))

	// Multiple +Inf logits must sample uniformly among them without NaN collapse.
	infLogits := make([]float32, 600)
	for i := range infLogits {
		infLogits[i] = float32(i%7) - 3
	}
	infLogits[5] = posInf
	infLogits[400] = posInf
	for _, s := range []Sampling{
		{Temperature: 0.8, TopP: 0.9},
		{Temperature: 0.8},
		{Temperature: 0.8, TopK: 10, TopP: 0.9},
	} {
		hits := map[int32]int{}
		for range 200 {
			tok := s.pick(slices.Clone(infLogits), nil, rng)
			hits[tok]++
		}
		if hits[5]+hits[400] != 200 || hits[5] < 50 || hits[400] < 50 {
			t.Errorf("%+v with +Inf: hits=%v", s, hits)
		}
	}

	// NaN scattered through logits (including index 0) must never be picked.
	nanLogits := make([]float32, 600)
	for i := range nanLogits {
		nanLogits[i] = -5
	}
	nanLogits[0] = nan
	nanLogits[10] = nan
	nanLogits[300] = 4.0
	nanLogits[301] = 4.0
	for _, s := range []Sampling{
		{Temperature: 0.7, TopP: 0.9},
		{Temperature: 0.7, TopP: 0.9, softcap: 30},
		{Temperature: 0.7, TopK: 5},
	} {
		for range 50 {
			tok := s.pick(slices.Clone(nanLogits), nil, rng)
			if tok == 0 || tok == 10 {
				t.Fatalf("%+v picked NaN token %d", s, tok)
			}
		}
	}

	// All -Inf or all NaN or tiny temperature must not panic or return out-of-range.
	for _, l := range [][]float32{
		{nan, nan, nan},
		{negInf, negInf, negInf},
		{1, 1, 1, 1},
	} {
		for _, s := range []Sampling{
			{Temperature: 0.7, TopP: 0.9},
			{Temperature: 1e-300, TopP: 0.9},
			{Temperature: 0.7, TopK: 2},
		} {
			tok := s.pick(slices.Clone(l), nil, rng)
			if tok < 0 || int(tok) >= len(l) {
				t.Fatalf("%+v on %v returned out-of-range %d", s, l, tok)
			}
		}
	}
}

// TestPickHistogramDistribution checks that the 2048-bucket Top-P / temperature sampler produces
// the same probability distribution as exact full-sort softmax+nucleus sampling.
func TestPickHistogramDistribution(t *testing.T) {
	refProbs := func(logits []float32, s Sampling) []float64 {
		n := len(logits)
		idx := make([]int, n)
		for i := range idx {
			idx[i] = i
		}
		sort.Slice(idx, func(a, b int) bool { return logits[idx[a]] > logits[idx[b]] })
		top := s.capped(logits[idx[0]])
		ps := make([]float64, n)
		var sum float64
		for i, j := range idx {
			ps[i] = math.Exp((s.capped(logits[j]) - top) / s.Temperature)
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
		out := make([]float64, n)
		for i, j := range idx {
			out[j] = ps[i] / sum
		}
		return out
	}

	r := rand.New(rand.NewPCG(99, 100))
	logits := make([]float32, 1024)
	for i := range logits {
		logits[i] = float32(r.NormFloat64() * 2.5)
	}
	// Put a clear top cluster like real LLM logits.
	logits[17] = 12.0
	logits[400] = 11.4
	logits[700] = 10.9
	logits[900] = 10.2
	logits[55] = 9.5

	for _, tc := range []struct {
		name string
		s    Sampling
	}{
		{"top-p 0.85", Sampling{Temperature: 0.8, TopP: 0.85}},
		{"top-p 0.95 softcap", Sampling{Temperature: 0.9, TopP: 0.95, softcap: 30}},
		{"temp only", Sampling{Temperature: 0.7}},
		{"temp only softcap", Sampling{Temperature: 1.2, softcap: 30}},
	} {
		want := refProbs(logits, tc.s)
		const trials = 40000
		counts := make([]int, len(logits))
		rng := rand.New(rand.NewPCG(1234, 5678))
		for range trials {
			tok := tc.s.pick(slices.Clone(logits), nil, rng)
			counts[tok]++
		}
		// Total variation distance and forbidden-token check (tokens strictly outside nucleus).
		var maxDiff float64
		outside := 0
		for i, p := range want {
			got := float64(counts[i]) / trials
			if diff := math.Abs(got - p); diff > maxDiff {
				maxDiff = diff
			}
			if p == 0 && counts[i] > 0 {
				outside += counts[i]
			}
		}
		if outside > 0 {
			t.Errorf("%s: sampled %d tokens strictly outside Top-P nucleus", tc.name, outside)
		}
		if maxDiff > 0.01 {
			t.Errorf("%s: max probability error %.4f > 0.01", tc.name, maxDiff)
		}
	}
}
