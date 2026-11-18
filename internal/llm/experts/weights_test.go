package experts

import (
	"math"
	"slices"
	"testing"
)

// The bias moves the choice, not the weights; the weights are the chosen experts' sigmoid scores
// over their sum (LFM2-MoE's router).
func TestChooseSigmoidBias(t *testing.T) {
	logits := []float32{2, 1, 0, -1}
	bias := []float32{0, 0, 3, 0}
	idx, w := make([]int, 2), make([]float32, 2)
	Routing{Score: "sigmoid", Bias: true, Norm: true}.Choose(logits, bias, idx, w)
	if !slices.Equal(idx, []int{2, 0}) {
		t.Fatalf("chose %v, want [2 0]", idx)
	}
	sig := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	s2, s0 := sig(0), sig(2)
	for i, want := range []float64{s2 / (s2 + s0), s0 / (s2 + s0)} {
		if math.Abs(float64(w[i])-want) > 1e-5 {
			t.Errorf("weight %d = %g, want %g", i, w[i], want)
		}
	}
}

func TestQuantizeRoundTrip(t *testing.T) {
	w := []float32{0.5, -0.25, 0.125, 1e-3, 0, 0, 0, 0}
	q, s := make([]byte, 8), make([]float32, 2)
	quantize(w, 2, 4, q, s)
	if s[1] != 1 {
		t.Errorf("a row of zeros has scale %g, want 1", s[1])
	}
	for c := 0; c < 4; c++ {
		got := float32(int8(q[c])) * s[0]
		if math.Abs(float64(got-w[c])) > float64(s[0])/2+1e-6 {
			t.Errorf("column %d: %g back as %g", c, w[c], got)
		}
	}
}
