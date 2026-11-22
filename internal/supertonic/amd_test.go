package supertonic

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestAMDHost lays RoPE's angles as IRON's RoPE reads them (cosine and sine interleaved) and
// works out the time blocks' rows: the sinusoid through the MLP (Mish) and each block's Linear.
func TestAMDHost(t *testing.T) {
	h := amdHost{theta: make([]float32, ropeDim), freqs: []float32{0.5, 2}, blocks: 2}
	for j := range h.theta {
		h.theta[j] = float32(j + 1)
	}
	a := h.angles(4, 6)
	if len(a) != 6*64 {
		t.Fatalf("angles: %d values", len(a))
	}
	for i := range 6 {
		for j := range ropeDim {
			s, c := math.Sincos(float64(i) / 4 * float64(j+1))
			if math.Abs(float64(a[i*64+2*j])-c) > 1e-6 || math.Abs(float64(a[i*64+2*j+1])-s) > 1e-6 {
				t.Fatalf("angles[%d][%d]: %g %g, want %g %g", i, j, a[i*64+2*j], a[i*64+2*j+1], c, s)
			}
		}
	}
	// the MLP: 4 -> 3 -> 2 (the sinusoid of 2 frequencies: sin, sin, cos, cos), the rows 2 -> 512
	h.w0 = []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1}
	h.b0 = []float32{0, 0, -1}
	h.w2 = []float32{1, 0, 0, 0, 1, 1}
	h.b2 = []float32{0.5, 0}
	h.rowsW = make([]float32, 2*512*2)
	h.rowsB = make([]float32, 2*512)
	for k := range 2 {
		for o := range 512 {
			h.rowsW[(k*512+o)*2+k] = float32(o)
			h.rowsB[k*512+o] = float32(k)
		}
	}
	mish := func(v float64) float64 { return v * math.Tanh(math.Log1p(math.Exp(v))) }
	tt := 3.0 / 8 * 1000
	s0, c0 := math.Sincos(tt * 0.5)
	s1, c1 := math.Sincos(tt * 2)
	u := []float64{mish(s0), mish(s1), mish(c0 + c1 - 1)}
	e := []float64{u[0] + 0.5, u[1] + u[2]}
	r := h.timeRows(3, 8)
	for k := range 2 {
		for _, o := range []int{0, 1, 511} {
			if want := float64(o)*e[k] + float64(k); math.Abs(float64(r[k*512+o])-want) > 1e-3*(1+math.Abs(want)) {
				t.Fatalf("timeRows[%d][%d] = %g, want %g", k, o, r[k*512+o], want)
			}
		}
	}
}

// TestRelAttention checks the host's windowed relative attention (sgemm) against the direct
// sums: the scores with the band of relative logits, the softmax over the valid keys, the values
// with the band of relative values; rows past n zero.
func TestRelAttention(t *testing.T) {
	const rows, n, H, D = 24, 19, 2, 32
	C := H * D
	r := rand.New(rand.NewPCG(5, 6))
	qkv := make([]float32, rows*3*C)
	for i := range qkv {
		qkv[i] = float32(r.NormFloat64()) * 0.5
	}
	ek, ev := make([]float32, 9*D), make([]float32, 9*D)
	for i := range ek {
		ek[i], ev[i] = float32(r.NormFloat64()), float32(r.NormFloat64())
	}
	got := make([]float32, rows*C)
	for i := range got {
		got[i] = 7 // overwritten
	}
	relAttention(qkv, got, n, H, D, ek, ev)
	at := func(i, part, h, d int) float64 { return float64(qkv[i*3*C+part*C+h*D+d]) }
	for h := range H {
		for i := range n {
			s := make([]float64, n)
			for j := range n {
				for d := range D {
					s[j] += at(i, 0, h, d) * at(j, 1, h, d)
					if rr := j - i + relWindow; rr >= 0 && rr <= 2*relWindow {
						s[j] += at(i, 0, h, d) * float64(ek[rr*D+d])
					}
				}
			}
			mx, sum := math.Inf(-1), 0.0
			for _, v := range s {
				mx = math.Max(mx, v)
			}
			for j := range s {
				s[j] = math.Exp(s[j] - mx)
				sum += s[j]
			}
			for d := range D {
				want := 0.0
				for j := range n {
					p := s[j] / sum
					want += p * at(j, 2, h, d)
					if rr := j - i + relWindow; rr >= 0 && rr <= 2*relWindow {
						want += p * float64(ev[rr*D+d])
					}
				}
				if g := float64(got[i*C+h*D+d]); math.Abs(g-want) > 1e-4 {
					t.Fatalf("out[%d][%d][%d] = %g, want %g", i, h, d, g, want)
				}
			}
		}
	}
	for i := n * C; i < rows*C; i++ {
		if got[i] != 0 {
			t.Fatalf("out[%d] = %g past the valid rows", i/C, got[i])
		}
	}
}
