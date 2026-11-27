package qwen35

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/shibukawa/hwnpu"
)

func refReplayState(c *Config, before, rows []byte, rowShape []int, n int, after []byte) {
	hv, dk, dv := c.LinearValueHeads, c.LinearKeyDim, c.LinearValueDim
	w := dk + dv + 2
	q := rowShape[2]
	for h := 0; h < hv; h++ {
		s := make([]float32, dk*dv)
		hwnpu.DecodeF16(s, before[2*h*dk*dv:2*(h+1)*dk*dv])
		r := make([]float32, q*w)
		hwnpu.DecodeF16(r, rows[2*h*q*w:2*(h+1)*q*w])
		delta := make([]float32, dv)
		for t := 0; t < n; t++ {
			row := r[t*w : (t+1)*w]
			k, v, beta, decay := row[:dk], row[dk:dk+dv], row[dk+dv], row[dk+dv+1]
			for i := range s {
				s[i] *= decay
			}
			copy(delta, v)
			for a := 0; a < dk; a++ {
				ka := k[a]
				sa := s[a*dv : (a+1)*dv]
				for e := range delta {
					delta[e] -= ka * sa[e]
				}
			}
			for e := range delta {
				delta[e] *= beta
			}
			for a := 0; a < dk; a++ {
				ka := k[a]
				sa := s[a*dv : (a+1)*dv]
				for e := range sa {
					sa[e] += ka * delta[e]
				}
			}
		}
		hwnpu.EncodeF16(after[2*h*dk*dv:2*(h+1)*dk*dv], s)
	}
}

func TestReplayStateMatchesReference(t *testing.T) {
	cfg := &Config{LinearValueHeads: 8, LinearKeyDim: 64, LinearValueDim: 64}
	const q = 8
	hv, dk, dv := cfg.LinearValueHeads, cfg.LinearKeyDim, cfg.LinearValueDim
	w := dk + dv + 2
	rng := rand.New(rand.NewPCG(7, 13))
	bf := make([]float32, hv*dk*dv)
	for i := range bf {
		bf[i] = float32(rng.NormFloat64()) * 0.25
	}
	rf := make([]float32, hv*q*w)
	for h := range hv {
		for ti := range q {
			row := rf[(h*q+ti)*w : (h*q+ti+1)*w]
			var kn float64
			for i := range dk {
				v := rng.NormFloat64()
				row[i] = float32(v)
				kn += v * v
			}
			inv := float32(1 / math.Sqrt(kn))
			for i := range dk {
				row[i] *= inv
			}
			for i := range dv {
				row[dk+i] = float32(rng.NormFloat64()) * 0.5
			}
			row[dk+dv] = 0.05 + 0.4*rng.Float32()
			row[dk+dv+1] = 0.7 + 0.29*rng.Float32()
		}
	}
	before := make([]byte, 2*len(bf))
	rows := make([]byte, 2*len(rf))
	hwnpu.EncodeF16(before, bf)
	hwnpu.EncodeF16(rows, rf)
	shape := []int{1, hv, q, w}

	for _, n := range []int{0, 1, 3, 7, 8} {
		want := make([]byte, len(before))
		got := make([]byte, len(before))
		refReplayState(cfg, before, rows, shape, n, want)
		if err := cfg.ReplayState("st_rec_0", before, rows, shape, n, got); err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		for i := range want {
			if want[i] != got[i] {
				t.Fatalf("n=%d: byte %d differs (%x vs %x)", n, i, got[i], want[i])
			}
		}
	}
}
