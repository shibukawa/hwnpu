package qwen35

import (
	"fmt"
	"sync"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/syncx"
)

type replayScratch struct {
	s     []float32
	r     []float32
	delta []float32
}

var replayScratchPool syncx.Pool[replayScratch]

func getReplayScratch(sLen, rLen, dLen int) *replayScratch {
	sc := replayScratchPool.Get()
	if sc == nil {
		sc = &replayScratch{}
	}
	if cap(sc.s) < sLen {
		sc.s = make([]float32, sLen)
	} else {
		sc.s = sc.s[:sLen]
	}
	if cap(sc.r) < rLen {
		sc.r = make([]float32, rLen)
	} else {
		sc.r = sc.r[:rLen]
	}
	if cap(sc.delta) < dLen {
		sc.delta = make([]float32, dLen)
	} else {
		sc.delta = sc.delta[:dLen]
	}
	return sc
}

// ReplayState brings a Gated DeltaNet's recurrent state st_rec_N from before a step to after its
// first n rows (llm.StateReplayer), from the program's st_rec_N_rows (1, value heads, rows,
// dk+dv+2): each row's key k (L2 normalized, the head's key head), value v, beta and decay, as
// the program's deltaRuleChunks takes them. For each row, in float32:
//
//	S = S * decay;  S = S + k^T ((v - k S) * beta)
//
// The heads are independent and run in parallel. The scale (S *= decay) and dot-product
// accumulation (delta -= k * S) are fused into one pass over S so S is streamed twice per row
// instead of three times, with pooled per-head scratch buffers.
func (c *Config) ReplayState(name string, before, rows []byte, rowShape []int, n int, after []byte) error {
	hv, dk, dv := c.LinearValueHeads, c.LinearKeyDim, c.LinearValueDim
	w := dk + dv + 2
	if len(rowShape) != 4 || rowShape[1] != hv || rowShape[3] != w {
		return fmt.Errorf("qwen35: %s_rows has shape %v, want (1, %d, rows, %d)", name, rowShape, hv, w)
	}
	q := rowShape[2]
	if n < 0 || n > q || len(rows) != 2*hv*q*w || len(before) != 2*hv*dk*dv || len(after) != len(before) {
		return fmt.Errorf("qwen35: %s: %d rows of %d, %d bytes of state", name, n, q, len(before))
	}
	var wg sync.WaitGroup
	for h := 0; h < hv; h++ {
		wg.Add(1)
		go func(h int) {
			defer wg.Done()
			sc := getReplayScratch(dk*dv, n*w, dv)
			defer replayScratchPool.Put(sc)
			s, r, delta := sc.s, sc.r, sc.delta
			hwnpu.DecodeF16(s, before[2*h*dk*dv:2*(h+1)*dk*dv])
			if n > 0 {
				hwnpu.DecodeF16(r, rows[2*h*q*w:2*(h*q+n)*w])
			}
			for t := 0; t < n; t++ {
				row := r[t*w : (t+1)*w]
				k, v, beta, decay := row[:dk], row[dk:dk+dv], row[dk+dv], row[dk+dv+1]
				copy(delta, v)
				// Pass 1 (fused): S *= decay and delta -= k * S
				for a := 0; a < dk; a++ {
					ka := k[a]
					sa := s[a*dv : (a+1)*dv]
					for e := 0; e < dv; e++ {
						vse := sa[e] * decay
						sa[e] = vse
						delta[e] -= ka * vse
					}
				}
				for e := 0; e < dv; e++ {
					delta[e] *= beta
				}
				// Pass 2: S += k^T * delta
				for a := 0; a < dk; a++ {
					ka := k[a]
					sa := s[a*dv : (a+1)*dv]
					for e := 0; e < dv; e++ {
						sa[e] += ka * delta[e]
					}
				}
			}
			hwnpu.EncodeF16(after[2*h*dk*dv:2*(h+1)*dk*dv], s)
		}(h)
	}
	wg.Wait()
	return nil
}

var _ llm.StateReplayer = (*Config)(nil)
