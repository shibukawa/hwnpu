package qwen35

import (
	"fmt"
	"sync"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
)

// ReplayState brings a Gated DeltaNet's recurrent state st_rec_N from before a step to after its
// first n rows (llm.StateReplayer), from the program's st_rec_N_rows (1, value heads, rows,
// dk+dv+2): each row's key k (L2 normalized, the head's key head), value v, beta and decay, as
// the program's deltaRule takes them. For each row, in float32:
//
//	S = S * decay;  S = S + k^T ((v - k S) * beta)
//
// The heads are independent and run in parallel.
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
				// delta = (v - k S) * beta
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
				// S += k^T delta
				for a := 0; a < dk; a++ {
					ka := k[a]
					sa := s[a*dv : (a+1)*dv]
					for e := range sa {
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
