package mxfp4

import (
	"math/rand/v2"
	"testing"
	"time"
)

// TestDotCache times Dot on one core over rows that stay in the cache (1 MB of them, again and
// again) and over rows that do not (256 MB, once through), HWNPU_DEV's mxfp4.bench=cache: what the
// kernel does, and what reading the weights from memory leaves of it.
func TestDotCache(t *testing.T) {
	if v, _ := Opt("bench"); v != "cache" {
		t.Skip("HWNPU_DEV=mxfp4.bench=cache")
	}
	r := rand.New(rand.NewPCG(1, 2))
	for _, c := range []struct{ rows, cols int }{{1024, 2048}, {262144, 2048}, {2048, 512}, {1048576, 512}} {
		rows, cols := c.rows, c.cols
		x := make([]float32, cols)
		for i := range x {
			x[i] = float32(r.NormFloat64())
		}
		v := NewVec(cols)
		v.Quantize(x)
		nb := cols / Block
		w, e := make([]byte, rows*nb*16), make([]byte, rows*nb)
		for i := range w {
			w[i] = byte(r.Uint32())
		}
		for i := range e {
			e[i] = 120
		}
		passes := (1 << 30) / (rows * cols)
		macs := float64(passes) * float64(rows) * float64(cols)
		out := make([]float32, rows)
		for _, k := range []struct {
			name string
			run  func()
		}{{"a call a row", func() {
			for i := range rows {
				out[i] = Dot(w[i*nb*16:(i+1)*nb*16], e[i*nb:(i+1)*nb], v)
			}
		}}, {"a call for 128 KB of rows", func() {
			n := 128 * 2048 / cols
			for i := 0; i < rows; i += n {
				Rows(out[i:i+n], w[i*nb*16:(i+n)*nb*16], e[i*nb:(i+n)*nb], v)
			}
		}}} {
			best := time.Duration(1 << 62)
			for range 3 {
				start := time.Now()
				for range passes {
					k.run()
				}
				best = min(best, time.Since(start))
			}
			t.Logf("%d rows of %d (%d MB of weights), %s: %.1f GMAC/s", rows, cols, len(w)>>20, k.name, macs/best.Seconds()/1e9)
		}
	}
}
