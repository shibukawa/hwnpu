package mxfp4

import (
	"math/rand/v2"
	"strconv"
	"testing"
	"time"
)

// TestTokenTime times a token's experts at gpt-oss-20b's shape (HWNPU_MXFP4=bench=LAYERS): 4 of
// 32 experts a layer, gate_up 5760 rows and down 2880 of 2880, over weights larger than the
// caches.
func TestTokenTime(t *testing.T) {
	n, _ := Opt("bench")
	layers, _ := strconv.Atoi(n)
	if layers == 0 {
		t.Skip("HWNPU_MXFP4=bench=LAYERS")
	}
	const H, I, E, K = 2880, 2880, 32, 4
	nb := H / Block
	type expert struct{ gw, ge, dw, de []byte }
	r := rand.New(rand.NewPCG(1, 2))
	ex := make([][]expert, layers)
	for l := range ex {
		ex[l] = make([]expert, E)
		for i := range ex[l] {
			gw, ge := randomRow(r, 2*I*nb)
			dw, de := randomRow(r, H*(I/Block))
			ex[l][i] = expert{gw, ge, dw, de}
		}
	}
	x := make([]float32, H)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	v := NewVec(H)
	hv := []*Vec{NewVec(I), NewVec(I), NewVec(I), NewVec(I)}
	gu, h, y := make([][]float32, K), make([]float32, I), make([][]float32, K)
	for k := range K {
		gu[k], y[k] = make([]float32, 2*I), make([]float32, H)
	}
	const tokens = 20
	start := time.Now()
	for range tokens {
		for l := range layers {
			v.Quantize(x)
			var chosen [K]expert
			jobs := make([]Job, K)
			for k := range K {
				chosen[k] = ex[l][r.IntN(E)]
				jobs[k] = Job{Out: [][]float32{gu[k]}, W: chosen[k].gw, E: chosen[k].ge, V: []*Vec{v}}
			}
			Run(jobs...)
			for k := range K {
				for j := range h {
					h[j] = gu[k][2*j] * gu[k][2*j+1] * 1e-3
				}
				hv[k].Quantize(h)
				jobs[k] = Job{Out: [][]float32{y[k]}, W: chosen[k].dw, E: chosen[k].de, V: []*Vec{hv[k]}}
			}
			Run(jobs...)
		}
	}
	per := time.Since(start) / tokens
	t.Logf("%d layers: %v a token, %v a layer; 24 layers: %.1f tokens/s", layers, per, per/time.Duration(layers), 1/(per.Seconds()*24/float64(layers)))
}
