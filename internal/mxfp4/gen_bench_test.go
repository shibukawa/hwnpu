package mxfp4

import (
	"math/rand/v2"
	"os"
	"testing"
	"time"
)

// BenchmarkGenLayer is a generated token's experts of a layer: 4 experts' gate_up (5760 rows of
// 2880) and down (2880 of 2880) with one vector each, 53 MB (out of the caches), over the cores.
func BenchmarkGenLayer(b *testing.B) {
	r := rand.New(rand.NewPCG(7, 8))
	const nb = 90
	type mat struct{ w, e []byte }
	var layers [][]mat
	for range 6 { // 318 MB: past the SLC between iterations
		var ms []mat
		for range 4 {
			for _, rows := range []int{5760, 2880} {
				w, e := randomRow(r, rows*nb)
				ms = append(ms, mat{w, e})
			}
		}
		layers = append(layers, ms)
	}
	x := make([]float32, nb*Block)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	v := NewVec(len(x))
	v.Quantize(x)
	outs := [][]float32{make([]float32, 5760), make([]float32, 2880)}
	b.SetBytes(int64(4 * (5760 + 2880) * nb * 17))
	// HWNPU_GPTOSS_GAP: a pause between layers (the Neural Engine's part), not timed
	gap, _ := time.ParseDuration(os.Getenv("HWNPU_GPTOSS_GAP"))
	i := 0
	for b.Loop() {
		if gap > 0 {
			b.StopTimer()
			for t0 := time.Now(); time.Since(t0) < gap; {
				time.Sleep(gap / 4)
			}
			b.StartTimer()
		}
		ms := layers[i%len(layers)]
		i++
		var gu, dn []Job
		for k, m := range ms {
			j := Job{Out: [][]float32{outs[k%2]}, W: m.w, E: m.e, V: []*Vec{v}}
			if k%2 == 0 {
				gu = append(gu, j)
			} else {
				dn = append(dn, j)
			}
		}
		Run(gu...)
		Run(dn...)
	}
}
