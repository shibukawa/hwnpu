package probe

import (
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// The chunk attention program, called once per chunk and merged on the host, attends as one
// softmax over the whole cache: chunks wholly visible, one partly visible with a causal mask over
// several query rows, and one wholly masked.
func TestChunkAttnMergeMatchesFull(t *testing.T) {
	const chunk, chunks, tokens = 64, 4, 3
	const cache = chunk * chunks
	const first = 148 // row i sees the keys before first+i+1: chunk 2 partly, chunk 3 not at all
	rng := rand.New(rand.NewPCG(3, 4))
	normal := func(n int, scale float64) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(scale * rng.NormFloat64())
		}
		return v
	}
	q := normal(QHeads*tokens*HeadDim, 1)
	k := normal(KVHeads*cache*HeadDim, 1) // (kv head, position, dim)
	v := normal(KVHeads*cache*HeadDim, 1)
	visible := func(row, pos int) bool { return pos < first+row+1 }

	// the reference: one softmax over the cache per query head and row, in float64
	want := make([]float64, QHeads*tokens*HeadDim)
	rep := QHeads / KVHeads
	for h := 0; h < QHeads; h++ {
		kh := h / rep
		for r := 0; r < tokens; r++ {
			qr := q[(h*tokens+r)*HeadDim:][:HeadDim]
			scores := make([]float64, cache)
			top := math.Inf(-1)
			for p := 0; p < cache; p++ {
				if !visible(r, p) {
					scores[p] = math.Inf(-1)
					continue
				}
				var dot float64
				for i, x := range qr {
					dot += float64(x) * float64(k[(kh*cache+p)*HeadDim+i])
				}
				scores[p] = dot / math.Sqrt(HeadDim)
				top = max(top, scores[p])
			}
			var sum float64
			out := want[(h*tokens+r)*HeadDim:][:HeadDim]
			for p, s := range scores {
				if math.IsInf(s, -1) {
					continue
				}
				e := math.Exp(s - top)
				sum += e
				for i := range out {
					out[i] += e * float64(v[(kh*cache+p)*HeadDim+i])
				}
			}
			for i := range out {
				out[i] /= sum
			}
		}
	}

	for _, form := range AttnForms {
		t.Run(form, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "attn.mlpackage")
			if err := WriteChunkAttnPackage(dir, form, tokens, chunk); err != nil {
				t.Fatal(err)
			}
			p, f, err := milcpu.OpenPackage(dir)
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			merge := NewChunkMerge(QHeads*tokens, HeadDim)
			for j := 0; j < chunks; j++ {
				part := func(x []float32) []float32 { // chunk j of every kv head
					out := make([]float32, 0, KVHeads*chunk*HeadDim)
					for h := 0; h < KVHeads; h++ {
						out = append(out, x[(h*cache+j*chunk)*HeadDim:(h*cache+(j+1)*chunk)*HeadDim]...)
					}
					return out
				}
				mask := make([]float32, tokens*chunk)
				for r := 0; r < tokens; r++ {
					for c := 0; c < chunk; c++ {
						if !visible(r, j*chunk+c) {
							mask[r*chunk+c] = -1e4
						}
					}
				}
				res, err := p.Run(map[string]milcpu.Tensor{
					"q":    {Shape: []int{1, QHeads, tokens, HeadDim}, Data: q},
					"k":    {Shape: []int{1, KVHeads, chunk, HeadDim}, Data: part(k)},
					"v":    {Shape: []int{1, KVHeads, chunk, HeadDim}, Data: part(v)},
					"mask": {Shape: []int{1, 1, tokens, chunk}, Data: mask},
				}, milcpu.Options{})
				if err != nil {
					t.Fatal(err)
				}
				merge.Add(res["o"].Data, res["m"].Data, res["l"].Data)
			}
			got := make([]float32, len(want))
			merge.Result(got)
			var num, den float64
			for i := range want {
				d := float64(got[i]) - want[i]
				num += d * d
				den += want[i] * want[i]
			}
			if e := math.Sqrt(num / den); e > 5e-3 || math.IsNaN(e) {
				t.Errorf("relative error %g against one softmax over the cache", e)
			} else {
				t.Logf("relative error %.2g", e)
			}
		})
	}
}

// The merge does not depend on the order of the chunks: a later chunk with the largest score
// rescales what came before.
func TestChunkMergeOrder(t *testing.T) {
	o := [][]float32{{1, 2}, {3, 4}, {5, 6}}
	m := [][]float32{{-1}, {4}, {2}}
	l := [][]float32{{2}, {1}, {3}}
	var res [][]float32
	for _, order := range [][]int{{0, 1, 2}, {2, 1, 0}, {1, 0, 2}} {
		c := NewChunkMerge(1, 2)
		for _, j := range order {
			c.Add(o[j], m[j], l[j])
		}
		out := make([]float32, 2)
		c.Result(out)
		res = append(res, out)
	}
	// by hand: weights l_j exp(m_j - 4)
	var w, x0, x1 float64
	for j := range o {
		wj := float64(l[j][0]) * math.Exp(float64(m[j][0])-4)
		w += wj
		x0 += wj * float64(o[j][0])
		x1 += wj * float64(o[j][1])
	}
	for _, r := range res {
		if math.Abs(float64(r[0])-x0/w) > 1e-5 || math.Abs(float64(r[1])-x1/w) > 1e-5 {
			t.Errorf("merged %v, want [%g %g] (%v)", r, x0/w, x1/w, fmt.Sprint(res))
		}
	}
}
