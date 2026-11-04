package paged

import (
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// The program, called once per chunk and merged, attends as one softmax over the whole cache:
// chunks wholly visible, one partly visible with a causal mask over several query rows, and one
// wholly masked, for both forms and for heads that share a kv head or not.
func TestMergeMatchesFull(t *testing.T) {
	for _, sh := range []Shape{{16, 4, 32}, {16, 8, 8}, {4, 4, 8}} {
		for _, form := range []string{"grouped", "tile"} {
			t.Run(fmt.Sprintf("%v %s", sh, form), func(t *testing.T) { checkMerge(t, sh, form) })
		}
	}
}

func checkMerge(t *testing.T, sh Shape, form string) {
	const chunk, chunks, tokens = 16, 4, 3
	const cache = chunk * chunks
	const first = 37 // row i sees the keys before first+i+1: chunk 2 partly, chunk 3 not at all
	nh, nkv, d := sh.Heads, sh.KVHeads, sh.HeadDim
	rng := rand.New(rand.NewPCG(3, 4))
	normal := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	q, k, v := normal(nh*tokens*d), normal(nkv*cache*d), normal(nkv*cache*d)
	visible := func(row, pos int) bool { return pos < first+row+1 }
	want := make([]float64, nh*tokens*d)
	rep := nh / nkv
	for h := 0; h < nh; h++ {
		kh := h / rep
		for r := 0; r < tokens; r++ {
			qr := q[(h*tokens+r)*d:][:d]
			scores := make([]float64, cache)
			top := math.Inf(-1)
			for p := 0; p < cache; p++ {
				if !visible(r, p) {
					scores[p] = math.Inf(-1)
					continue
				}
				var dot float64
				for i, x := range qr {
					dot += float64(x) * float64(k[(kh*cache+p)*d+i])
				}
				scores[p] = dot / math.Sqrt(float64(d))
				top = max(top, scores[p])
			}
			var sum float64
			out := want[(h*tokens+r)*d:][:d]
			for p, s := range scores {
				if math.IsInf(s, -1) {
					continue
				}
				e := math.Exp(s - top)
				sum += e
				for i := range out {
					out[i] += e * float64(v[(kh*cache+p)*d+i])
				}
			}
			for i := range out {
				out[i] /= sum
			}
		}
	}
	dir := filepath.Join(t.TempDir(), "attn.mlpackage")
	if _, _, err := Write(dir, sh, form, tokens, chunk, ""); err != nil {
		t.Fatal(err)
	}
	p, f, err := milcpu.OpenPackage(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	merge := NewMerge(nh*tokens, d)
	for j := 0; j < chunks; j++ {
		part := func(x []float32) []float32 {
			out := make([]float32, 0, nkv*chunk*d)
			for h := 0; h < nkv; h++ {
				out = append(out, x[(h*cache+j*chunk)*d:(h*cache+(j+1)*chunk)*d]...)
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
			"q":    {Shape: []int{1, nh, tokens, d}, Data: q},
			"k":    {Shape: []int{1, nkv, chunk, d}, Data: part(k)},
			"v":    {Shape: []int{1, nkv, chunk, d}, Data: part(v)},
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
		e := float64(got[i]) - want[i]
		num += e * e
		den += want[i] * want[i]
	}
	if e := math.Sqrt(num / den); e > 5e-3 || math.IsNaN(e) {
		t.Errorf("relative error %g against one softmax over the cache", e)
	}
}

// The merge does not depend on the order of the chunks: a later chunk with the largest score
// rescales what came before.
func TestMergeOrder(t *testing.T) {
	o := [][]float32{{1, 2}, {3, 4}, {5, 6}}
	m := [][]float32{{-1}, {4}, {2}}
	l := [][]float32{{2}, {1}, {3}}
	var w, x0, x1 float64
	for j := range o {
		wj := float64(l[j][0]) * math.Exp(float64(m[j][0])-4)
		w += wj
		x0 += wj * float64(o[j][0])
		x1 += wj * float64(o[j][1])
	}
	for _, order := range [][]int{{0, 1, 2}, {2, 1, 0}, {1, 0, 2}} {
		c := NewMerge(1, 2)
		for _, j := range order {
			c.Add(o[j], m[j], l[j])
		}
		out := make([]float32, 2)
		c.Result(out)
		if math.Abs(float64(out[0])-x0/w) > 1e-5 || math.Abs(float64(out[1])-x1/w) > 1e-5 {
			t.Errorf("order %v: merged %v, want [%g %g]", order, out, x0/w, x1/w)
		}
	}
}
