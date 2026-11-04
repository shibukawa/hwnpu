package paged

import (
	"fmt"
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// The mass program, given the rows' normalizers from the attention program over all the chunks
// of summaries (Merge.Norm), gives each summary's softmax summed over the rows of its kv head:
// over two chunks, the first summary of the first chunk not counted (masked), the summaries
// counted with log S added, one padding row left out (r of -1e4); and with unscaled scores (Gemma
// 4's, one kv head for all).
func TestMassMatchesSoftmax(t *testing.T) {
	for _, sh := range []Shape{{Heads: 16, KVHeads: 4, HeadDim: 32}, {Heads: 4, KVHeads: 4, HeadDim: 8}, {Heads: 8, KVHeads: 1, HeadDim: 16, Scale: 1}} {
		for _, form := range []string{"grouped", "tile"} {
			t.Run(fmt.Sprintf("%v %s", sh, form), func(t *testing.T) { checkMass(t, sh, form) })
		}
	}
}

func checkMass(t *testing.T, sh Shape, form string) {
	const chunk, chunks, tokens, valid = 16, 2, 3, 2
	const n = chunk*chunks - 5 // the summaries there are; the rest of the last chunk is masked
	logS := float32(math.Log(8))
	nh, nkv, d := sh.Heads, sh.KVHeads, sh.HeadDim
	rep := nh / nkv
	scale := 1 / math.Sqrt(float64(d))
	if sh.Scale != 0 {
		scale = sh.Scale
	}
	rng := rand.New(rand.NewPCG(5, 6))
	normal := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	q, k, v := normal(nh*tokens*d), normal(nkv*chunk*chunks*d), normal(nkv*chunk*chunks*d)
	counted := func(b int) bool { return b >= 1 && b < n }
	// the reference: each row's softmax over the counted summaries, summed over the valid rows
	want := make([]float64, nkv*chunk*chunks)
	for h := 0; h < nh; h++ {
		g := h / rep
		for r := 0; r < valid; r++ {
			sc := make([]float64, chunk*chunks)
			top := math.Inf(-1)
			for b := range sc {
				if !counted(b) {
					sc[b] = math.Inf(-1)
					continue
				}
				var dot float64
				for i := 0; i < d; i++ {
					dot += float64(q[(h*tokens+r)*d+i]) * float64(k[(g*chunk*chunks+b)*d+i])
				}
				sc[b] = dot*scale + float64(logS)
				top = max(top, sc[b])
			}
			var sum float64
			for _, s := range sc {
				sum += math.Exp(s - top)
			}
			for b, s := range sc {
				want[g*chunk*chunks+b] += math.Exp(s-top) / sum
			}
		}
	}
	dir := t.TempDir()
	apath, mpath := filepath.Join(dir, "attn.mlpackage"), filepath.Join(dir, "mass.mlpackage")
	if _, _, err := Write(apath, sh, form, tokens, chunk, ""); err != nil {
		t.Fatal(err)
	}
	if _, _, err := WriteMass(mpath, sh, form, tokens, chunk, ""); err != nil {
		t.Fatal(err)
	}
	ap, af, err := milcpu.OpenPackage(apath)
	if err != nil {
		t.Fatal(err)
	}
	defer af.Close()
	mp, mf, err := milcpu.OpenPackage(mpath)
	if err != nil {
		t.Fatal(err)
	}
	defer mf.Close()
	part := func(x []float32, j int) []float32 {
		out := make([]float32, 0, nkv*chunk*d)
		for h := 0; h < nkv; h++ {
			out = append(out, x[(h*chunk*chunks+j*chunk)*d:(h*chunk*chunks+(j+1)*chunk)*d]...)
		}
		return out
	}
	masks := make([][]float32, chunks)
	merge := NewMerge(nh*tokens, d)
	for j := range chunks {
		masks[j] = make([]float32, tokens*chunk)
		for r := 0; r < tokens; r++ {
			for c := 0; c < chunk; c++ {
				masks[j][r*chunk+c] = -1e4
				if counted(j*chunk + c) {
					masks[j][r*chunk+c] = logS
				}
			}
		}
		res, err := ap.Run(map[string]milcpu.Tensor{
			"q":    {Shape: []int{1, nh, tokens, d}, Data: q},
			"k":    {Shape: []int{1, nkv, chunk, d}, Data: part(k, j)},
			"v":    {Shape: []int{1, nkv, chunk, d}, Data: part(v, j)},
			"mask": {Shape: []int{1, 1, tokens, chunk}, Data: masks[j]},
		}, milcpu.Options{})
		if err != nil {
			t.Fatal(err)
		}
		merge.Add(res["o"].Data, res["m"].Data, res["l"].Data)
	}
	norm := make([]float32, nh*tokens)
	merge.Norm(norm)
	for h := 0; h < nh; h++ {
		for r := 0; r < tokens; r++ {
			if r < valid {
				norm[h*tokens+r] = -norm[h*tokens+r]
			} else {
				norm[h*tokens+r] = -1e4
			}
		}
	}
	var num, den float64
	for j := range chunks {
		res, err := mp.Run(map[string]milcpu.Tensor{
			"q":    {Shape: []int{1, nh, tokens, d}, Data: q},
			"k":    {Shape: []int{1, nkv, chunk, d}, Data: part(k, j)},
			"mask": {Shape: []int{1, 1, tokens, chunk}, Data: masks[j]},
			"r":    {Shape: []int{1, nh, tokens, 1}, Data: norm},
		}, milcpu.Options{})
		if err != nil {
			t.Fatal(err)
		}
		w := res["w"].Data
		if len(w) != nkv*chunk {
			t.Fatalf("w has %d values, want %d", len(w), nkv*chunk)
		}
		for g := 0; g < nkv; g++ {
			for c := 0; c < chunk; c++ {
				e := float64(w[g*chunk+c]) - want[g*chunk*chunks+j*chunk+c]
				num += e * e
				den += want[g*chunk*chunks+j*chunk+c] * want[g*chunk*chunks+j*chunk+c]
			}
		}
	}
	if e := math.Sqrt(num / den); e > 5e-3 || math.IsNaN(e) {
		t.Errorf("relative error %g against the softmax over the summaries", e)
	}
}

// Taking a chunk out of a merge (Sub) leaves the merge of the others, and AddMerge combines two
// merges as one: chunks a, b and c merged, minus b, plus a merge of d, is a, c and d.
func TestMergeSubAndAddMerge(t *testing.T) {
	const rows, dim = 5, 4
	rng := rand.New(rand.NewPCG(7, 8))
	chunk := func(scale float32) (o, m, l []float32) {
		o, m, l = make([]float32, rows*dim), make([]float32, rows), make([]float32, rows)
		for i := range o {
			o[i] = float32(rng.NormFloat64())
		}
		for r := range rows {
			m[r] = scale * float32(rng.NormFloat64())
			l[r] = 1 + 10*rng.Float32()
		}
		return
	}
	type c struct{ o, m, l []float32 }
	var cs []c
	for _, s := range []float32{1, 3, 0.5, 2} {
		o, m, l := chunk(s)
		cs = append(cs, c{o, m, l})
	}
	got := NewMerge(rows, dim)
	for _, x := range cs[:3] {
		got.Add(x.o, x.m, x.l)
	}
	got.Sub(cs[1].o, cs[1].m, cs[1].l)
	other := NewMerge(rows, dim)
	other.Add(cs[3].o, cs[3].m, cs[3].l)
	got.AddMerge(other)
	want := NewMerge(rows, dim)
	for _, i := range []int{0, 2, 3} {
		want.Add(cs[i].o, cs[i].m, cs[i].l)
	}
	g, w := make([]float32, rows*dim), make([]float32, rows*dim)
	got.Result(g)
	want.Result(w)
	for i := range g {
		if math.Abs(float64(g[i]-w[i])) > 1e-4 {
			t.Fatalf("value %d: %g, want %g", i, g[i], w[i])
		}
	}
	gn, wn := make([]float32, rows), make([]float32, rows)
	got.Norm(gn)
	want.Norm(wn)
	for r := range gn {
		if math.Abs(float64(gn[r]-wn[r])) > 1e-4 {
			t.Fatalf("row %d: log sum %g, want %g", r, gn[r], wn[r])
		}
	}
}
