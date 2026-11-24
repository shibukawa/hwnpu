package milpart

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// noWeights is the weights of a part that reads none.
type noWeights struct{}

func (noWeights) Get(name string) ([]float32, []int, error) { return nil, nil, errors.New(name) }
func (noWeights) Rows(name string, _, _ int) ([]float32, error) {
	return nil, errors.New(name)
}
func (noWeights) Source() string { return "test" }
func (noWeights) Close() error   { return nil }

// TestHeadAttention checks that HeadAttention (a head and a chunk of queries at a time, the
// softmax's sum divided before the product with the values or after it, a bias on the keys, the
// keys in two parts with the bias on the second) gives what QChunkAttention gives, on the CPU
// interpreter: chunks that do not divide the queries, more keys than queries and fewer, and
// PieceKeys keys or more (each part's softmax by itself, joined by their log-sum-exp).
func TestHeadAttention(t *testing.T) {
	const heads, d = 3, 4
	for _, c := range []struct {
		n, m, chunk int
		masked      int // the last keys the bias hides (0: no bias)
	}{{10, 7, 4, 0}, {6, 9, 0, 0}, {5, 5, 5, 0}, {8, 3, 1, 0}, {10, 7, 4, 3}, {6, 9, 0, 1}, {4, 6, 3, 5},
		{5, PieceKeys + 512, 2, 0}, {3, PieceKeys + 512, 0, 300}, {4, PieceKeys, 3, 0}} {
		run := func(name string, body func(g *Graph, q, k, v, bias *mlpackage.Var) *mlpackage.Var) []float32 {
			p := &Part{Name: name, Model: "test", Body: func(g *Graph) {
				q := g.Input("q", 1, heads, c.n, d)
				k := g.Input("k", 1, heads, c.m, d)
				v := g.Input("v", 1, heads, c.m, d)
				var bias *mlpackage.Var
				if c.masked > 0 {
					bias = g.Input("kbias", 1, 1, 1, c.m)
				}
				g.Output(body(g, q, k, v, bias))
			}}
			path := filepath.Join(t.TempDir(), name+".mlpackage")
			if _, err := llm.WritePart(path, p, noWeights{}, llm.Options{Backend: llm.BackendCoreML}, nil); err != nil {
				t.Fatal(err)
			}
			prog, closer, err := milcpu.OpenPackage(path)
			if err != nil {
				t.Fatal(err)
			}
			defer closer.Close()
			rng := rand.New(rand.NewSource(1))
			in := map[string]milcpu.Tensor{}
			for _, x := range []struct {
				name string
				rows int
				s    float64
			}{{"q", c.n, 1.5}, {"k", c.m, 1.5}, {"v", c.m, 40}} {
				data := make([]float32, heads*x.rows*d)
				for i := range data {
					data[i] = float32(rng.NormFloat64() * x.s)
				}
				in[x.name] = milcpu.Tensor{Shape: []int{1, heads, x.rows, d}, Data: data}
			}
			if c.masked > 0 {
				b := make([]float32, c.m)
				for i := c.m - c.masked; i < c.m; i++ {
					b[i] = -10000
				}
				in["kbias"] = milcpu.Tensor{Shape: []int{1, 1, 1, c.m}, Data: b}
			}
			out, err := prog.Run(in, milcpu.Options{})
			if err != nil {
				t.Fatal(err)
			}
			return out["y"].Data
		}
		want := run("sdpa", func(g *Graph, q, k, v, bias *mlpackage.Var) *mlpackage.Var {
			return g.QChunkAttention(q, k, v, 1, bias, 0)
		})
		for _, o := range []HeadOptions{{}, {Whole: true}, {Deferred: true}, {Split: 1}, {Split: c.m - max(c.masked, 1)}, {Split: 1, Whole: true}} {
			got := run("heads", func(g *Graph, q, k, v, bias *mlpackage.Var) *mlpackage.Var {
				o.Chunk, o.Bias = c.chunk, bias
				if o.Split > 0 && bias != nil { // the bias of the keys past the split
					o.Bias = g.Slice(bias, -1, o.Split, c.m)
				}
				return g.HeadAttention(q, k, v, o)
			})
			if len(got) != len(want) || len(got) != heads*c.n*d {
				t.Fatalf("%+v: %d values, want %d", c, len(got), len(want))
			}
			for i := range got {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-3*(1+math.Abs(float64(want[i]))) {
					t.Fatalf("%+v, deferred %v, split %d: value %d is %g, want %g", c, o.Deferred, o.Split, i, got[i], want[i])
				}
			}
		}
	}
}

// TestKeyParts checks that KeyParts covers the keys once, in order, in parts of powers of two (but
// the last) of 4096 at most.
func TestKeyParts(t *testing.T) {
	for _, c := range []struct {
		lo, hi int
		want   []int
	}{{0, 4096, []int{4096}}, {4096, 4608, []int{512}}, {0, 2304, []int{2048, 256}}, {0, 3600, []int{2048, 1024, 528}},
		{0, 9216, []int{4096, 4096, 1024}}, {0, 7, []int{4, 2, 1}}, {0, 1, []int{1}}, {3, 8194, []int{4096, 2048, 1024, 1023}}} {
		at := c.lo
		got := KeyParts(c.lo, c.hi)
		var ws []int
		for _, p := range got {
			if p[0] != at || p[1] <= p[0] || p[1]-p[0] > maxKeyPart {
				t.Fatalf("%d..%d: %v", c.lo, c.hi, got)
			}
			ws = append(ws, p[1]-p[0])
			at = p[1]
		}
		if at != c.hi || fmt.Sprint(ws) != fmt.Sprint(c.want) {
			t.Errorf("%d..%d: parts of %v, want %v", c.lo, c.hi, ws, c.want)
		}
	}
}

// TestPaddedDeferred checks that a Deferred HeadAttention over queries not a multiple of
// DeferredRows runs on queries padded to one, and that Note is given the real rows alone: each
// head's n rows in all, by chunks or at once.
func TestPaddedDeferred(t *testing.T) {
	const heads, d, m = 3, 4, 7
	for _, c := range []struct{ n, chunk int }{{10, 0}, {10, 4}, {13, 8}, {5, 2}, {16, 0}} {
		var rows []int
		p := &Part{Name: "p", Model: "test", Body: func(g *Graph) {
			q := g.Input("q", 1, heads, c.n, d)
			k := g.Input("k", 1, heads, m, d)
			v := g.Input("v", 1, heads, m, d)
			o := g.HeadAttention(q, k, v, HeadOptions{Deferred: true, Chunk: c.chunk, Note: func(p *mlpackage.Var) { rows = append(rows, p.Shape[2]) }})
			if o.Shape[2] != c.n {
				t.Errorf("%+v: %v", c, o.Shape)
			}
			g.Output(o)
		}}
		if _, err := llm.WritePart(filepath.Join(t.TempDir(), "p.mlpackage"), p, noWeights{}, llm.Options{Backend: llm.BackendCoreML}, nil); err != nil {
			t.Fatal(err)
		}
		total := 0
		for _, r := range rows {
			total += r
		}
		if total != heads*c.n {
			t.Errorf("%+v: Note was given %v rows, want %d in all", c, rows, heads*c.n)
		}
	}
}
