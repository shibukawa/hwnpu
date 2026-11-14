package milpart

import (
	"errors"
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
// softmax's sum divided before the product with the values or after it) gives what
// QChunkAttention gives, on the CPU interpreter: chunks that do not divide the queries, more keys
// than queries and fewer.
func TestHeadAttention(t *testing.T) {
	const heads, d = 3, 4
	for _, c := range []struct{ n, m, chunk int }{{10, 7, 4}, {6, 9, 0}, {5, 5, 5}, {8, 3, 1}} {
		run := func(name string, body func(g *Graph, q, k, v *mlpackage.Var) *mlpackage.Var) []float32 {
			p := &Part{Name: name, Model: "test", Body: func(g *Graph) {
				q := g.Input("q", 1, heads, c.n, d)
				k := g.Input("k", 1, heads, c.m, d)
				v := g.Input("v", 1, heads, c.m, d)
				g.Output(body(g, q, k, v))
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
			out, err := prog.Run(in, milcpu.Options{})
			if err != nil {
				t.Fatal(err)
			}
			return out["y"].Data
		}
		want := run("sdpa", func(g *Graph, q, k, v *mlpackage.Var) *mlpackage.Var { return g.QChunkAttention(q, k, v, 1, nil, 0) })
		for _, deferred := range []bool{false, true} {
			got := run("heads", func(g *Graph, q, k, v *mlpackage.Var) *mlpackage.Var {
				return g.HeadAttention(q, k, v, c.chunk, deferred, nil)
			})
			if len(got) != len(want) || len(got) != heads*c.n*d {
				t.Fatalf("%+v: %d values, want %d", c, len(got), len(want))
			}
			for i := range got {
				if diff := math.Abs(float64(got[i] - want[i])); diff > 1e-3*(1+math.Abs(float64(want[i]))) {
					t.Fatalf("%+v, deferred %v: value %d is %g, want %g", c, deferred, i, got[i], want[i])
				}
			}
		}
	}
}
