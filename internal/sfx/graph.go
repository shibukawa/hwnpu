package sfx

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Part is one program of MOSS-SoundEffect: a model for llm.WritePart whose Build is build. Its
// inputs are fp16 and its output is y.
type Part struct {
	Name  string
	build func(g *graph)
}

// The llm.Model methods a part does not use.
func (p *Part) Hidden() int               { return 0 }
func (p *Part) Layers() int               { return 0 }
func (p *Part) Output() (int, bool)       { return 0, false }
func (p *Part) Embedding() string         { return "" }
func (p *Part) Tensors() map[string][]int { return nil }

// Build builds the part's program.
func (p *Part) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Format == "" {
		opts.Format = "fp16"
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	g := &graph{Graph: &blocks.Graph{Name: "sfx " + p.Name, B: b, W: w, Opts: opts}}
	if err := blocks.Run(func() { p.build(g) }); err != nil {
		return nil, out, err
	}
	if g.y == nil {
		return nil, out, fmt.Errorf("sfx %s: no output", p.Name)
	}
	return g.ins, mlpackage.Feature{Name: g.y.Name, DType: mlpackage.Float16, Shape: g.y.Shape}, nil
}

type graph struct {
	*blocks.Graph
	ins []mlpackage.Feature
	y   *mlpackage.Var
}

func (g *graph) input(name string, shape ...int) *mlpackage.Var {
	x := g.B.Input(name, mlpackage.Float16, shape...)
	g.ins = append(g.ins, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: x.Shape})
	return x
}

func (g *graph) output(x *mlpackage.Var) {
	g.y = g.B.NamedOp("identity", "y", map[string]any{"x": x}, mlpackage.Float16, x.Shape...)
}

func (g *graph) op(typ string, inputs map[string]any, shape ...int) *mlpackage.Var {
	return g.Op(typ, inputs, shape...)
}

func (g *graph) unary(typ string, x *mlpackage.Var) *mlpackage.Var {
	return g.op(typ, map[string]any{"x": x}, x.Shape...)
}

func (g *graph) add(x, y *mlpackage.Var) *mlpackage.Var {
	return g.op("add", map[string]any{"x": x, "y": y}, broadcast(x.Shape, y.Shape)...)
}

func (g *graph) sub(x, y *mlpackage.Var) *mlpackage.Var {
	return g.op("sub", map[string]any{"x": x, "y": y}, broadcast(x.Shape, y.Shape)...)
}

func (g *graph) mul(x, y *mlpackage.Var) *mlpackage.Var {
	return g.op("mul", map[string]any{"x": x, "y": y}, broadcast(x.Shape, y.Shape)...)
}

func (g *graph) scale(x *mlpackage.Var, s float32) *mlpackage.Var {
	return g.op("mul", map[string]any{"x": x, "y": g.B.ScalarF16(s)}, x.Shape...)
}

func (g *graph) addScalar(x *mlpackage.Var, s float32) *mlpackage.Var {
	return g.op("add", map[string]any{"x": x, "y": g.B.ScalarF16(s)}, x.Shape...)
}

func (g *graph) clip(x *mlpackage.Var, lo, hi float32) *mlpackage.Var {
	return g.op("clip", map[string]any{"x": x, "alpha": g.B.ScalarF16(lo), "beta": g.B.ScalarF16(hi)}, x.Shape...)
}

func (g *graph) reshape(x *mlpackage.Var, shape ...int) *mlpackage.Var {
	s := make([]int32, len(shape))
	for i, d := range shape {
		s[i] = int32(d)
	}
	return g.op("reshape", map[string]any{"x": x, "shape": g.B.Ints(s...)}, shape...)
}

func (g *graph) transpose(x *mlpackage.Var, perm ...int) *mlpackage.Var {
	p := make([]int32, len(perm))
	shape := make([]int, len(perm))
	for i, a := range perm {
		p[i] = int32(a)
		shape[i] = x.Shape[a]
	}
	return g.op("transpose", map[string]any{"x": x, "perm": g.B.Ints(p...)}, shape...)
}

// slice is x[..., lo:hi] on axis (negative from the end).
func (g *graph) slice(x *mlpackage.Var, axis, lo, hi int) *mlpackage.Var {
	n := len(x.Shape)
	if axis < 0 {
		axis += n
	}
	begin, end := make([]int32, n), make([]int32, n)
	bm, em := make([]bool, n), make([]bool, n)
	shape := append([]int(nil), x.Shape...)
	for i := range n {
		bm[i], em[i] = i != axis, i != axis
	}
	begin[axis], end[axis] = int32(lo), int32(hi)
	shape[axis] = hi - lo
	return g.op("slice_by_index", map[string]any{"x": x, "begin": g.B.Ints(begin...), "end": g.B.Ints(end...),
		"begin_mask": g.Bools(bm...), "end_mask": g.Bools(em...)}, shape...)
}

func (g *graph) concat(axis int, xs ...*mlpackage.Var) *mlpackage.Var {
	shape := append([]int(nil), xs[0].Shape...)
	a := axis
	if a < 0 {
		a += len(shape)
	}
	shape[a] = 0
	for _, x := range xs {
		shape[a] += x.Shape[a]
	}
	return g.op("concat", map[string]any{"values": xs, "axis": g.B.ScalarI32(int32(axis)), "interleave": g.B.ScalarBool(false)}, shape...)
}

func broadcast(a, b []int) []int {
	if len(b) > len(a) {
		a, b = b, a
	}
	out := append([]int(nil), a...)
	off := len(a) - len(b)
	for i, d := range b {
		if d > out[off+i] {
			out[off+i] = d
		}
	}
	return out
}
