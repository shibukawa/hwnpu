package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// DitScale is what the DiT's residual stream is divided by in its programs: its largest value
// measured (2.7e5, blocks 12 to 20 at 512 x 512, design.md 16.69) becomes 8.4e3, and the
// largest gated branch (2.3e5) 7.3e3.
const DitScale = 32

// Part is one program of Anima: a model for llm.WritePart whose Build is build.
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

// Build builds the part's program: its inputs, and its output y.
func (p *Part) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Format == "" {
		opts.Format = "fp16"
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	g := &graph{Graph: &blocks.Graph{Name: "anima " + p.Name, B: b, W: w, Opts: opts}}
	err = blocks.Run(func() { p.build(g) })
	if err != nil {
		return nil, out, err
	}
	if g.y == nil {
		return nil, out, fmt.Errorf("anima %s: no output", p.Name)
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

func (g *graph) add(x, y *mlpackage.Var) *mlpackage.Var {
	return g.op("add", map[string]any{"x": x, "y": y}, broadcast(x.Shape, y.Shape)...)
}

func (g *graph) mul(x, y *mlpackage.Var) *mlpackage.Var {
	return g.op("mul", map[string]any{"x": x, "y": y}, broadcast(x.Shape, y.Shape)...)
}

func (g *graph) scale(x *mlpackage.Var, s float32) *mlpackage.Var {
	return g.op("mul", map[string]any{"x": x, "y": g.B.ScalarF16(s)}, x.Shape...)
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

// maxScaled divides each row of x (over axis) by its largest magnitude (at least 1e-3), so a
// norm after it squares nothing larger than 1: a norm does not change with its input's scale.
func (g *graph) maxScaled(x *mlpackage.Var, axis int) *mlpackage.Var {
	neg := g.scale(x, -1)
	abs := g.op("maximum", map[string]any{"x": x, "y": neg}, x.Shape...)
	rs := append([]int(nil), x.Shape...)
	a := axis
	if a < 0 {
		a += len(rs)
	}
	rs[a] = 1
	m := g.op("reduce_max", map[string]any{"x": abs, "axes": g.B.Ints(int32(axis)), "keep_dims": g.B.ScalarBool(true)}, rs...)
	m = g.op("maximum", map[string]any{"x": m, "y": g.B.ScalarF16(1e-3)}, rs...)
	return g.op("real_div", map[string]any{"x": x, "y": m}, x.Shape...)
}

// rmsNorm is the RMS norm over the last axis with the named weight, of x scaled by maxScaled.
func (g *graph) rmsNorm(x *mlpackage.Var, name string, eps float64) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	return g.RMSNormWith(g.maxScaled(x, -1), name, g.Get(name, n), eps, 1)
}

// layerNorm is a LayerNorm over the last axis without weight and bias (eps 1e-6), of x scaled
// by maxScaled.
func (g *graph) layerNorm(x *mlpackage.Var) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	ones, zeros := make([]float32, n), make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	gamma := g.F16s(fmt.Sprintf("ln_ones_%d", n), []int{n}, ones)
	beta := g.F16s(fmt.Sprintf("ln_zeros_%d", n), []int{n}, zeros)
	return g.op("layer_norm", map[string]any{"x": g.maxScaled(x, -1), "axes": g.B.Ints(-1), "gamma": gamma, "beta": beta,
		"epsilon": g.B.ScalarF16(1e-6)}, x.Shape...)
}

// attention is SDPA of q (1, heads, n, d) over k, v (1, kv heads, m, d), the queries chunk at
// a time so that no score matrix is over heads x chunk x m. mask, when not nil, is added to
// the scores (broadcast over the queries).
func (g *graph) attention(q, k, v *mlpackage.Var, scale float64, mask *mlpackage.Var, chunk int) *mlpackage.Var {
	var mf func() *mlpackage.Var
	if mask != nil {
		mf = func() *mlpackage.Var { return mask }
	}
	n := q.Shape[2]
	if chunk <= 0 || n <= chunk {
		return g.SDPAScale(q, k, v, scale, mf)
	}
	var outs []*mlpackage.Var
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		outs = append(outs, g.SDPAScale(g.slice(q, 2, lo, hi), k, v, scale, mf))
	}
	return g.concat(2, outs...)
}

// heads is x (1, n, h*d) as (1, h, n, d).
func (g *graph) heads(x *mlpackage.Var, h, d int) *mlpackage.Var {
	n := x.Shape[1]
	return g.transpose(g.reshape(x, 1, n, h, d), 0, 2, 1, 3)
}

// merge is o (1, h, n, d) as (1, n, h*d).
func (g *graph) merge(o *mlpackage.Var) *mlpackage.Var {
	h, n, d := o.Shape[1], o.Shape[2], o.Shape[3]
	return g.reshape(g.transpose(o, 0, 2, 1, 3), 1, n, h*d)
}

// linearConst is a linear with the given fp16 weight (out, in) and optional bias.
func (g *graph) linearConst(x *mlpackage.Var, name string, w []float32, out int, bias []float32) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	inputs := map[string]any{"x": x, "weight": g.F16s(name, []int{out, in}, w)}
	if bias != nil {
		inputs["bias"] = g.F16s(name+".bias", []int{out}, bias)
	}
	shape := append(append([]int(nil), x.Shape[:len(x.Shape)-1]...), out)
	return g.op("linear", inputs, shape...)
}

func invFreq(dim int, theta float64) []float64 {
	f := make([]float64, dim/2)
	for i := range f {
		f[i] = 1 / math.Pow(theta, float64(2*i)/float64(dim))
	}
	return f
}
