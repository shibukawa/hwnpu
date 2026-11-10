// Package milpart has what the MIL programs of the image models share (internal/imagegen/anima/mil,
// internal/imagegen/sana/mil): a Part, one program that llm.WritePart writes as a Core ML package
// or lowers to OpenVINO IR, and a Graph with the tensor operations their layers are written in.
package milpart

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Part is one program of an image model: a model for llm.WritePart whose Build runs Body.
type Part struct {
	Name string
	// Model prefixes the build's errors ("anima", "sana").
	Model string
	Body  func(g *Graph)
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
	g := &Graph{Graph: &blocks.Graph{Name: p.Model + " " + p.Name, B: b, W: w, Opts: opts}}
	err = blocks.Run(func() { p.Body(g) })
	if err != nil {
		return nil, out, err
	}
	if g.y == nil {
		return nil, out, fmt.Errorf("%s %s: no output", p.Model, p.Name)
	}
	return g.ins, mlpackage.Feature{Name: g.y.Name, DType: mlpackage.Float16, Shape: g.y.Shape}, nil
}

// Graph is a part under construction: blocks.Graph and the operations of the image models.
type Graph struct {
	*blocks.Graph
	ins []mlpackage.Feature
	y   *mlpackage.Var
}

// Input adds an fp16 input.
func (g *Graph) Input(name string, shape ...int) *mlpackage.Var {
	x := g.B.Input(name, mlpackage.Float16, shape...)
	g.ins = append(g.ins, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: x.Shape})
	return x
}

// Output makes x the part's output y.
func (g *Graph) Output(x *mlpackage.Var) {
	g.y = g.B.NamedOp("identity", "y", map[string]any{"x": x}, mlpackage.Float16, x.Shape...)
}

func (g *Graph) Add(x, y *mlpackage.Var) *mlpackage.Var {
	return g.Op("add", map[string]any{"x": x, "y": y}, Broadcast(x.Shape, y.Shape)...)
}

func (g *Graph) Sub(x, y *mlpackage.Var) *mlpackage.Var {
	return g.Op("sub", map[string]any{"x": x, "y": y}, Broadcast(x.Shape, y.Shape)...)
}

func (g *Graph) Mul(x, y *mlpackage.Var) *mlpackage.Var {
	return g.Op("mul", map[string]any{"x": x, "y": y}, Broadcast(x.Shape, y.Shape)...)
}

func (g *Graph) Div(x, y *mlpackage.Var) *mlpackage.Var {
	return g.Op("real_div", map[string]any{"x": x, "y": y}, Broadcast(x.Shape, y.Shape)...)
}

// Scale is x times s.
func (g *Graph) Scale(x *mlpackage.Var, s float32) *mlpackage.Var {
	return g.Op("mul", map[string]any{"x": x, "y": g.B.ScalarF16(s)}, x.Shape...)
}

// AddScalar is x plus s.
func (g *Graph) AddScalar(x *mlpackage.Var, s float32) *mlpackage.Var {
	return g.Op("add", map[string]any{"x": x, "y": g.B.ScalarF16(s)}, x.Shape...)
}

// Unary is the elementwise operation typ (relu, silu, exp, tanh...) of x.
func (g *Graph) Unary(typ string, x *mlpackage.Var) *mlpackage.Var {
	return g.Op(typ, map[string]any{"x": x}, x.Shape...)
}

func (g *Graph) Reshape(x *mlpackage.Var, shape ...int) *mlpackage.Var {
	s := make([]int32, len(shape))
	for i, d := range shape {
		s[i] = int32(d)
	}
	return g.Op("reshape", map[string]any{"x": x, "shape": g.B.Ints(s...)}, shape...)
}

func (g *Graph) Transpose(x *mlpackage.Var, perm ...int) *mlpackage.Var {
	p := make([]int32, len(perm))
	shape := make([]int, len(perm))
	for i, a := range perm {
		p[i] = int32(a)
		shape[i] = x.Shape[a]
	}
	return g.Op("transpose", map[string]any{"x": x, "perm": g.B.Ints(p...)}, shape...)
}

// Slice is x[..., lo:hi] on axis (negative from the end).
func (g *Graph) Slice(x *mlpackage.Var, axis, lo, hi int) *mlpackage.Var {
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
	return g.Op("slice_by_index", map[string]any{"x": x, "begin": g.B.Ints(begin...), "end": g.B.Ints(end...),
		"begin_mask": g.Bools(bm...), "end_mask": g.Bools(em...)}, shape...)
}

func (g *Graph) Concat(axis int, xs ...*mlpackage.Var) *mlpackage.Var {
	shape := append([]int(nil), xs[0].Shape...)
	a := axis
	if a < 0 {
		a += len(shape)
	}
	shape[a] = 0
	for _, x := range xs {
		shape[a] += x.Shape[a]
	}
	return g.Op("concat", map[string]any{"values": xs, "axis": g.B.ScalarI32(int32(axis)), "interleave": g.B.ScalarBool(false)}, shape...)
}

// Reduce is reduce_<typ> (sum, mean, max) of x over axis, kept as size 1.
func (g *Graph) Reduce(typ string, x *mlpackage.Var, axis int) *mlpackage.Var {
	rs := append([]int(nil), x.Shape...)
	a := axis
	if a < 0 {
		a += len(rs)
	}
	rs[a] = 1
	return g.Op("reduce_"+typ, map[string]any{"x": x, "axes": g.B.Ints(int32(axis)), "keep_dims": g.B.ScalarBool(true)}, rs...)
}

// MatMul is x times y, either transposed first.
func (g *Graph) MatMul(x, y *mlpackage.Var, tx, ty bool) *mlpackage.Var {
	n := len(x.Shape)
	shape := append([]int(nil), Broadcast(x.Shape[:n-2], y.Shape[:len(y.Shape)-2])...)
	rows, cols := x.Shape[n-2], y.Shape[len(y.Shape)-1]
	if tx {
		rows = x.Shape[n-1]
	}
	if ty {
		cols = y.Shape[len(y.Shape)-2]
	}
	shape = append(shape, rows, cols)
	return g.Op("matmul", map[string]any{"x": x, "y": y, "transpose_x": g.B.ScalarBool(tx), "transpose_y": g.B.ScalarBool(ty)}, shape...)
}

// Broadcast is the shape of an elementwise operation of shapes a and b.
func Broadcast(a, b []int) []int {
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

// MaxScaled divides each row of x (over axis) by its largest magnitude (at least 1e-3), so a
// norm after it squares nothing larger than 1: a norm does not change with its input's scale.
func (g *Graph) MaxScaled(x *mlpackage.Var, axis int) *mlpackage.Var {
	neg := g.Scale(x, -1)
	abs := g.Op("maximum", map[string]any{"x": x, "y": neg}, x.Shape...)
	m := g.Reduce("max", abs, axis)
	m = g.Op("maximum", map[string]any{"x": m, "y": g.B.ScalarF16(1e-3)}, m.Shape...)
	return g.Op("real_div", map[string]any{"x": x, "y": m}, x.Shape...)
}

// ScaledRMSNorm is the RMS norm over the last axis with the named weight, of x scaled by MaxScaled.
func (g *Graph) ScaledRMSNorm(x *mlpackage.Var, name string, eps float64) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	return g.RMSNormWith(g.MaxScaled(x, -1), name, g.Get(name, n), eps, 1)
}

// ScaledLayerNorm is a LayerNorm over the last axis without weight and bias, of x scaled by
// MaxScaled.
func (g *Graph) ScaledLayerNorm(x *mlpackage.Var, eps float64) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	ones, zeros := make([]float32, n), make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	gamma := g.F16s(fmt.Sprintf("ln_ones_%d", n), []int{n}, ones)
	beta := g.F16s(fmt.Sprintf("ln_zeros_%d", n), []int{n}, zeros)
	return g.Op("layer_norm", map[string]any{"x": g.MaxScaled(x, -1), "axes": g.B.Ints(-1), "gamma": gamma, "beta": beta,
		"epsilon": g.B.ScalarF16(float32(eps))}, x.Shape...)
}

// QChunkAttention is SDPA of q (1, heads, n, d) over k, v (1, kv heads, m, d), the queries chunk
// at a time so that no score matrix is over heads x chunk x m. mask, when not nil, is added to
// the scores (broadcast over the queries).
func (g *Graph) QChunkAttention(q, k, v *mlpackage.Var, scale float64, mask *mlpackage.Var, chunk int) *mlpackage.Var {
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
		outs = append(outs, g.SDPAScale(g.Slice(q, 2, lo, hi), k, v, scale, mf))
	}
	return g.Concat(2, outs...)
}

// Heads is x (1, n, h*d) as (1, h, n, d).
func (g *Graph) Heads(x *mlpackage.Var, h, d int) *mlpackage.Var {
	n := x.Shape[1]
	return g.Transpose(g.Reshape(x, 1, n, h, d), 0, 2, 1, 3)
}

// Merge is o (1, h, n, d) as (1, n, h*d).
func (g *Graph) Merge(o *mlpackage.Var) *mlpackage.Var {
	h, n, d := o.Shape[1], o.Shape[2], o.Shape[3]
	return g.Reshape(g.Transpose(o, 0, 2, 1, 3), 1, n, h*d)
}

// LinearConst is a linear with the given fp16 weight (out, in) and optional bias.
func (g *Graph) LinearConst(x *mlpackage.Var, name string, w []float32, out int, bias []float32) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	inputs := map[string]any{"x": x, "weight": g.F16s(name, []int{out, in}, w)}
	if bias != nil {
		inputs["bias"] = g.F16s(name+".bias", []int{out}, bias)
	}
	shape := append(append([]int(nil), x.Shape[:len(x.Shape)-1]...), out)
	return g.Op("linear", inputs, shape...)
}

// Conv2D is a 2 D convolution of x (1, cin, H, W) with the fp16 weight wt (cout, cin/groups, k,
// k) and the optional bias, zero padded to the same size, stride 1.
func (g *Graph) Conv2D(x *mlpackage.Var, name string, wt, bias []float32, cout, k, groups int) *mlpackage.Var {
	cin, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	if len(wt) != cout*cin/groups*k*k {
		g.Fail("%s: %d weights for %d to %d channels, %d x %d in %d groups", name, len(wt), cin, cout, k, k, groups)
	}
	p := int32(k / 2)
	inputs := map[string]any{
		"x": x, "weight": g.F16s(name+".k", []int{cout, cin / groups, k, k}, wt),
		"strides": g.B.Ints(1, 1), "pad_type": g.Str("custom"), "pad": g.B.Ints(p, p, p, p), "dilations": g.B.Ints(1, 1), "groups": g.B.ScalarI32(int32(groups)),
	}
	if bias != nil {
		inputs["bias"] = g.F16s(name+".b", []int{cout}, bias)
	}
	return g.Op("conv", inputs, 1, cout, H, W)
}

// InvFreq is RoPE's inverse frequencies of a head of dim at theta.
func InvFreq(dim int, theta float64) []float64 {
	f := make([]float64, dim/2)
	for i := range f {
		f[i] = 1 / math.Pow(theta, float64(2*i)/float64(dim))
	}
	return f
}
