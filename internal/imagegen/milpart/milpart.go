// Package milpart has what the MIL programs of the image models share (internal/imagegen/anima/mil,
// chroma/mil, flux2klein/mil): a Part, one program that llm.WritePart writes as a Core ML package
// or lowers to OpenVINO IR, and a Graph with the tensor operations their layers are written in.
package milpart

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/milspec"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Part is one program of an image model: a model for llm.WritePart whose Build runs Body.
type Part struct {
	Name string
	// Model prefixes the build's errors ("anima", "chroma").
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

// InputI32 adds an int32 input (indices, offsets).
func (g *Graph) InputI32(name string, shape ...int) *mlpackage.Var {
	x := g.B.Input(name, mlpackage.Int32, shape...)
	g.ins = append(g.ins, mlpackage.Feature{Name: name, DType: mlpackage.Int32, Shape: x.Shape})
	return x
}

// InputI16 adds an int16 input (indices the Neural Engine takes: it has no int32).
func (g *Graph) InputI16(name string, shape ...int) *mlpackage.Var {
	x := g.B.Input(name, mlpackage.Int16, shape...)
	g.ins = append(g.ins, mlpackage.Feature{Name: name, DType: mlpackage.Int16, Shape: x.Shape})
	return x
}

// InputI8 adds an int8 input (weights passed at run time).
func (g *Graph) InputI8(name string, shape ...int) *mlpackage.Var {
	x := g.B.Input(name, mlpackage.Int8, shape...)
	g.ins = append(g.ins, mlpackage.Feature{Name: name, DType: mlpackage.Int8, Shape: x.Shape})
	return x
}

// InputI4 adds a signed 4-bit input (a Core AI program's: Core ML's model description has none).
func (g *Graph) InputI4(name string, shape ...int) *mlpackage.Var {
	x := g.B.Input(name, milspec.DataType_INT4, shape...)
	g.ins = append(g.ins, mlpackage.Feature{Name: name, DType: milspec.DataType_INT4, Shape: x.Shape})
	return x
}

// InputU4 adds an unsigned 4-bit input, two a byte, the low nibble first (a Core AI program's).
func (g *Graph) InputU4(name string, shape ...int) *mlpackage.Var {
	x := g.B.Input(name, milspec.DataType_UINT4, shape...)
	g.ins = append(g.ins, mlpackage.Feature{Name: name, DType: milspec.DataType_UINT4, Shape: x.Shape})
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

// AbsScaled is MaxScaled with the magnitude as abs, one operation where MaxScaled's maximum of x
// and -x is two: the same values, and on the Neural Engine through Core AI 0.8 ms less of each
// norm over 4096 x 2048 values (Anima's DiT at 1024 x 1024, M1: 4.6 ms of a block's 221 over its
// six norms, design.md 16.201). MaxScaled stays as it is for the programs already exported with
// it: a model takes AbsScaled when its export is next made again.
func (g *Graph) AbsScaled(x *mlpackage.Var, axis int) *mlpackage.Var {
	m := g.Reduce("max", g.Unary("abs", x), axis)
	m = g.Op("maximum", map[string]any{"x": m, "y": g.B.ScalarF16(1e-3)}, m.Shape...)
	return g.Op("real_div", map[string]any{"x": x, "y": m}, x.Shape...)
}

// AbsScaledLayerNorm is ScaledLayerNorm over AbsScaled.
func (g *Graph) AbsScaledLayerNorm(x *mlpackage.Var, eps float64) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	ones, zeros := make([]float32, n), make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	gamma := g.F16s(fmt.Sprintf("ln_ones_%d", n), []int{n}, ones)
	beta := g.F16s(fmt.Sprintf("ln_zeros_%d", n), []int{n}, zeros)
	return g.Op("layer_norm", map[string]any{"x": g.AbsScaled(x, -1), "axes": g.B.Ints(-1), "gamma": gamma, "beta": beta,
		"epsilon": g.B.ScalarF16(float32(eps))}, x.Shape...)
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

// maxKeyPart is the most keys of a tensor of HeadAttention's scores.
const maxKeyPart = 4096

// KeyParts cuts the keys lo..hi-1 into the parts HeadAttention makes score tensors of: powers of
// two, maxKeyPart at most, the largest first, and at most 3 parts past the whole maxKeyParts, the
// last taking what is left (2304 keys: 2048 and 256; 3600: 2048, 1024 and 528).
func KeyParts(lo, hi int) [][2]int {
	var out [][2]int
	for hi-lo > maxKeyPart {
		out = append(out, [2]int{lo, lo + maxKeyPart})
		lo += maxKeyPart
	}
	for n := 0; lo < hi; n++ {
		p := 1
		for 2*p <= hi-lo {
			p *= 2
		}
		if n == 2 {
			p = hi - lo
		}
		out = append(out, [2]int{lo, lo + p})
		lo += p
	}
	return out
}

// HeadOptions are HeadAttention's.
type HeadOptions struct {
	// Chunk is the queries a chunk (0: all of them).
	Chunk int
	// Split, when not 0, is where the keys part in two kinds (an image's, then a text's): Bias
	// is then of the keys past it alone, and KeyParts cuts each kind by itself.
	Split int
	// Bias, when not nil, is added to the scores of every query: 0 for the keys attended to and
	// -10000 for the others, (1, 1, 1, keys), or with Split (1, 1, 1, keys - Split) for the keys
	// past it alone (the others are all attended to).
	Bias *mlpackage.Var
	// Whole keeps the keys in one tensor however many they are (KeyParts).
	Whole bool
	// Deferred divides the softmax's sum after the product with the values, not before: 29 ms
	// of softmax in place of 41 for Anima's self-attention. The product is then of exp(x - max),
	// up to one for each key, so its sum over the keys must fit fp16: use it only where the
	// values and the keys are few and small enough (Anima's cross-attention over 512 text rows:
	// 8.7 at most; its self-attention over 4096 patches reaches fp16's largest). Scaling the
	// values down for it in their projection's int8 weights makes products under fp16's normal
	// range, which the Neural Engine drops (Anima's pictures changed, latent cosine 0.75). The
	// keys are then not cut in parts.
	Deferred bool
	// Note, when not nil, is given each chunk's product before Deferred's division (1, 1,
	// queries, d), for checking its range.
	Note func(*mlpackage.Var)
}

// HeadAttention is the attention of q (1, heads, n, d), which holds the scores' scale, over k and
// v (1, heads, m, d), as the Neural Engine runs it fastest through Core AI: a head at a time and
// a chunk of queries at a time, a chunk's scores laid out (1, queries, 1, keys) with the softmax
// over the last axis. Core AI runs a softmax as passes over the scores (exp(x - max), the sum,
// the division), and those are what an image's self-attention costs: of Anima's 4096 x 4096
// scores of 16 heads the two products are 31 ms and the softmax 59 as QChunkAttention writes
// them, 41 here (M1; every head in one tensor laid out the same way is slower than
// QChunkAttention, design.md 16.202).
//
// The layout is fast when the keys are a power of two, 4096 at most: 8 heads of 4096 queries
// took 0.27 ns a score over 4096 keys, 0.38 to 0.47 over 2304, 3072, 4160, 5120 and 8192, where
// QChunkAttention takes 0.39 to 0.44 over any (M1, design.md 16.211). So the keys are cut in
// such parts (KeyParts), each part's scores a tensor of its own under the one softmax: the
// largest of their maxima, the sum of their sums. Qwen-Image's 4096 image keys and 512 text
// keys (32 heads, 4096 queries): 278 ms in one tensor, 173 in two, QChunkAttention 248. With
// parts the probabilities are the exponentials times the sum's reciprocal: dividing two parts
// by the one sum left the Neural Engine. Past 4096 keys of a kind the parts are not tuned
// (8192 as two parts of 4096: 0.46 ns a score, in one tensor 0.38).
func (g *Graph) HeadAttention(q, k, v *mlpackage.Var, o HeadOptions) *mlpackage.Var {
	heads, n, d := q.Shape[1], q.Shape[2], q.Shape[3]
	m := k.Shape[2]
	if o.Deferred && n%DeferredRows != 0 {
		return g.paddedDeferred(q, k, v, o)
	}
	chunk := o.Chunk
	if chunk <= 0 || chunk > n {
		chunk = n
	}
	if o.Split < 0 || o.Split >= m || (o.Split > 0 && o.Deferred) {
		g.Fail("HeadAttention: split %d of %d keys, deferred %v", o.Split, m, o.Deferred)
		return q
	}
	// the keys' parts: [lo, hi) and their bias (nil: none)
	type span struct {
		lo, hi int
		bias   *mlpackage.Var
	}
	var parts []span
	for _, kind := range [][2]int{{0, o.Split}, {o.Split, m}} {
		if kind[0] == kind[1] {
			continue
		}
		biased := o.Bias != nil && (o.Split == 0 || kind[0] == o.Split)
		cut := [][2]int{kind}
		if !o.Whole && !o.Deferred {
			cut = KeyParts(kind[0], kind[1])
		}
		for _, c := range cut {
			p := span{lo: c[0], hi: c[1]}
			if biased {
				p.bias = o.Bias
				if c[1]-c[0] < o.Bias.Shape[3] {
					p.bias = g.Slice(o.Bias, -1, c[0]-kind[0], c[1]-kind[0])
				}
			}
			parts = append(parts, p)
		}
	}
	var outs []*mlpackage.Var
	for h := 0; h < heads; h++ {
		qh, kh, vh := q, k, v
		if heads > 1 {
			qh, kh, vh = g.Slice(q, 1, h, h+1), g.Slice(k, 1, h, h+1), g.Slice(v, 1, h, h+1)
		}
		ks, vs := []*mlpackage.Var{kh}, []*mlpackage.Var{vh}
		if len(parts) > 1 {
			ks, vs = nil, nil
			for _, p := range parts {
				ks, vs = append(ks, g.Slice(kh, 2, p.lo, p.hi)), append(vs, g.Slice(vh, 2, p.lo, p.hi))
			}
		}
		var rows []*mlpackage.Var
		for lo := 0; lo < n; lo += chunk {
			hi := min(lo+chunk, n)
			c := hi - lo
			qc := qh
			if c < n {
				qc = g.Slice(qh, 2, lo, hi)
			}
			var ss []*mlpackage.Var // the parts' scores (1, queries, 1, keys of the part)
			// joined is the parts' reductions (1, queries, 1, 1) as one: side by side and reduced
			// again. A maximum (or a sum) of two reductions of tensors of unlike widths fails
			// the Neural Engine's compiler (4096 and 1024 keys: "Interleave should be matched
			// for dual-input PEs").
			joined := func(typ string, xs []*mlpackage.Var) *mlpackage.Var {
				if len(xs) == 1 {
					return xs[0]
				}
				return g.Reduce(typ, g.Concat(-1, xs...), -1)
			}
			var pms []*mlpackage.Var
			for i, p := range parts {
				s := g.Reshape(g.MatMul(qc, ks[i], false, true), 1, c, 1, p.hi-p.lo)
				if p.bias != nil {
					s = g.Add(s, p.bias)
				}
				ss = append(ss, s)
				pms = append(pms, g.Reduce("max", s, -1))
			}
			mx := joined("max", pms)
			var es, pss []*mlpackage.Var
			for _, s := range ss {
				e := g.Unary("exp", g.Sub(s, mx))
				es = append(es, e)
				pss = append(pss, g.Reduce("sum", e, -1))
			}
			sum := joined("sum", pss) // (1, queries, 1, 1)
			switch {
			case len(parts) > 1:
				r := g.Op("real_div", map[string]any{"x": g.B.ScalarF16(1), "y": sum}, sum.Shape...)
				var row *mlpackage.Var
				for i, p := range parts {
					pv := g.MatMul(g.Reshape(g.Mul(es[i], r), 1, 1, c, p.hi-p.lo), vs[i], false, false)
					if row == nil {
						row = pv
					} else {
						row = g.Add(row, pv)
					}
				}
				rows = append(rows, row)
			case !o.Deferred:
				rows = append(rows, g.MatMul(g.Reshape(g.Div(es[0], sum), 1, 1, c, m), vh, false, false))
			default:
				p := g.MatMul(g.Reshape(es[0], 1, 1, c, m), vh, false, false)
				if o.Note != nil {
					o.Note(p)
				}
				rows = append(rows, g.Div(p, g.Reshape(sum, 1, 1, c, 1)))
			}
		}
		r := rows[0]
		if len(rows) > 1 {
			r = g.Concat(2, rows...)
		}
		outs = append(outs, r)
	}
	out := outs[0]
	if len(outs) > 1 {
		out = g.Concat(1, outs...)
	}
	if out.Shape[1] != heads || out.Shape[2] != n || out.Shape[3] != d {
		g.Fail("HeadAttention: %v from %v", out.Shape, q.Shape)
	}
	return out
}

// DeferredRows divides the queries of a Deferred HeadAttention: over other counts the Neural
// Engine's compiler fails it from M2 on (h14g, h15g and h16g: "unexpected failure", the whole
// program then on the GPU), where M1's (h13g) compiles it. Anima's cross-attention over its 512
// text rows at 880 x 1168 (4015 patches) and 752 x 1360 (3995); 4014, 4020 failed too, 4016 and
// 4056 did not, and the softmax divided before the product (not Deferred) took 4015 (design.md
// 16.277).
const DeferredRows = 8

// paddedDeferred is a Deferred HeadAttention on q padded with zero rows to a multiple of
// DeferredRows, the padding's rows dropped from the result (and from what Note is given).
func (g *Graph) paddedDeferred(q, k, v *mlpackage.Var, o HeadOptions) *mlpackage.Var {
	heads, n, d := q.Shape[1], q.Shape[2], q.Shape[3]
	pad := (n+DeferredRows-1)/DeferredRows*DeferredRows - n
	z := g.F16s(fmt.Sprintf("head_attention_pad_%d", pad), []int{1, heads, pad, d}, make([]float32, heads*pad*d))
	padded := o
	if o.Chunk >= n {
		padded.Chunk = 0
	}
	if note := o.Note; note != nil {
		// the chunks' products in order, a head's n + pad rows after another's
		seen := 0
		padded.Note = func(p *mlpackage.Var) {
			c := p.Shape[2]
			lo := seen % (n + pad)
			seen += c
			if keep := min(c, n-lo); keep == c {
				note(p)
			} else if keep > 0 {
				note(g.Slice(p, 2, 0, keep))
			}
		}
	}
	return g.Slice(g.HeadAttention(g.Concat(2, q, z), k, v, padded), 2, 0, n)
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
	return g.Conv2DStride(x, name, wt, bias, cout, k, groups, 1)
}

// Conv2DStride is Conv2D with a stride: padded by k / 2 on each side, the output ceil(H /
// stride) x ceil(W / stride) for an odd k.
func (g *Graph) Conv2DStride(x *mlpackage.Var, name string, wt, bias []float32, cout, k, groups, stride int) *mlpackage.Var {
	cin, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	if len(wt) != cout*cin/groups*k*k {
		g.Fail("%s: %d weights for %d to %d channels, %d x %d in %d groups", name, len(wt), cin, cout, k, k, groups)
	}
	p := int32(k / 2)
	inputs := map[string]any{
		"x": x, "weight": g.F16s(name+".k", []int{cout, cin / groups, k, k}, wt),
		"strides": g.B.Ints(int32(stride), int32(stride)), "pad_type": g.Str("custom"), "pad": g.B.Ints(p, p, p, p), "dilations": g.B.Ints(1, 1), "groups": g.B.ScalarI32(int32(groups)),
	}
	if bias != nil {
		inputs["bias"] = g.F16s(name+".b", []int{cout}, bias)
	}
	oh, ow := (H+2*int(p)-k)/stride+1, (W+2*int(p)-k)/stride+1
	return g.Op("conv", inputs, 1, cout, oh, ow)
}

// InvFreq is RoPE's inverse frequencies of a head of dim at theta.
func InvFreq(dim int, theta float64) []float64 {
	f := make([]float64, dim/2)
	for i := range f {
		f[i] = 1 / math.Pow(theta, float64(2*i)/float64(dim))
	}
	return f
}
