// Package blocks has the layers that the language model architectures share (linear layers in
// the chosen weight format, RMS norm, SwiGLU, RoPE, grouped query attention), written against the
// Core ML program builder. An architecture package embeds Graph and adds its own layers.
//
// Errors inside a build panic with Fail and are turned back into an error by Run, so that the
// layer code reads like the reference forward pass.
package blocks

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/palette"
)

// Graph is a program under construction with its weight source and options.
type Graph struct {
	// Name prefixes error messages, normally the architecture name.
	Name string
	B    *mlpackage.Builder
	W    llm.Weights
	Opts llm.Options
}

type buildError struct{ err error }

// Run calls build and returns the error passed to Fail, if any.
func Run(build func()) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(buildError); ok {
				err = e.err
				return
			}
			panic(r)
		}
	}()
	build()
	return nil
}

// Fail stops the build started by Run with an error.
func (g *Graph) Fail(format string, args ...any) {
	panic(buildError{fmt.Errorf(g.Name+": "+format, args...)})
}

// Get reads a tensor and checks its shape when want is given.
func (g *Graph) Get(name string, want ...int) []float32 {
	v, shape, err := g.W.Get(name)
	if err != nil {
		g.Fail("%v", err)
	}
	if len(want) > 0 {
		ok := len(shape) == len(want)
		for i := range want {
			ok = ok && shape[i] == want[i]
		}
		if !ok {
			g.Fail("%s: shape %v, expected %v", name, shape, want)
		}
	}
	return v
}

// Op adds an fp16 operation.
func (g *Graph) Op(typ string, inputs map[string]any, shape ...int) *mlpackage.Var {
	return g.B.Op(typ, inputs, mlpackage.Float16, shape...)
}

// F16s adds an fp16 constant.
func (g *Graph) F16s(name string, shape []int, vals []float32) *mlpackage.Var {
	v, err := g.B.F16s(name, shape, vals)
	if err != nil {
		g.Fail("%v", err)
	}
	return v
}

// Bools adds a constant bool vector.
func (g *Graph) Bools(v ...bool) *mlpackage.Var {
	x, err := g.B.Bools("c", []int{len(v)}, v)
	if err != nil {
		g.Fail("%v", err)
	}
	return x
}

// Str adds a constant string.
func (g *Graph) Str(s string) *mlpackage.Var {
	return g.B.Str("c", s)
}

// Weight adds a (rows, cols) matrix in the chosen format, multiplied by scale.
func (g *Graph) Weight(name string, rows, cols int, scale float32) *mlpackage.Var {
	return g.matrix(name, name, g.Get(name, rows, cols), rows, cols, scale)
}

// WeightRows is Weight for rows [begin, end) of the named matrix of cols columns: one part of a
// fused projection (PLaMo's qkv_proj and gate_up_proj). The constant is called name_begin_end.
func (g *Graph) WeightRows(name string, begin, end, cols int, scale float32) *mlpackage.Var {
	w, err := g.W.Rows(name, begin, end)
	if err != nil {
		g.Fail("%v", err)
	}
	if len(w) != (end-begin)*cols {
		g.Fail("%s: rows %d..%d have %d values, expected %d columns", name, begin, end, len(w), cols)
	}
	return g.matrix(name, fmt.Sprintf("%s_%d_%d", name, begin, end), w, end-begin, cols, scale)
}

// matrix writes the (rows, cols) values w of the tensor name as the constant constName.
func (g *Graph) matrix(name, constName string, w []float32, rows, cols int, scale float32) *mlpackage.Var {
	if scale != 1 {
		for i := range w {
			w[i] *= scale
		}
	}
	var v *mlpackage.Var
	var err error
	format := g.Opts.FormatOf(name)
	switch format {
	case "fp16":
		v, err = g.B.F16s(constName, []int{rows, cols}, w)
	case "lut4", "lut6", "lut8":
		bits := map[string]int{"lut4": 4, "lut6": 6, "lut8": 8}[format]
		mode := palette.Uniform
		switch g.Opts.Quant {
		case "", "uniform":
		case "kmeans":
			mode = palette.KMeans
		case "kmeans-minmax":
			mode = palette.KMeansMinMax
		default:
			g.Fail("unknown quantizer %q", g.Opts.Quant)
		}
		q := palette.QuantizeBits(w, rows, cols, g.Opts.Group, g.Opts.ColBlock, bits, 8, mode)
		v, err = g.B.LUT(constName, bits, []int{rows, cols}, []int{q.Group, q.ColBlock}, q.Indices, q.LUT)
	default:
		g.Fail("unknown format %q", g.Opts.Format)
	}
	if err != nil {
		g.Fail("%v", err)
	}
	return v
}

// Linear is x (..., in) times the weight (out, in) transposed. outName names the result ("" for
// an automatic name).
func (g *Graph) Linear(x *mlpackage.Var, name, outName string, outDim int) *mlpackage.Var {
	return g.LinearS(x, name, outName, outDim, 1)
}

// LinearS is Linear with the weights multiplied by s, so that the output is s times larger;
// the consumer must undo it (an RMSNorm with inScale s).
func (g *Graph) LinearS(x *mlpackage.Var, name, outName string, outDim int, s float32) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	w := g.Weight(name, outDim, in, s)
	shape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), outDim)
	if outName != "" {
		return g.B.NamedOp("linear", outName, map[string]any{"x": x, "weight": w}, mlpackage.Float16, shape...)
	}
	return g.Op("linear", map[string]any{"x": x, "weight": w}, shape...)
}

// LinearRows is Linear with rows [begin, end) of the named weight: one output of a fused
// projection, end-begin wide.
func (g *Graph) LinearRows(x *mlpackage.Var, name string, begin, end int) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	w := g.WeightRows(name, begin, end, in, 1)
	shape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), end-begin)
	return g.Op("linear", map[string]any{"x": x, "weight": w}, shape...)
}

// RMSNorm normalizes over the last axis (size n) and scales by the named weight. inScale says by
// how much the input has been scaled up by its producer (LinearS), 1 if not: the squares are
// taken of x/inScale (an exact elementwise multiply, so they cannot overflow fp16) and the weight
// is divided by inScale on the output path, where x itself is used so that the compiler cannot
// fold the division back into the producer's weights.
func (g *Graph) RMSNorm(x *mlpackage.Var, name string, n int, eps float64, inScale float32) *mlpackage.Var {
	k := float32(1)
	if inScale > 0 {
		k = inScale
	}
	w := g.Get(name, n)
	for i := range w {
		w[i] /= k
	}
	wshape := make([]int, len(x.Shape))
	for i := range wshape {
		wshape[i] = 1
	}
	wshape[len(wshape)-1] = n
	gamma := g.F16s(name, wshape, w)
	xs := x
	if k != 1 {
		xs = g.Op("mul", map[string]any{"x": x, "y": g.B.ScalarF16(1 / k)}, x.Shape...)
	}
	sq := g.Op("mul", map[string]any{"x": xs, "y": xs}, x.Shape...)
	msShape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), 1)
	ms := g.Op("reduce_mean", map[string]any{"x": sq, "axes": g.B.Ints(-1), "keep_dims": g.B.ScalarBool(true)}, msShape...)
	r := g.Op("rsqrt", map[string]any{"x": ms, "epsilon": g.B.ScalarF16(float32(eps))}, msShape...)
	y := g.Op("mul", map[string]any{"x": x, "y": r}, x.Shape...)
	return g.Op("mul", map[string]any{"x": y, "y": gamma}, x.Shape...)
}

// SwiGLU is down(silu(gate(x)) * up(x)) with the hidden width inter and output width out; the
// down projection's weights are multiplied by scale as in LinearS.
func (g *Graph) SwiGLU(x *mlpackage.Var, gate, up, down string, inter, out int, scale float32) *mlpackage.Var {
	gv := g.Linear(x, gate, "", inter)
	u := g.Linear(x, up, "", inter)
	gv = g.Op("silu", map[string]any{"x": gv}, gv.Shape...)
	f := g.Op("mul", map[string]any{"x": gv, "y": u}, gv.Shape...)
	return g.LinearS(f, down, "", out, scale)
}

// RopeTables returns cos and sin tables of shape (1, seq, 1, head_dim) for positions 0..seq-1,
// laid out as cat(freqs, freqs) like the Hugging Face reference. invFreq has head_dim/2 entries.
// The constants are named rope_cos_<suffix> and rope_sin_<suffix>.
func (g *Graph) RopeTables(suffix string, invFreq []float64, seq, headDim int) (cos, sin *mlpackage.Var) {
	c, s := RopeValues(invFreq, 0, seq, headDim)
	shape := []int{1, seq, 1, headDim}
	return g.F16s("rope_cos_"+suffix, shape, c), g.F16s("rope_sin_"+suffix, shape, s)
}

// RopeValues returns the cos and sin tables of RopeTables for the n positions from start, for a
// program that takes them as inputs (a step with a KV cache, at a position known only at run time).
func RopeValues(invFreq []float64, start, n, headDim int) (cos, sin []float32) {
	half := headDim / 2
	c := make([]float32, n*headDim)
	s := make([]float32, n*headDim)
	for i := 0; i < n; i++ {
		pos := start + i
		for j := 0; j < half; j++ {
			a := float64(pos) * invFreq[j]
			c[i*headDim+j], c[i*headDim+j+half] = float32(math.Cos(a)), float32(math.Cos(a))
			s[i*headDim+j], s[i*headDim+j+half] = float32(math.Sin(a)), float32(math.Sin(a))
		}
	}
	return c, s
}

// ApplyRope rotates x (1, seq, heads, head_dim): x*cos + rotate_half(x)*sin, where rotate_half
// is cat(-x2, x1) over the two halves of the last axis.
func (g *Graph) ApplyRope(x, cos, sin *mlpackage.Var) *mlpackage.Var {
	sh := x.Shape
	hd := sh[3]
	half := hd / 2
	halfShape := append(append([]int{}, sh[:3]...), half)
	x1 := g.Op("slice_by_index", map[string]any{"x": x, "begin": g.B.Ints(0, 0, 0, 0), "end": g.B.Ints(0, 0, 0, int32(half)),
		"begin_mask": g.Bools(true, true, true, true), "end_mask": g.Bools(true, true, true, false)}, halfShape...)
	x2 := g.Op("slice_by_index", map[string]any{"x": x, "begin": g.B.Ints(0, 0, 0, int32(half)), "end": g.B.Ints(0, 0, 0, int32(hd)),
		"begin_mask": g.Bools(true, true, true, false), "end_mask": g.Bools(true, true, true, true)}, halfShape...)
	nx2 := g.Op("mul", map[string]any{"x": x2, "y": g.B.ScalarF16(-1)}, halfShape...)
	rot := g.Op("concat", map[string]any{"values": []*mlpackage.Var{nx2, x1}, "axis": g.B.ScalarI32(-1), "interleave": g.B.ScalarBool(false)}, sh...)
	a := g.Op("mul", map[string]any{"x": x, "y": cos}, sh...)
	b := g.Op("mul", map[string]any{"x": rot, "y": sin}, sh...)
	return g.Op("add", map[string]any{"x": a, "y": b}, sh...)
}

// CausalMask is the additive attention mask (1, 1, seq, seq): 0 where a query may see a key, a
// large negative value elsewhere. With window > 0 a query sees only the last window keys.
func (g *Graph) CausalMask(name string, seq, window int) *mlpackage.Var {
	vals := make([]float32, seq*seq)
	for q := 0; q < seq; q++ {
		for k := 0; k < seq; k++ {
			visible := k <= q
			if window > 0 {
				visible = visible && k > q-window
			}
			if !visible {
				vals[q*seq+k] = -1e4
			}
		}
	}
	return g.F16s(name, []int{1, 1, seq, seq}, vals)
}

// MaskBias turns an encoder's padding mask (1, seq), 1 for tokens and 0 for padding, into the
// additive attention bias (1, 1, 1, seq): 0 for tokens and -10000 for padding, as transformers'
// extended attention mask (fp16 safe).
func (g *Graph) MaskBias(mask *mlpackage.Var) *mlpackage.Var {
	s := mask.Shape[1]
	v := g.Op("mul", map[string]any{"x": mask, "y": g.B.ScalarF16(1e4)}, 1, s)
	v = g.Op("add", map[string]any{"x": v, "y": g.B.ScalarF16(-1e4)}, 1, s)
	return g.Op("reshape", map[string]any{"x": v, "shape": g.B.Ints(1, 1, 1, int32(s))}, 1, 1, 1, s)
}

// Attention is causal scaled dot product attention (SDPA with CausalMask(maskName, seq, window)).
func (g *Graph) Attention(q, k, v *mlpackage.Var, maskName string, window int) *mlpackage.Var {
	return g.SDPA(q, k, v, func() *mlpackage.Var { return g.CausalMask(maskName, q.Shape[2], window) })
}

// SDPA is scaled dot product attention over q (1, heads, seq, head_dim) and k, v (1, kv_heads,
// keys, head_dim); query head j reads kv head j / (heads/kv_heads) (repeat_kv). The keys are the
// queries' own positions, or with a KV cache the cached positions followed by them. mask returns
// the additive mask, broadcastable to (1, heads, seq, keys); it is called where the mask is added,
// and may be nil for none. The result is (1, heads, seq, head_dim).
func (g *Graph) SDPA(q, k, v *mlpackage.Var, mask func() *mlpackage.Var) *mlpackage.Var {
	return g.SDPAScale(q, k, v, 1/math.Sqrt(float64(q.Shape[3])), mask)
}

// SDPAScale is SDPA with the scores multiplied by scale instead of 1/sqrt(head_dim); a scale of
// 1 adds no multiply (Gemma 4, whose query and key norms take the place of the scaling).
func (g *Graph) SDPAScale(q, k, v *mlpackage.Var, scale float64, mask func() *mlpackage.Var) *mlpackage.Var {
	nh, s, hd := q.Shape[1], q.Shape[2], q.Shape[3]
	nkv, sk := k.Shape[1], k.Shape[2]
	if rep := nh / nkv; rep > 1 {
		expand := func(x *mlpackage.Var) *mlpackage.Var {
			x = g.Op("reshape", map[string]any{"x": x, "shape": g.B.Ints(1, int32(nkv), 1, int32(sk), int32(hd))}, 1, nkv, 1, sk, hd)
			x = g.Op("tile", map[string]any{"x": x, "reps": g.B.Ints(1, 1, int32(rep), 1, 1)}, 1, nkv, rep, sk, hd)
			return g.Op("reshape", map[string]any{"x": x, "shape": g.B.Ints(1, int32(nh), int32(sk), int32(hd))}, 1, nh, sk, hd)
		}
		k, v = expand(k), expand(v)
	}
	scores := g.Op("matmul", map[string]any{"x": q, "y": k, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(true)}, 1, nh, s, sk)
	if scale != 1 {
		scores = g.Op("mul", map[string]any{"x": scores, "y": g.B.ScalarF16(float32(scale))}, 1, nh, s, sk)
	}
	if mask != nil {
		scores = g.Op("add", map[string]any{"x": scores, "y": mask()}, 1, nh, s, sk)
	}
	probs := g.Op("softmax", map[string]any{"x": scores, "axis": g.B.ScalarI32(-1)}, 1, nh, s, sk)
	return g.Op("matmul", map[string]any{"x": probs, "y": v, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, nh, s, hd)
}

// SDPAGrouped is SDPAScale without copying the keys and values for each query head: the query
// heads that share a kv head are stacked along the sequence axis, (1, kv_heads, rep*seq,
// head_dim), so that one matrix product per kv head covers them all. With a long KV cache and few
// kv heads (Gemma 4: 8 query heads on 1, keys of 16384 x 512), the tile of SDPA copies the cache
// eight times per layer and step. mask is broadcast over the query heads as in SDPA.
func (g *Graph) SDPAGrouped(q, k, v *mlpackage.Var, scale float64, mask func() *mlpackage.Var) *mlpackage.Var {
	nh, s, hd := q.Shape[1], q.Shape[2], q.Shape[3]
	nkv, sk := k.Shape[1], k.Shape[2]
	rep := nh / nkv
	if rep == 1 {
		return g.SDPAScale(q, k, v, scale, mask)
	}
	// head j reads kv head j / rep: heads are grouped consecutively
	qg := g.Op("reshape", map[string]any{"x": q, "shape": g.B.Ints(1, int32(nkv), int32(rep*s), int32(hd))}, 1, nkv, rep*s, hd)
	scores := g.Op("matmul", map[string]any{"x": qg, "y": k, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(true)}, 1, nkv, rep*s, sk)
	if scale != 1 {
		scores = g.Op("mul", map[string]any{"x": scores, "y": g.B.ScalarF16(float32(scale))}, scores.Shape...)
	}
	if mask != nil {
		scores = g.Op("reshape", map[string]any{"x": scores, "shape": g.B.Ints(1, int32(nh), int32(s), int32(sk))}, 1, nh, s, sk)
		scores = g.Op("add", map[string]any{"x": scores, "y": mask()}, 1, nh, s, sk)
	}
	probs := g.Op("softmax", map[string]any{"x": scores, "axis": g.B.ScalarI32(-1)}, scores.Shape...)
	if mask != nil {
		probs = g.Op("reshape", map[string]any{"x": probs, "shape": g.B.Ints(1, int32(nkv), int32(rep*s), int32(sk))}, 1, nkv, rep*s, sk)
	}
	o := g.Op("matmul", map[string]any{"x": probs, "y": v, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, nkv, rep*s, hd)
	return g.Op("reshape", map[string]any{"x": o, "shape": g.B.Ints(1, int32(nh), int32(s), int32(hd))}, 1, nh, s, hd)
}

// ChunkedSDPA is SDPA over keys and values in parts, ks[j] and vs[j] (1, nkv, Lj, hd) with
// masks[j] (1, 1, s, Lj): a KV cache in chunks and the step's own tokens, none of them over the
// Neural Engine's dimension limit as one concatenated key axis would be. The parts meet in
// elementwise operations only: the exps of every part's scores are taken from the largest score of
// all plus ln 4n (n parts), their sums and the values weighted by them are matrix products, added
// over the parts and divided once. With the shift the sums are at most a quarter of the keys over
// n, and the weighted values that times the largest value, in fp16. Holding the parts on an axis
// of one tensor, or weighting each part by its share of the whole, put the whole program on the
// CPU, and the sums taken by reduce_sum made the placement of 64 tokens per call change from one
// compile to the next (M3, macOS 27.0.1; npuprobe kv -kv-modes host-chunk, TestChunkedSDPAOnNeuralEngine).
func (g *Graph) ChunkedSDPA(q *mlpackage.Var, ks, vs, masks []*mlpackage.Var) *mlpackage.Var {
	nh, s, hd := q.Shape[1], q.Shape[2], q.Shape[3]
	rows := []int{1, nh, s, 1}
	expand := func(x *mlpackage.Var) *mlpackage.Var {
		nkv, sk := x.Shape[1], x.Shape[2]
		rep := nh / nkv
		if rep == 1 {
			return x
		}
		x = g.Op("reshape", map[string]any{"x": x, "shape": g.B.Ints(1, int32(nkv), 1, int32(sk), int32(hd))}, 1, nkv, 1, sk, hd)
		x = g.Op("tile", map[string]any{"x": x, "reps": g.B.Ints(1, 1, int32(rep), 1, 1)}, 1, nkv, rep, sk, hd)
		return g.Op("reshape", map[string]any{"x": x, "shape": g.B.Ints(1, int32(nh), int32(sk), int32(hd))}, 1, nh, sk, hd)
	}
	scale := g.B.ScalarF16(float32(1 / math.Sqrt(float64(hd))))
	var scores []*mlpackage.Var
	var top *mlpackage.Var
	for j, k := range ks {
		sk := k.Shape[2]
		sc := g.Op("matmul", map[string]any{"x": q, "y": expand(k), "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(true)}, 1, nh, s, sk)
		sc = g.Op("mul", map[string]any{"x": sc, "y": scale}, 1, nh, s, sk)
		sc = g.Op("add", map[string]any{"x": sc, "y": masks[j]}, 1, nh, s, sk)
		scores = append(scores, sc)
		mj := g.Op("reduce_max", map[string]any{"x": sc, "axes": g.B.Ints(3), "keep_dims": g.B.ScalarBool(true)}, rows...)
		if top == nil {
			top = mj
		} else {
			top = g.Op("maximum", map[string]any{"x": top, "y": mj}, rows...)
		}
	}
	top = g.Op("add", map[string]any{"x": top, "y": g.B.ScalarF16(float32(math.Log(float64(4 * len(ks)))))}, rows...)
	var o, sum *mlpackage.Var
	for j, sc := range scores {
		sk := sc.Shape[3]
		e := g.Op("exp", map[string]any{"x": g.Op("sub", map[string]any{"x": sc, "y": top}, 1, nh, s, sk)}, 1, nh, s, sk)
		// the sum of the exps and the values weighted by them, as matrix products
		ones := g.F16s(fmt.Sprintf("ones_%d", sk), []int{sk, 1}, constOnes(sk))
		l := g.Op("matmul", map[string]any{"x": e, "y": ones, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, rows...)
		oj := g.Op("matmul", map[string]any{"x": e, "y": expand(vs[j]), "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, nh, s, hd)
		if o == nil {
			o, sum = oj, l
			continue
		}
		o = g.Op("add", map[string]any{"x": o, "y": oj}, 1, nh, s, hd)
		sum = g.Op("add", map[string]any{"x": sum, "y": l}, rows...)
	}
	return g.Op("real_div", map[string]any{"x": o, "y": sum}, 1, nh, s, hd)
}

func constOnes(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

// LinearBias is Linear plus the named bias (out), which stays fp16 whatever the weight format.
func (g *Graph) LinearBias(x *mlpackage.Var, name, bias, outName string, outDim int) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	w := g.Weight(name, outDim, in, 1)
	bv := g.F16s(bias, []int{outDim}, g.Get(bias, outDim))
	shape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), outDim)
	inputs := map[string]any{"x": x, "weight": w, "bias": bv}
	if outName != "" {
		return g.B.NamedOp("linear", outName, inputs, mlpackage.Float16, shape...)
	}
	return g.Op("linear", inputs, shape...)
}

// LayerNorm normalizes over the last axis (size n) and applies the named weight and bias.
func (g *Graph) LayerNorm(x *mlpackage.Var, weight, bias string, n int, eps float64) *mlpackage.Var {
	gamma := g.F16s(weight, []int{n}, g.Get(weight, n))
	beta := g.F16s(bias, []int{n}, g.Get(bias, n))
	return g.Op("layer_norm", map[string]any{"x": x, "axes": g.B.Ints(-1), "gamma": gamma, "beta": beta, "epsilon": g.B.ScalarF16(float32(eps))}, x.Shape...)
}

// Activation applies a Hugging Face hidden_act: gelu (exact), gelu_new / gelu_pytorch_tanh
// (tanh approximation), relu, silu.
func (g *Graph) Activation(x *mlpackage.Var, act string) *mlpackage.Var {
	switch act {
	case "gelu":
		return g.Op("gelu", map[string]any{"x": x, "mode": g.Str("EXACT")}, x.Shape...)
	case "gelu_new", "gelu_pytorch_tanh":
		return g.Op("gelu", map[string]any{"x": x, "mode": g.Str("TANH_APPROXIMATION")}, x.Shape...)
	case "relu", "silu":
		return g.Op(act, map[string]any{"x": x}, x.Shape...)
	}
	g.Fail("activation %q is not supported", act)
	return nil
}
