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
	"strings"

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
	w := g.Get(name, rows, cols)
	if scale != 1 {
		for i := range w {
			w[i] *= scale
		}
	}
	var v *mlpackage.Var
	var err error
	format := g.Opts.Format
	for _, keep := range g.Opts.KeepFP16 {
		if strings.Contains(name, keep) {
			format = "fp16"
		}
	}
	switch format {
	case "fp16":
		v, err = g.B.F16s(name, []int{rows, cols}, w)
	case "lut4":
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
		q := palette.Quantize(w, rows, cols, g.Opts.Group, g.Opts.ColBlock, 8, mode)
		v, err = g.B.LUT4(name, []int{rows, cols}, []int{q.Group, q.ColBlock}, q.Indices, q.LUT)
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
	half := headDim / 2
	c := make([]float32, seq*headDim)
	s := make([]float32, seq*headDim)
	for pos := 0; pos < seq; pos++ {
		for j := 0; j < half; j++ {
			a := float64(pos) * invFreq[j]
			c[pos*headDim+j], c[pos*headDim+j+half] = float32(math.Cos(a)), float32(math.Cos(a))
			s[pos*headDim+j], s[pos*headDim+j+half] = float32(math.Sin(a)), float32(math.Sin(a))
		}
	}
	shape := []int{1, seq, 1, headDim}
	return g.F16s("rope_cos_"+suffix, shape, c), g.F16s("rope_sin_"+suffix, shape, s)
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

// Attention is causal scaled dot product attention over q (1, heads, seq, head_dim) and k, v
// (1, kv_heads, seq, head_dim); query head j reads kv head j / (heads/kv_heads) (repeat_kv). The
// mask is CausalMask(maskName, seq, window). The result is (1, heads, seq, head_dim).
func (g *Graph) Attention(q, k, v *mlpackage.Var, maskName string, window int) *mlpackage.Var {
	nh, s, hd := q.Shape[1], q.Shape[2], q.Shape[3]
	nkv := k.Shape[1]
	if rep := nh / nkv; rep > 1 {
		expand := func(x *mlpackage.Var) *mlpackage.Var {
			x = g.Op("reshape", map[string]any{"x": x, "shape": g.B.Ints(1, int32(nkv), 1, int32(s), int32(hd))}, 1, nkv, 1, s, hd)
			x = g.Op("tile", map[string]any{"x": x, "reps": g.B.Ints(1, 1, int32(rep), 1, 1)}, 1, nkv, rep, s, hd)
			return g.Op("reshape", map[string]any{"x": x, "shape": g.B.Ints(1, int32(nh), int32(s), int32(hd))}, 1, nh, s, hd)
		}
		k, v = expand(k), expand(v)
	}
	scores := g.Op("matmul", map[string]any{"x": q, "y": k, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(true)}, 1, nh, s, s)
	scores = g.Op("mul", map[string]any{"x": scores, "y": g.B.ScalarF16(float32(1 / math.Sqrt(float64(hd))))}, 1, nh, s, s)
	scores = g.Op("add", map[string]any{"x": scores, "y": g.CausalMask(maskName, s, window)}, 1, nh, s, s)
	probs := g.Op("softmax", map[string]any{"x": scores, "axis": g.B.ScalarI32(-1)}, 1, nh, s, s)
	return g.Op("matmul", map[string]any{"x": probs, "y": v, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, nh, s, hd)
}
