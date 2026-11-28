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
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/palette"
	"github.com/shibukawa/hwnpu/internal/syncx"
)

// Graph is a program under construction with its weight source and options.
type Graph struct {
	// Name prefixes error messages, normally the architecture name.
	Name string
	B    *mlpackage.Builder
	W    llm.Weights
	Opts llm.Options

	luts map[string]palette.LUT4 // the whole matrices MatrixSlice cut blocks of, by name
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

// weightKey uniquely identifies a weight tensor's quantized representation in a shared OpCache.
func (g *Graph) weightKey(kind, name, constName string, a, b, rows, cols int, scale float32) string {
	return fmt.Sprintf("%s|%s|%s|%s|%d:%d|%dx%d|%s|%d|%d|%s|%s|%s|%s|%08x",
		kind, g.W.Source(), name, constName, a, b, rows, cols,
		g.Opts.FormatOf(name), g.Opts.Group, g.Opts.ColBlock, g.Opts.Quant, g.Opts.Codes,
		g.Opts.Backend, g.Opts.Params[llm.RotateParam], math.Float32bits(scale))
}

// Weight adds a (rows, cols) matrix in the chosen format, multiplied by scale.
func (g *Graph) Weight(name string, rows, cols int, scale float32) *mlpackage.Var {
	if g.B.HasOpCache() && len(g.Opts.HeadRows) == 0 {
		key := g.weightKey("W", name, name, 0, 0, rows, cols, scale)
		v, err := g.B.CachedOp(key, name, func() (*mlpackage.Var, error) {
			return g.matrix(name, name, g.Get(name, rows, cols), rows, cols, scale), nil
		})
		if err != nil {
			g.Fail("%v", err)
		}
		return v
	}
	return g.matrix(name, name, g.Get(name, rows, cols), rows, cols, scale)
}

// Matrix adds the (rows, cols) values w as the matrix name in the chosen format: Weight for a
// matrix the model lays out itself (a transposed Flax kernel).
func (g *Graph) Matrix(name string, w []float32, rows, cols int) *mlpackage.Var {
	return g.matrix(name, name, w, rows, cols, 1)
}

// WeightRows is Weight for rows [begin, end) of the named matrix of cols columns: one part of a
// fused projection (PLaMo's qkv_proj and gate_up_proj). The constant is called name_begin_end.
func (g *Graph) WeightRows(name string, begin, end, cols int, scale float32) *mlpackage.Var {
	constName := fmt.Sprintf("%s_%d_%d", name, begin, end)
	if g.B.HasOpCache() && len(g.Opts.HeadRows) == 0 {
		key := g.weightKey("WR", name, constName, begin, end, end-begin, cols, scale)
		v, err := g.B.CachedOp(key, constName, func() (*mlpackage.Var, error) {
			w, err := g.W.Rows(name, begin, end)
			if err != nil {
				g.Fail("%v", err)
			}
			if len(w) != (end-begin)*cols {
				g.Fail("%s: rows %d..%d have %d values, expected %d columns", name, begin, end, len(w), cols)
			}
			return g.matrix(name, constName, w, end-begin, cols, scale), nil
		})
		if err != nil {
			g.Fail("%v", err)
		}
		return v
	}
	w, err := g.W.Rows(name, begin, end)
	if err != nil {
		g.Fail("%v", err)
	}
	if len(w) != (end-begin)*cols {
		g.Fail("%s: rows %d..%d have %d values, expected %d columns", name, begin, end, len(w), cols)
	}
	return g.matrix(name, constName, w, end-begin, cols, scale)
}

// rowsOnly refuses a weight of tables or scales per block of columns in a Core AI program: the
// Neural Engine compiles it (fully placed, at the speed of whole rows) but every row group takes
// its first column block's table or scale for all its columns, where Core AI's CPU is right
// (macOS 27.2, M1: docs/design/llm-qwen.md). Core ML and Intel's NPU are left as they were.
func (g *Graph) rowsOnly(name string, colBlock, cols int) {
	if g.Opts.Backend == llm.BackendCoreAI && colBlock > 0 && colBlock != cols {
		g.Fail("%s: tables or scales of %d columns (of %d): the Neural Engine reads only the first block's through Core AI; whole rows only", name, colBlock, cols)
	}
}

// matrix writes the (rows, cols) values w of the tensor name as the constant constName.
func (g *Graph) matrix(name, constName string, w []float32, rows, cols int, scale float32) *mlpackage.Var {
	format := g.Opts.FormatOf(name)
	if bits := map[string]int{"lut4": 4, "lut6": 6, "lut8": 8}[format]; bits != 0 && g.Opts.Codes != "" {
		// codes chosen for the weight as the checkpoint holds it, before the scale
		c, err := LoadCodes(g.Opts.Codes)
		if err != nil {
			g.Fail("%v", err)
		}
		q, ok, err := c.Lookup(w, rows, cols, g.Opts.Group, g.Opts.ColBlock, bits, scale)
		if err != nil {
			g.Fail("%s: %v", name, err)
		}
		if ok {
			g.rowsOnly(name, q.ColBlock, cols)
			v, err := g.B.LUT(constName, bits, []int{rows, cols}, []int{q.Group, q.ColBlock}, q.Indices, q.LUT)
			if err != nil {
				g.Fail("%v", err)
			}
			return v
		}
	}
	if format == "int8" && g.Opts.Codes != "" {
		// integers chosen for the weight as the checkpoint holds it (a scale a row)
		c, err := LoadCodes(g.Opts.Codes)
		if err != nil {
			g.Fail("%v", err)
		}
		q, ok, err := c.LookupLinear(w, rows, cols, scale)
		if err != nil {
			g.Fail("%s: %v", name, err)
		}
		if ok {
			v, err := g.B.Int8Scaled(constName, []int{rows, cols}, []int{1, cols}, q.Data, q.Scales)
			if err != nil {
				g.Fail("%v", err)
			}
			return v
		}
	}
	if scale != 1 {
		for i := range w {
			w[i] *= scale
		}
	}
	var v *mlpackage.Var
	var err error
	switch format {
	case "fp16":
		v, err = g.B.F16s(constName, []int{rows, cols}, w)
	case "lut4", "lut6", "lut8":
		bits := map[string]int{"lut4": 4, "lut6": 6, "lut8": 8}[format]
		q := palette.QuantizeBits(w, rows, cols, g.Opts.Group, g.Opts.ColBlock, bits, 8, g.lutMode())
		g.rowsOnly(name, q.ColBlock, cols)
		v, err = g.B.LUT(constName, bits, []int{rows, cols}, []int{q.Group, q.ColBlock}, q.Indices, q.LUT)
	default:
		bits, block, ok := llm.LinearFormat(format)
		if !ok {
			g.Fail("unknown format %q", format)
		}
		if block > 0 && cols%block != 0 {
			g.Fail("%s: %d columns are not a multiple of the %s blocks", name, cols, format)
		}
		g.rowsOnly(name, block, cols)
		search := true
		switch g.Opts.Quant {
		case "", "mse":
		case "uniform":
			search = false
		case "kmeans", "kmeans-minmax":
			// the tables' fitting of a lutN model (hwnpu-chat's), for a weight FormatFor makes
			// integer: the scales' search
			if !strings.HasPrefix(g.Opts.Format, "lut") {
				g.Fail("quantizer %q is not for %s; use uniform or mse", g.Opts.Quant, format)
			}
		default:
			g.Fail("quantizer %q is not for %s; use uniform or mse", g.Opts.Quant, format)
		}
		q := palette.QuantizeLinear(w, rows, cols, block, bits, search)
		if bits == 4 {
			v, err = g.B.Int4Scaled(constName, []int{rows, cols}, []int{1, q.ColBlock}, q.Data, q.Scales)
		} else {
			v, err = g.B.Int8Scaled(constName, []int{rows, cols}, []int{1, q.ColBlock}, q.Data, q.Scales)
		}
	}
	if err != nil {
		g.Fail("%v", err)
	}
	return v
}

// MatrixSlice is the block of rows [r0, r1) and columns [c0, c1) of the named (rows, cols)
// weight as a constant of its own, for a weight split to keep an axis within the Neural
// Engine's 16384 (Qwen2.5-Coder-7B's MLP of 18944). In a lutN format it is that block of the
// whole matrix's tables, fitted once (or the codes chosen for the whole matrix, Opts.Codes), so
// that the parts together are the matrix palettized whole: the rows whole groups, the columns
// whole column blocks (any with tables of whole rows). In another format, the block's values.
// The constant is called name_r0_r1_c0_c1.
func (g *Graph) MatrixSlice(name string, rows, cols, r0, r1, c0, c1 int) *mlpackage.Var {
	constName := fmt.Sprintf("%s_%d_%d_%d_%d", name, r0, r1, c0, c1)
	if g.B.HasOpCache() && len(g.Opts.HeadRows) == 0 {
		key := g.weightKey("MS", name, constName, r0*cols+c0, r1*cols+c1, rows, cols, 1)
		v, err := g.B.CachedOp(key, constName, func() (*mlpackage.Var, error) {
			return g.matrixSlice(name, constName, rows, cols, r0, r1, c0, c1), nil
		})
		if err != nil {
			g.Fail("%v", err)
		}
		return v
	}
	return g.matrixSlice(name, constName, rows, cols, r0, r1, c0, c1)
}

func (g *Graph) matrixSlice(name, constName string, rows, cols, r0, r1, c0, c1 int) *mlpackage.Var {
	format := g.Opts.FormatOf(name)
	bits := map[string]int{"lut4": 4, "lut6": 6, "lut8": 8}[format]
	if bits == 0 {
		w := g.Get(name, rows, cols)
		part := make([]float32, 0, (r1-r0)*(c1-c0))
		for r := r0; r < r1; r++ {
			part = append(part, w[r*cols+c0:r*cols+c1]...)
		}
		return g.matrix(name, constName, part, r1-r0, c1-c0, 1)
	}
	q, ok := g.luts[name]
	if !ok {
		w := g.Get(name, rows, cols)
		found := false
		if g.Opts.Codes != "" {
			c, err := LoadCodes(g.Opts.Codes)
			if err != nil {
				g.Fail("%v", err)
			}
			if q, found, err = c.Lookup(w, rows, cols, g.Opts.Group, g.Opts.ColBlock, bits, 1); err != nil {
				g.Fail("%s: %v", name, err)
			}
		}
		if !found {
			q = palette.QuantizeBits(w, rows, cols, g.Opts.Group, g.Opts.ColBlock, bits, 8, g.lutMode())
		}
		if g.luts == nil {
			g.luts = map[string]palette.LUT4{}
		}
		g.luts[name] = q
	}
	s := q.Sub(r0, r1, c0, c1)
	g.rowsOnly(name, s.ColBlock, s.Cols)
	v, err := g.B.LUT(constName, bits, []int{s.Rows, s.Cols}, []int{s.Group, s.ColBlock}, s.Indices, s.LUT)
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
	if bits := RowScaledBits(g.Opts.FormatOf(name)); bits != 0 {
		return g.linearRows(x, name, "", outName, outDim, bits, s)
	}
	w := g.Weight(name, outDim, in, s)
	shape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), outDim)
	if outName != "" {
		return g.B.NamedOp("linear", outName, map[string]any{"x": x, "weight": w}, mlpackage.Float16, shape...)
	}
	return g.Op("linear", map[string]any{"x": x, "weight": w}, shape...)
}

// LinearPadded is Linear with the weight's rows padded with zeros to a multiple of the format's
// row group (Opts.Group), which the lookup tables need, and the result cut back to outDim:
// Whisper's output layer, its 51866 rows not a multiple of 16. With rows that divide already it
// is Linear.
func (g *Graph) LinearPadded(x *mlpackage.Var, name, outName string, outDim int) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	group := max(g.Opts.Group, 1)
	rows := (outDim + group - 1) / group * group
	if rows == outDim {
		return g.Linear(x, name, outName, outDim)
	}
	w := append(g.Get(name, outDim, in), make([]float32, (rows-outDim)*in)...)
	y := g.Op("linear", map[string]any{"x": x, "weight": g.matrix(name, name, w, rows, in, 1)}, append(append([]int{}, x.Shape[:len(x.Shape)-1]...), rows)...)
	shape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), outDim)
	n := len(shape)
	begin, end := make([]int32, n), make([]int32, n)
	bm, em := make([]bool, n), make([]bool, n)
	for i := range shape {
		end[i] = int32(shape[i])
		bm[i], em[i] = true, true
	}
	bm[n-1], em[n-1] = false, false
	inputs := map[string]any{"x": y, "begin": g.B.Ints(begin...), "end": g.B.Ints(end...), "begin_mask": g.Bools(bm...), "end_mask": g.Bools(em...)}
	if outName != "" {
		return g.B.NamedOp("slice_by_index", outName, inputs, mlpackage.Float16, shape...)
	}
	return g.Op("slice_by_index", inputs, shape...)
}

// LastRow is the row of x (1, seq, d) that sel (1, 1, seq), one-hot, picks: (1, 1, d). A matrix
// product, since the Neural Engine runs gather on the CPU; with one 1 it is the row exactly.
func (g *Graph) LastRow(x, sel *mlpackage.Var) *mlpackage.Var {
	return g.Op("matmul", map[string]any{"x": sel, "y": x, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, 1, x.Shape[2])
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
	return g.RMSNormWith(x, name, g.Get(name, n), eps, inScale)
}

// RMSNormWith is RMSNorm with the weight's values given, n = len(w): for a norm that scales by a
// function of the stored weight, as the zero centered norms of Qwen3.5 and Gemma 3 (1 + w). The
// constant is still called name.
func (g *Graph) RMSNormWith(x *mlpackage.Var, name string, w []float32, eps float64, inScale float32) *mlpackage.Var {
	n := len(w)
	k := float32(1)
	if inScale > 0 {
		k = inScale
	}
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

// MaxAbsRMSNorm normalizes over the last axis with scale-invariant max-abs prescaling
// (floor="add": scale = max(|x|) + 2^-14) so tiny activations (down to 1e-3) do not underflow
// x^2 below fp16's normal range (2^-14) and zero rows (e.g. padding) never trigger ANE rsqrt
// +Inf/NaN, while avoiding NEFUSED_EW max-clamp issues.
func (g *Graph) MaxAbsRMSNorm(x *mlpackage.Var, name string, w []float32) *mlpackage.Var {
	const fp16MinNormal = float32(1.0 / 16384.0) // 2^-14 = 0x0400
	wshape := make([]int, len(x.Shape))
	for i := range wshape {
		wshape[i] = 1
	}
	wshape[len(wshape)-1] = len(w)
	gamma := g.F16s(name, wshape, w)
	msShape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), 1)
	absX := g.Op("abs", map[string]any{"x": x}, x.Shape...)
	maxAbs := g.Op("reduce_max", map[string]any{"x": absX, "axes": g.B.Ints(-1), "keep_dims": g.B.ScalarBool(true)}, msShape...)
	scale := g.Op("add", map[string]any{"x": maxAbs, "y": g.B.ScalarF16(fp16MinNormal)}, msShape...)
	xs := g.Op("real_div", map[string]any{"x": x, "y": scale}, x.Shape...)
	sq := g.Op("mul", map[string]any{"x": xs, "y": xs}, x.Shape...)
	ms := g.Op("reduce_mean", map[string]any{"x": sq, "axes": g.B.Ints(-1), "keep_dims": g.B.ScalarBool(true)}, msShape...)
	r := g.Op("rsqrt", map[string]any{"x": ms, "epsilon": g.B.ScalarF16(fp16MinNormal)}, msShape...)
	y := g.Op("mul", map[string]any{"x": xs, "y": r}, x.Shape...)
	return g.Op("mul", map[string]any{"x": y, "y": gamma}, x.Shape...)
}

// SwiGLU is down(silu(gate(x)) * up(x)) with the hidden width inter and output width out; the
// down projection's weights are multiplied by scale as in LinearS.
func (g *Graph) SwiGLU(x *mlpackage.Var, gate, up, down string, inter, out int, scale float32) *mlpackage.Var {
	gv := g.Linear(x, gate, "", inter)
	u := g.Linear(x, up, "", inter)
	gv = g.SiLU(gv)
	f := g.Op("mul", map[string]any{"x": gv, "y": u}, gv.Shape...)
	return g.LinearS(f, down, "", out, scale)
}

// SiLU is silu(x) = x sigmoid(x), written as x / (1 + exp(-x)). The Neural Engine's own silu (and
// x times its sigmoid, which gives the same values) is about 1.4% off on an M3, more on the small
// values of its tail (docs/design/llm-qwen.md): the logits' KL from PyTorch's fp32 on the Neural Engine
// fell 2.5 times with this form for LFM3-1.7B in lut8 (0.058 to 0.022), to that of the same program
// on Core ML's CPU, and by a third for PLaMo 3 in lut8 (docs/design/apple-neural-engine.md). It costs 3 operations
// more: a prompt is read about 9% slower, a step about 3%. Past x = -11, exp(-x) overflows fp16 to
// inf and the quotient is 0 (silu is under 2e-4 there).
func (g *Graph) SiLU(x *mlpackage.Var) *mlpackage.Var {
	neg := g.Op("mul", map[string]any{"x": x, "y": g.B.ScalarF16(-1)}, x.Shape...)
	den := g.Op("add", map[string]any{"x": g.Op("exp", map[string]any{"x": neg}, x.Shape...), "y": g.B.ScalarF16(1)}, x.Shape...)
	return g.Op("real_div", map[string]any{"x": x, "y": den}, x.Shape...)
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
// compile to the next (M3, macOS 27.0.1; hwnpu-dev probe kv -kv-modes host-chunk, TestChunkedSDPAOnNeuralEngine).
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

// ChunkedSDPAGrouped is ChunkedSDPA without copying the keys and values for each query head,
// as SDPAGrouped: the query heads of a kv head are stacked along the sequence axis for the
// matrix products, and the scores are laid out per query head for the masks and the maximum.
func (g *Graph) ChunkedSDPAGrouped(q *mlpackage.Var, ks, vs, masks []*mlpackage.Var) *mlpackage.Var {
	nh, s, hd := q.Shape[1], q.Shape[2], q.Shape[3]
	nkv := ks[0].Shape[1]
	rep := nh / nkv
	if rep == 1 {
		return g.ChunkedSDPA(q, ks, vs, masks)
	}
	rows := []int{1, nh, s, 1}
	qg := g.Op("reshape", map[string]any{"x": q, "shape": g.B.Ints(1, int32(nkv), int32(rep*s), int32(hd))}, 1, nkv, rep*s, hd)
	scale := g.B.ScalarF16(float32(1 / math.Sqrt(float64(hd))))
	var scores []*mlpackage.Var
	var top *mlpackage.Var
	for j, k := range ks {
		sk := k.Shape[2]
		sc := g.Op("matmul", map[string]any{"x": qg, "y": k, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(true)}, 1, nkv, rep*s, sk)
		sc = g.Op("reshape", map[string]any{"x": sc, "shape": g.B.Ints(1, int32(nh), int32(s), int32(sk))}, 1, nh, s, sk)
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
		e = g.Op("reshape", map[string]any{"x": e, "shape": g.B.Ints(1, int32(nkv), int32(rep*s), int32(sk))}, 1, nkv, rep*s, sk)
		ones := g.F16s(fmt.Sprintf("ones_%d", sk), []int{sk, 1}, constOnes(sk))
		l := g.Op("matmul", map[string]any{"x": e, "y": ones, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, nkv, rep*s, 1)
		oj := g.Op("matmul", map[string]any{"x": e, "y": vs[j], "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, nkv, rep*s, hd)
		if o == nil {
			o, sum = oj, l
			continue
		}
		o = g.Op("add", map[string]any{"x": o, "y": oj}, 1, nkv, rep*s, hd)
		sum = g.Op("add", map[string]any{"x": sum, "y": l}, 1, nkv, rep*s, 1)
	}
	o = g.Op("real_div", map[string]any{"x": o, "y": sum}, 1, nkv, rep*s, hd)
	return g.Op("reshape", map[string]any{"x": o, "shape": g.B.Ints(1, int32(nh), int32(s), int32(hd))}, 1, nh, s, hd)
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
	if bits := RowScaledBits(g.Opts.FormatOf(name)); bits != 0 {
		return g.linearRows(x, name, bias, outName, outDim, bits, 1)
	}
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

// GateWidth is the width WideAct repeats a narrow tensor to: twice the 32 values a row that
// Apple's guide for the Neural Engine asks of a tensor's last axis.
const GateWidth = 64

// WideAct is act over x (1, rows, n), a gate of a few values a row (one a head): for n under
// GateWidth the values are repeated along the last axis to GateWidth or more, act runs over
// that, and the first n are kept.
//
// Why: coreai-build's compiler for the M1 (h13g) keeps one kernel for all the activations of a
// program that have the same constants (the exp and the inverse steps of every sigmoid and silu)
// and gives it the size the last of them needs. An activation over a tensor laid out with its
// rows as channels has a set of coefficients a channel (8 of 128 bytes for 8 rows), one over a
// tensor laid out as a Linear's output a single set. A narrow gate was laid out the first way
// and the MLP's silu after it the second, so the gate's activation read 1024 bytes of a kernel
// of 128: the M1 finished such a prebuilt program and never ended its first call (Qwen3.5's
// Gated DeltaNet and LFM3's attention gate, docs/known-issues.md 16, 2026-11-27). Repeated to 64
// values a row the gate is laid out as the Linear's output; a reshape to (rows, 1, 1, n) alone
// left a gate of 16 values with its 8 sets. tools/coreai/llirtasks -check reads a prebuilt
// program for such tasks, and hwnpu-dev prebuild refuses a program that has one.
func (g *Graph) WideAct(x *mlpackage.Var, act func(*mlpackage.Var) *mlpackage.Var) *mlpackage.Var {
	if len(x.Shape) != 3 {
		g.Fail("WideAct over a tensor of rank %d", len(x.Shape))
	}
	n := x.Shape[2]
	if n >= GateWidth {
		return act(x)
	}
	vals := make([]*mlpackage.Var, (GateWidth+n-1)/n)
	for i := range vals {
		vals[i] = x
	}
	wide := g.Op("concat", map[string]any{"values": vals, "axis": g.B.ScalarI32(2), "interleave": g.B.ScalarBool(false)}, x.Shape[0], x.Shape[1], len(vals)*n)
	y := act(wide)
	return g.Op("slice_by_index", map[string]any{"x": y, "begin": g.B.Ints(0, 0, 0), "end": g.B.Ints(int32(x.Shape[0]), int32(x.Shape[1]), int32(n)),
		"begin_mask": g.Bools(true, true, false), "end_mask": g.Bools(true, true, false)}, x.Shape...)
}

// Sigmoid is the Neural Engine's own sigmoid of x.
func (g *Graph) Sigmoid(x *mlpackage.Var) *mlpackage.Var {
	return g.Op("sigmoid", map[string]any{"x": x}, x.Shape...)
}

// Activation applies a Hugging Face hidden_act: gelu (exact), gelu_new / gelu_pytorch_tanh
// (tanh approximation), relu, silu (SiLU). The Neural Engine's own gelu is kept: written with exp
// and real_div, Gemma 4 E2B's KL was the same (docs/design/apple-neural-engine.md).
func (g *Graph) Activation(x *mlpackage.Var, act string) *mlpackage.Var {
	switch act {
	case "gelu":
		return g.Op("gelu", map[string]any{"x": x, "mode": g.Str("EXACT")}, x.Shape...)
	case "gelu_new", "gelu_pytorch_tanh":
		return g.Op("gelu", map[string]any{"x": x, "mode": g.Str("TANH_APPROXIMATION")}, x.Shape...)
	case "silu":
		return g.SiLU(x)
	case "relu":
		return g.Op(act, map[string]any{"x": x}, x.Shape...)
	}
	g.Fail("activation %q is not supported", act)
	return nil
}

var codes syncx.SyncMap[string, *palette.Codes] // by directory

// LoadCodes reads a codes directory (llm.Options.Codes) once per process.
func LoadCodes(dir string) (*palette.Codes, error) {
	if c, ok := codes.Load(dir); ok {
		return c, nil
	}
	c, err := palette.LoadCodes(dir)
	if err != nil {
		return nil, err
	}
	got, _ := codes.LoadOrStore(dir, c)
	return got, nil
}

// RowScaledBits is the table width of a lutN format with a scale a row (lut4r, lut6r, lut8r), 0
// for another format.
func RowScaledBits(format string) int {
	return map[string]int{"lut4r": 4, "lut6r": 6, "lut8r": 8}[format]
}

// linearRows is LinearBias for a weight in lutNr: each row divided by its scale (its largest
// magnitude over the matrix's, in fp16) is palettized in tables of Opts.Group rows, and the
// output multiplied back by the scales before the bias (none for Linear; the scales times s for
// LinearS). Rows of very different magnitudes in one table leave the small ones few levels
// (gpt-oss's attention: rows of one projection differ by up to 6600 times); with the scales each
// row spans its table. The output is bounded as the weight's own (the scales are at most 1).
func (g *Graph) linearRows(x *mlpackage.Var, name, bias, outName string, outDim, bits int, s float32) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	wKey := g.weightKey("LR", name, name, bits, 0, outDim, in, 1)
	sKey := g.weightKey("LRS", name, name+"_rows", bits, 0, outDim, in, s)
	useCache := g.B.HasOpCache() && len(g.Opts.HeadRows) == 0
	var wv *mlpackage.Var
	var scales []float32
	if useCache && g.B.HasCachedOp(wKey) && g.B.HasCachedOp(sKey) {
		var err error
		if wv, err = g.B.CachedOp(wKey, name, nil); err != nil {
			g.Fail("%v", err)
		}
	} else {
		var q palette.LUT4
		found := false
		if g.Opts.Codes != "" {
			// codes chosen for the weight as the checkpoint holds it (gptoss_attn_gptq.py's lutrow)
			c, err := LoadCodes(g.Opts.Codes)
			if err != nil {
				g.Fail("%v", err)
			}
			if q, scales, found, err = c.LookupRows(g.Get(name, outDim, in), outDim, in, g.Opts.Group, g.Opts.ColBlock, bits); err != nil {
				g.Fail("%s: %v", name, err)
			}
		}
		if !found {
			w := slices.Clone(g.Get(name, outDim, in))
			scales = RowScales(w, outDim, in)
			q = palette.QuantizeBits(w, outDim, in, g.Opts.Group, g.Opts.ColBlock, bits, 8, g.lutMode())
		}
		if s != 1 {
			for i := range scales {
				scales[i] *= s
			}
		}
		g.rowsOnly(name, q.ColBlock, in)
		var err error
		if useCache {
			wv, err = g.B.CachedOp(wKey, name, func() (*mlpackage.Var, error) {
				return g.B.LUT(name, bits, []int{outDim, in}, []int{q.Group, q.ColBlock}, q.Indices, q.LUT)
			})
		} else {
			wv, err = g.B.LUT(name, bits, []int{outDim, in}, []int{q.Group, q.ColBlock}, q.Indices, q.LUT)
		}
		if err != nil {
			g.Fail("%v", err)
		}
	}
	shape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), outDim)
	y := g.Op("linear", map[string]any{"x": x, "weight": wv}, shape...)
	var rs *mlpackage.Var
	if useCache {
		var err error
		if rs, err = g.B.CachedOp(sKey, name+"_rows", func() (*mlpackage.Var, error) {
			return g.B.F16s(name+"_rows", []int{outDim}, scales)
		}); err != nil {
			g.Fail("%v", err)
		}
	} else {
		rs = g.F16s(name+"_rows", []int{outDim}, scales)
	}
	if bias == "" {
		if outName != "" {
			return g.B.NamedOp("mul", outName, map[string]any{"x": y, "y": rs}, mlpackage.Float16, shape...)
		}
		return g.Op("mul", map[string]any{"x": y, "y": rs}, shape...)
	}
	y = g.Op("mul", map[string]any{"x": y, "y": rs}, shape...)
	inputs := map[string]any{"x": y, "y": g.F16s(bias, []int{outDim}, g.Get(bias, outDim))}
	if outName != "" {
		return g.B.NamedOp("add", outName, inputs, mlpackage.Float16, shape...)
	}
	return g.Op("add", inputs, shape...)
}

// RowScales divides each row of w (rows x cols) by its scale, its largest magnitude over the
// matrix's rounded to fp16 (1 for a row of zeros), and returns the scales.
func RowScales(w []float32, rows, cols int) []float32 {
	top := make([]float32, rows)
	var m float32
	for r := range rows {
		for _, v := range w[r*cols : (r+1)*cols] {
			top[r] = max(top[r], float32(math.Abs(float64(v))))
		}
		m = max(m, top[r])
	}
	scales := make([]float32, rows)
	for r := range rows {
		s := float32(1)
		if top[r] > 0 && m > 0 {
			s = mlpackage.Float16Value(mlpackage.Float16Bits(top[r] / m))
		}
		scales[r] = s
		for c := range cols {
			w[r*cols+c] /= s
		}
	}
	return scales
}

// lutMode is Opts.Quant's table fitting for a lutN weight.
func (g *Graph) lutMode() palette.Mode {
	switch g.Opts.Quant {
	case "", "uniform":
		return palette.Uniform
	case "kmeans":
		return palette.KMeans
	case "kmeans-minmax":
		return palette.KMeansMinMax
	}
	g.Fail("unknown quantizer %q", g.Opts.Quant)
	return 0
}
