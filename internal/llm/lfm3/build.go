package lfm3

import (
	"fmt"
	"math"
	"strings"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/palette"
)

// Options select what to build.
type Options struct {
	// Seq is the fixed sequence length of the program.
	Seq int
	// Layers truncates the model to its first Layers layers (0 means all); with a truncated
	// model the output is the hidden state instead of the logits, for checking layer by layer.
	Layers int
	// Start skips the first Start layers: the input is then the hidden state entering layer
	// Start, and Layers counts from there. Only for checking single layers.
	Start int
	// Hidden makes the output the final hidden state (after the last norm) rather than logits.
	Hidden bool
	// Format is "lut4" (4 bit palettized, k-means per 16 rows) or "fp16".
	Format string
	// Group is the number of rows sharing a lookup table (16 is what the Neural Engine reads fast).
	Group int
	// ColBlock splits each table's rows into blocks of this many columns (0: whole rows).
	ColBlock int
	// KeepFP16 lists substrings of tensor names to keep in fp16 regardless of Format, such as
	// "embed_tokens" for the tied embedding and output layer.
	KeepFP16 []string
	// Quant selects the table fitting for lut4: "uniform" (default), "kmeans" or "kmeans-minmax".
	Quant string
	// PostScale multiplies the weights of the projections that feed the sandwich norms (conv
	// out_proj, attention o_proj, MLP w2) by this power of two, and the following norm's epsilon
	// by its square, which leaves the model unchanged. The Neural Engine's matrix products have
	// an absolute error floor of about 2.5e-4, and these projections' outputs are around 0.01
	// in LFM3, so without it they lose several percent. 0 or 1 disables; 64 is a good value.
	PostScale float32
	// ConvMode selects how the short convolution runs: "taps" (default) as 7 shifted multiply
	// adds, which the Neural Engine computes at full fp16 accuracy, or "conv" as one depthwise
	// conv operation, which it computes with 5 to 12 percent error per layer.
	ConvMode string
}

type builder struct {
	cfg  *Config
	w    Weights
	opts Options
	b    *mlpackage.Builder
	seq  int
	d    int
}

// Build writes the program for the model into b. The program's input is x, the fp16 token
// embeddings of shape (1, seq, hidden): the embedding lookup stays on the host because Core ML
// runs gather on the CPU. The output is y: logits (1, seq, vocab) or a hidden state (1, seq,
// hidden). It returns the input and output descriptions.
func Build(b *mlpackage.Builder, cfg *Config, w Weights, opts Options) (in, out mlpackage.Feature, err error) {
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = "lut4"
	}
	if opts.Layers <= 0 || opts.Start+opts.Layers > cfg.NumLayers {
		opts.Layers = cfg.NumLayers - opts.Start
	}
	if opts.ConvMode == "" {
		opts.ConvMode = "taps"
	}
	if opts.PostScale <= 0 {
		opts.PostScale = 1
	}
	if opts.PostScale != 1 && !cfg.UseSandwichNorm {
		return in, out, fmt.Errorf("lfm3: PostScale needs sandwich norms to absorb the scale")
	}
	m := &builder{cfg: cfg, w: w, opts: opts, b: b, seq: opts.Seq, d: cfg.HiddenSize}
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(buildError); ok {
				err = e.err
				return
			}
			panic(r)
		}
	}()
	x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
	in = mlpackage.Feature{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}
	h := x
	for i := opts.Start; i < opts.Start+opts.Layers; i++ {
		h = m.layer(i, h)
	}
	var y *mlpackage.Var
	switch {
	case opts.Start+opts.Layers < cfg.NumLayers:
		y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
	default:
		h = m.rmsNorm(h, "model.embedding_norm.weight", m.d)
		if opts.Hidden {
			y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		} else {
			head := "lm_head.weight"
			if cfg.TieWordEmbed {
				head = "model.embed_tokens.weight"
			}
			y = m.linear(h, head, "y", cfg.VocabSize)
		}
	}
	out = mlpackage.Feature{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}
	return in, out, nil
}

type buildError struct{ err error }

func (m *builder) fail(format string, args ...any) {
	panic(buildError{fmt.Errorf("lfm3: "+format, args...)})
}

func (m *builder) get(name string, want ...int) []float32 {
	v, shape, err := m.w.Get(name)
	if err != nil {
		m.fail("%v", err)
	}
	if len(want) > 0 {
		ok := len(shape) == len(want)
		for i := range want {
			ok = ok && shape[i] == want[i]
		}
		if !ok {
			m.fail("%s: shape %v, expected %v", name, shape, want)
		}
	}
	return v
}

func (m *builder) op(typ string, inputs map[string]any, shape ...int) *mlpackage.Var {
	return m.b.Op(typ, inputs, mlpackage.Float16, shape...)
}

// weight adds a (rows, cols) matrix in the chosen format, multiplied by scale.
func (m *builder) weight(name string, rows, cols int, scale float32) *mlpackage.Var {
	w := m.get(name, rows, cols)
	if scale != 1 {
		for i := range w {
			w[i] *= scale
		}
	}
	var v *mlpackage.Var
	var err error
	format := m.opts.Format
	for _, keep := range m.opts.KeepFP16 {
		if strings.Contains(name, keep) {
			format = "fp16"
		}
	}
	switch format {
	case "fp16":
		v, err = m.b.F16s(name, []int{rows, cols}, w)
	case "lut4":
		mode := palette.Uniform
		switch m.opts.Quant {
		case "", "uniform":
		case "kmeans":
			mode = palette.KMeans
		case "kmeans-minmax":
			mode = palette.KMeansMinMax
		default:
			m.fail("unknown quantizer %q", m.opts.Quant)
		}
		q := palette.Quantize(w, rows, cols, m.opts.Group, m.opts.ColBlock, 8, mode)
		v, err = m.b.LUT4(name, []int{rows, cols}, []int{q.Group, q.ColBlock}, q.Indices, q.LUT)
	default:
		m.fail("unknown format %q", m.opts.Format)
	}
	if err != nil {
		m.fail("%v", err)
	}
	return v
}

// linear is x (1, seq, in) times the weight (out, in) transposed.
func (m *builder) linear(x *mlpackage.Var, name, outName string, outDim int) *mlpackage.Var {
	return m.linearS(x, name, outName, outDim, 1)
}

// linearS is linear with the weights multiplied by s, so that the output is s times larger;
// the consumer must undo it (a norm with its epsilon scaled by s*s).
func (m *builder) linearS(x *mlpackage.Var, name, outName string, outDim int, s float32) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	w := m.weight(name, outDim, in, s)
	shape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), outDim)
	if outName != "" {
		return m.b.NamedOp("linear", outName, map[string]any{"x": x, "weight": w}, mlpackage.Float16, shape...)
	}
	return m.op("linear", map[string]any{"x": x, "weight": w}, shape...)
}

// rmsNorm normalizes over the last axis (size n) and scales by the named weight. inScale says
// by how much the input has been scaled up by its producer (PostScale): the squares are taken
// of x/inScale (an exact elementwise multiply, so they cannot overflow fp16) and the weight is
// divided by inScale on the output path, where x itself is used so that the compiler cannot
// fold the division back into the producer's weights.
func (m *builder) rmsNorm(x *mlpackage.Var, name string, n int, inScale ...float32) *mlpackage.Var {
	k := float32(1)
	if len(inScale) > 0 && inScale[0] > 0 {
		k = inScale[0]
	}
	g := m.get(name, n)
	for i := range g {
		g[i] /= k
	}
	wshape := make([]int, len(x.Shape))
	for i := range wshape {
		wshape[i] = 1
	}
	wshape[len(wshape)-1] = n
	gamma, err := m.b.F16s(name, wshape, g)
	if err != nil {
		m.fail("%v", err)
	}
	xs := x
	if k != 1 {
		xs = m.op("mul", map[string]any{"x": x, "y": m.b.ScalarF16(1 / k)}, x.Shape...)
	}
	sq := m.op("mul", map[string]any{"x": xs, "y": xs}, x.Shape...)
	msShape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), 1)
	ms := m.op("reduce_mean", map[string]any{"x": sq, "axes": m.b.Ints(-1), "keep_dims": m.b.ScalarBool(true)}, msShape...)
	r := m.op("rsqrt", map[string]any{"x": ms, "epsilon": m.b.ScalarF16(float32(m.cfg.NormEps))}, msShape...)
	y := m.op("mul", map[string]any{"x": x, "y": r}, x.Shape...)
	return m.op("mul", map[string]any{"x": y, "y": gamma}, x.Shape...)
}

func (m *builder) layer(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("model.layers.%d.", i)
	residual := h
	h = m.rmsNorm(h, p+"operator_norm.weight", m.d)
	if m.cfg.IsAttention(i) {
		h = m.attention(i, h)
	} else {
		h = m.shortConv(i, h)
	}
	if m.cfg.UseSandwichNorm {
		h = m.rmsNorm(h, p+"post_operator_norm.weight", m.d, m.opts.PostScale)
	}
	h = m.op("add", map[string]any{"x": h, "y": residual}, h.Shape...)
	residual = h
	f := m.rmsNorm(h, p+"ffn_norm.weight", m.d)
	g := m.linear(f, p+"feed_forward.w1.weight", "", m.cfg.IntermediateSize)
	u := m.linear(f, p+"feed_forward.w3.weight", "", m.cfg.IntermediateSize)
	g = m.op("silu", map[string]any{"x": g}, g.Shape...)
	f = m.op("mul", map[string]any{"x": g, "y": u}, g.Shape...)
	f = m.linearS(f, p+"feed_forward.w2.weight", "", m.d, m.opts.PostScale)
	if m.cfg.UseSandwichNorm {
		f = m.rmsNorm(f, p+"post_ffn_norm.weight", m.d, m.opts.PostScale)
	}
	return m.op("add", map[string]any{"x": f, "y": residual}, f.Shape...)
}

// shortConv is the gated short convolution: in_proj splits into B, C and x; the depthwise causal
// convolution of width K runs over B*x along the sequence, and C gates the result. In "taps"
// mode it stays in the (1, seq, d) layout: the convolution is K shifted multiply adds over a
// left padded sequence, with no transposes.
func (m *builder) shortConv(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("model.layers.%d.conv.", i)
	d, s, k := m.d, m.seq, m.cfg.ConvKernel
	bcx := m.linear(h, p+"in_proj.weight", "", 3*d) // (1, s, 3d)
	cw := m.get(p+"conv.weight", d, 1, k)
	var y, cg *mlpackage.Var
	switch m.opts.ConvMode {
	case "conv":
		t := m.op("transpose", map[string]any{"x": bcx, "perm": m.b.Ints(0, 2, 1)}, 1, 3*d, s) // (1, 3d, s)
		part := func(j int) *mlpackage.Var {
			return m.op("slice_by_index", map[string]any{"x": t, "begin": m.b.Ints(0, int32(j*d), 0), "end": m.b.Ints(1, int32((j+1)*d), int32(s)),
				"begin_mask": m.bools(true, false, true), "end_mask": m.bools(true, false, true)}, 1, d, s)
		}
		bg, xg := part(0), part(2)
		cg = part(1)
		bx := m.op("mul", map[string]any{"x": bg, "y": xg}, 1, d, s)
		padded := m.op("pad", map[string]any{"x": bx, "pad": m.b.Ints(int32(k-1), 0), "mode": m.str("constant"), "constant_val": m.b.ScalarF16(0)}, 1, d, s+k-1)
		wv, err := m.b.F16s(p+"conv.weight", []int{d, 1, k}, cw)
		if err != nil {
			m.fail("%v", err)
		}
		conv := map[string]any{"x": padded, "weight": wv, "strides": m.b.Ints(1), "pad_type": m.str("valid"), "pad": m.b.Ints(0, 0),
			"dilations": m.b.Ints(1), "groups": m.b.ScalarI32(int32(d))}
		if m.cfg.ConvBias {
			bias, err := m.b.F16s(p+"conv.bias", []int{d}, m.get(p+"conv.bias", d))
			if err != nil {
				m.fail("%v", err)
			}
			conv["bias"] = bias
		}
		y = m.op("conv", conv, 1, d, s)
		y = m.op("mul", map[string]any{"x": cg, "y": y}, 1, d, s)
		y = m.op("transpose", map[string]any{"x": y, "perm": m.b.Ints(0, 2, 1)}, 1, s, d)
	case "taps":
		part := func(j int) *mlpackage.Var {
			return m.op("slice_by_index", map[string]any{"x": bcx, "begin": m.b.Ints(0, 0, int32(j*d)), "end": m.b.Ints(1, int32(s), int32((j+1)*d)),
				"begin_mask": m.bools(true, true, false), "end_mask": m.bools(true, true, false)}, 1, s, d)
		}
		bg, xg := part(0), part(2)
		cg = part(1)
		bx := m.op("mul", map[string]any{"x": bg, "y": xg}, 1, s, d)
		// pad the sequence axis on the left; pad covers the last two axes
		padded := m.op("pad", map[string]any{"x": bx, "pad": m.b.Ints(int32(k-1), 0, 0, 0), "mode": m.str("constant"), "constant_val": m.b.ScalarF16(0)}, 1, s+k-1, d)
		// y[t] = sum_j w[j] * bx[t + j - (k-1)]: tap j reads the padded sequence shifted by j
		for j := 0; j < k; j++ {
			tap := make([]float32, d)
			for c := 0; c < d; c++ {
				tap[c] = cw[c*k+j]
			}
			wv, err := m.b.F16s(fmt.Sprintf("%sconv.weight.tap%d", p, j), []int{1, 1, d}, tap)
			if err != nil {
				m.fail("%v", err)
			}
			seg := m.op("slice_by_index", map[string]any{"x": padded, "begin": m.b.Ints(0, int32(j), 0), "end": m.b.Ints(1, int32(j+s), int32(d)),
				"begin_mask": m.bools(true, false, true), "end_mask": m.bools(true, false, true)}, 1, s, d)
			term := m.op("mul", map[string]any{"x": seg, "y": wv}, 1, s, d)
			if y == nil {
				y = term
			} else {
				y = m.op("add", map[string]any{"x": y, "y": term}, 1, s, d)
			}
		}
		if m.cfg.ConvBias {
			bias, err := m.b.F16s(p+"conv.bias", []int{1, 1, d}, m.get(p+"conv.bias", d))
			if err != nil {
				m.fail("%v", err)
			}
			y = m.op("add", map[string]any{"x": y, "y": bias}, 1, s, d)
		}
		y = m.op("mul", map[string]any{"x": cg, "y": y}, 1, s, d)
	default:
		m.fail("unknown conv mode %q", m.opts.ConvMode)
	}
	return m.linearS(y, p+"out_proj.weight", "", d, m.opts.PostScale)
}

// attention is grouped query attention with QK RMS norm, partial RoPE, a causal (and for sliding
// layers windowed) mask and a sigmoid gate per head.
func (m *builder) attention(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("model.layers.%d.self_attn.", i)
	s, nh, nkv, hd := m.seq, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim
	layerType := m.cfg.LayerTypes[i]
	cos, sin := m.rope(layerType)

	q := m.linear(h, p+"q_proj.weight", "", nh*hd)
	q = m.op("reshape", map[string]any{"x": q, "shape": m.b.Ints(1, int32(s), int32(nh), int32(hd))}, 1, s, nh, hd)
	q = m.rmsNorm(q, p+"q_norm.weight", hd)
	q = m.applyRope(q, cos, sin)
	q = m.op("transpose", map[string]any{"x": q, "perm": m.b.Ints(0, 2, 1, 3)}, 1, nh, s, hd)

	k := m.linear(h, p+"k_proj.weight", "", nkv*hd)
	k = m.op("reshape", map[string]any{"x": k, "shape": m.b.Ints(1, int32(s), int32(nkv), int32(hd))}, 1, s, nkv, hd)
	k = m.rmsNorm(k, p+"k_norm.weight", hd)
	k = m.applyRope(k, cos, sin)
	k = m.op("transpose", map[string]any{"x": k, "perm": m.b.Ints(0, 2, 1, 3)}, 1, nkv, s, hd)

	v := m.linear(h, p+"v_proj.weight", "", nkv*hd)
	v = m.op("reshape", map[string]any{"x": v, "shape": m.b.Ints(1, int32(s), int32(nkv), int32(hd))}, 1, s, nkv, hd)
	v = m.op("transpose", map[string]any{"x": v, "perm": m.b.Ints(0, 2, 1, 3)}, 1, nkv, s, hd)

	// repeat_kv: head j reads kv head j / (nh/nkv), so each kv head is repeated consecutively
	if rep := nh / nkv; rep > 1 {
		expand := func(x *mlpackage.Var) *mlpackage.Var {
			x = m.op("reshape", map[string]any{"x": x, "shape": m.b.Ints(1, int32(nkv), 1, int32(s), int32(hd))}, 1, nkv, 1, s, hd)
			x = m.op("tile", map[string]any{"x": x, "reps": m.b.Ints(1, 1, int32(rep), 1, 1)}, 1, nkv, rep, s, hd)
			return m.op("reshape", map[string]any{"x": x, "shape": m.b.Ints(1, int32(nh), int32(s), int32(hd))}, 1, nh, s, hd)
		}
		k, v = expand(k), expand(v)
	}

	scores := m.op("matmul", map[string]any{"x": q, "y": k, "transpose_x": m.b.ScalarBool(false), "transpose_y": m.b.ScalarBool(true)}, 1, nh, s, s)
	scores = m.op("mul", map[string]any{"x": scores, "y": m.b.ScalarF16(float32(1 / math.Sqrt(float64(hd))))}, 1, nh, s, s)
	scores = m.op("add", map[string]any{"x": scores, "y": m.mask(layerType)}, 1, nh, s, s)
	probs := m.op("softmax", map[string]any{"x": scores, "axis": m.b.ScalarI32(-1)}, 1, nh, s, s)
	o := m.op("matmul", map[string]any{"x": probs, "y": v, "transpose_x": m.b.ScalarBool(false), "transpose_y": m.b.ScalarBool(false)}, 1, nh, s, hd)
	o = m.op("transpose", map[string]any{"x": o, "perm": m.b.Ints(0, 2, 1, 3)}, 1, s, nh, hd)

	gate := m.linear(h, p+"gate_proj.weight", "", nh)
	gate = m.op("sigmoid", map[string]any{"x": gate}, 1, s, nh)
	gate = m.op("reshape", map[string]any{"x": gate, "shape": m.b.Ints(1, int32(s), int32(nh), 1)}, 1, s, nh, 1)
	o = m.op("mul", map[string]any{"x": o, "y": gate}, 1, s, nh, hd)
	o = m.op("reshape", map[string]any{"x": o, "shape": m.b.Ints(1, int32(s), int32(nh*hd))}, 1, s, nh*hd)
	return m.linearS(o, p+"o_proj.weight", "", m.d, m.opts.PostScale)
}

// rope returns cos and sin tables of shape (1, seq, 1, head_dim) for positions 0..seq-1, laid out
// as cat(freqs, freqs) like the reference.
func (m *builder) rope(layerType string) (cos, sin *mlpackage.Var) {
	inv := m.cfg.RopeInvFreq(layerType)
	hd, half := m.cfg.HeadDim, m.cfg.HeadDim/2
	c := make([]float32, m.seq*hd)
	s := make([]float32, m.seq*hd)
	for pos := 0; pos < m.seq; pos++ {
		for j := 0; j < half; j++ {
			a := float64(pos) * inv[j]
			c[pos*hd+j], c[pos*hd+j+half] = float32(math.Cos(a)), float32(math.Cos(a))
			s[pos*hd+j], s[pos*hd+j+half] = float32(math.Sin(a)), float32(math.Sin(a))
		}
	}
	shape := []int{1, m.seq, 1, hd}
	var err error
	if cos, err = m.b.F16s("rope_cos_"+layerType, shape, c); err != nil {
		m.fail("%v", err)
	}
	if sin, err = m.b.F16s("rope_sin_"+layerType, shape, s); err != nil {
		m.fail("%v", err)
	}
	return cos, sin
}

// applyRope rotates x (1, seq, heads, head_dim): x*cos + rotate_half(x)*sin, where rotate_half
// is cat(-x2, x1) over the two halves of the last axis.
func (m *builder) applyRope(x, cos, sin *mlpackage.Var) *mlpackage.Var {
	hd, half := m.cfg.HeadDim, m.cfg.HeadDim/2
	sh := x.Shape
	halfShape := append(append([]int{}, sh[:3]...), half)
	x1 := m.op("slice_by_index", map[string]any{"x": x, "begin": m.b.Ints(0, 0, 0, 0), "end": m.b.Ints(0, 0, 0, int32(half)),
		"begin_mask": m.bools(true, true, true, true), "end_mask": m.bools(true, true, true, false)}, halfShape...)
	x2 := m.op("slice_by_index", map[string]any{"x": x, "begin": m.b.Ints(0, 0, 0, int32(half)), "end": m.b.Ints(0, 0, 0, int32(hd)),
		"begin_mask": m.bools(true, true, true, false), "end_mask": m.bools(true, true, true, true)}, halfShape...)
	nx2 := m.op("mul", map[string]any{"x": x2, "y": m.b.ScalarF16(-1)}, halfShape...)
	rot := m.op("concat", map[string]any{"values": []*mlpackage.Var{nx2, x1}, "axis": m.b.ScalarI32(-1), "interleave": m.b.ScalarBool(false)}, sh...)
	a := m.op("mul", map[string]any{"x": x, "y": cos}, sh...)
	b := m.op("mul", map[string]any{"x": rot, "y": sin}, sh...)
	return m.op("add", map[string]any{"x": a, "y": b}, sh...)
}

// mask is the additive attention mask (1, 1, seq, seq): 0 where a query may see a key, a large
// negative value elsewhere (causal, and within the window on sliding layers).
func (m *builder) mask(layerType string) *mlpackage.Var {
	s := m.seq
	vals := make([]float32, s*s)
	for q := 0; q < s; q++ {
		for k := 0; k < s; k++ {
			visible := k <= q
			if layerType == "sliding_attention" && m.cfg.SlidingWindow > 0 {
				visible = visible && k > q-m.cfg.SlidingWindow
			}
			if !visible {
				vals[q*s+k] = -1e4
			}
		}
	}
	v, err := m.b.F16s("mask_"+layerType, []int{1, 1, s, s}, vals)
	if err != nil {
		m.fail("%v", err)
	}
	return v
}

func (m *builder) bools(v ...bool) *mlpackage.Var {
	x, err := m.b.Bools("c", []int{len(v)}, v)
	if err != nil {
		m.fail("%v", err)
	}
	return x
}

func (m *builder) str(s string) *mlpackage.Var {
	return m.b.Str("c", s)
}
