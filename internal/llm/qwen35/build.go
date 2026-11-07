package qwen35

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

type builder struct {
	*blocks.Graph
	cfg  *Config
	opts llm.Options
	seq  int
	d    int
	// With a KV cache (opts.Cache): the inputs and outputs besides x and y.
	ins     map[string]*mlpackage.Var
	inList  []mlpackage.Feature
	outList []*mlpackage.Var
}

// input declares an fp16 program input once and returns it.
func (m *builder) input(name string, shape ...int) *mlpackage.Var {
	if v, ok := m.ins[name]; ok {
		return v
	}
	v := m.B.Input(name, mlpackage.Float16, shape...)
	m.ins[name] = v
	m.inList = append(m.inList, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: shape})
	return v
}

// output makes v a program output called name.
func (m *builder) output(v *mlpackage.Var, name string) {
	m.outList = append(m.outList, m.B.NamedOp("identity", name, map[string]any{"x": v}, mlpackage.Float16, v.Shape...))
}

// Build writes the program for the model into b. The program's input is x, the fp16 token
// embeddings (1, seq, hidden), looked up on the host; the sequence starts at position 0 with the
// Gated DeltaNet's state at zero. The output is y: logits (1, seq, vocab), or a hidden state
// (1, seq, hidden) for a truncated model. A program with a KV cache has more inputs and outputs;
// build it with BuildCached.
func Build(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts llm.Options) (in, out mlpackage.Feature, err error) {
	if opts.Cache > 0 {
		return in, out, fmt.Errorf("qwen35: a program with a KV cache has more inputs and outputs; use BuildCached")
	}
	ins, outs, err := BuildCached(b, cfg, w, opts)
	if err != nil {
		return in, out, err
	}
	return ins[0], outs[0], nil
}

// BuildCached is Build for any Options, returning every input and output: x and y first, then,
// with opts.Cache, the state the host holds between steps (llm.Cached):
//
//   - kv_k_N, kv_v_N: the KV cache of each full attention layer N
//   - win_conv_N: the last ConvKernel-1 rows of the Gated DeltaNet layer N's projected queries,
//     keys and values, the state of its short convolution
//   - st_rec_N (1, value heads, key dim, value dim) with output st_rec_N_new: the Gated DeltaNet's
//     recurrent state, which the step replaces as a whole. It is the state after the step's valid
//     tokens: the padding after them leaves it as it is (step_valid)
//
// and the inputs StepInputs fills: mask_full_attention, rope_cos_full_attention,
// rope_sin_full_attention and step_valid (1, seq, 1), 1 for the step's tokens and 0 for padding.
//
// With opts.Paged the program is one part of the model split at its full attention layers
// (llm.Paged): a part that ends where such a layer N begins outputs its queries attn_q_N, its
// keys and values kv_k_N_new and kv_v_N_new and its output gate attn_gate_N (1, seq, heads,
// head_dim), and the next part takes the attention's output attn_o_N and the gate.
func BuildCached(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = Arch.Format
	}
	if opts.Layers <= 0 || opts.Start+opts.Layers > cfg.NumLayers {
		opts.Layers = cfg.NumLayers - opts.Start
	}
	if opts.Paged > 0 && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("qwen35: a program split at its attention (Paged) needs a KV cache")
	}
	if opts.LastRow && (opts.Cache <= 0 || opts.Hidden) {
		return nil, nil, fmt.Errorf("qwen35: the logits of the last row alone (LastRow) are of a program with a KV cache")
	}
	if _, chunked := llm.CacheChunks(opts.Cache); opts.Cache > 0 && opts.Paged == 0 && chunked {
		return nil, nil, fmt.Errorf("qwen35: a KV cache of %d is over the Neural Engine's dimension limit; split the model at its attention (Paged) for a longer one", opts.Cache)
	}
	m := &builder{Graph: &blocks.Graph{Name: "qwen35", B: b, W: w, Opts: opts}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize, ins: map[string]*mlpackage.Var{}}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		h := x
		for i := opts.Start; i < opts.Start+opts.Layers; i++ {
			if i == opts.Start && i > 0 && m.paged(i) {
				h = m.afterAttention(i, h) // the host attended between the parts
				continue
			}
			h = m.decoderLayer(i, h)
		}
		if end := opts.Start + opts.Layers; end < cfg.NumLayers && m.paged(end) {
			m.beforeAttention(end, h) // the host attends next; y is the hidden state entering end
		}
		var y *mlpackage.Var
		switch {
		case opts.Start+opts.Layers < cfg.NumLayers || opts.NoHead:
			y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		default:
			if opts.LastRow {
				h = m.LastRow(h, m.input("last", 1, 1, m.seq))
			}
			h = m.norm(h, cfg.prefix()+"norm.weight")
			if opts.Hidden {
				y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
			} else {
				y = m.Linear(h, cfg.Head(), "y", cfg.VocabSize)
			}
		}
		outs = []mlpackage.Feature{{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}}
		for _, v := range m.outList {
			outs = append(outs, mlpackage.Feature{Name: v.Name, DType: mlpackage.Float16, Shape: v.Shape})
		}
		ins = append(ins, m.inList...)
	})
	return ins, outs, err
}

// norm is Qwen3_5RMSNorm, zero centered: x normalized times 1 + w.
func (m *builder) norm(x *mlpackage.Var, name string) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	w := m.Get(name, n)
	for i := range w {
		w[i]++
	}
	return m.RMSNormWith(x, name, w, m.cfg.NormEps, 1)
}

// decoderLayer is Qwen3_5DecoderLayer: the token mixer (Gated DeltaNet or full attention) and
// the MLP, each after its norm and added to the residual.
func (m *builder) decoderLayer(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("%slayers.%d.", m.cfg.prefix(), i)
	residual := h
	h = m.norm(h, p+"input_layernorm.weight")
	if m.cfg.IsLinear(i) {
		h = m.gatedDeltaNet(i, h)
	} else {
		h = m.attention(i, h)
	}
	return m.layerRest(i, h, residual)
}

// layerRest is a layer after its token mixer (h) given the residual: the residual add and the
// MLP after its norm.
func (m *builder) layerRest(i int, h, residual *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("%slayers.%d.", m.cfg.prefix(), i)
	h = m.Op("add", map[string]any{"x": h, "y": residual}, h.Shape...)
	residual = h
	f := m.norm(h, p+"post_attention_layernorm.weight")
	gate := m.silu(m.Linear(f, p+"mlp.gate_proj.weight", "", m.cfg.IntermediateSize))
	f = m.Op("mul", map[string]any{"x": gate, "y": m.Linear(f, p+"mlp.up_proj.weight", "", m.cfg.IntermediateSize)}, gate.Shape...)
	f = m.Linear(f, p+"mlp.down_proj.weight", "", m.d) // blocks.SwiGLU with silu (m.silu)
	return m.Op("add", map[string]any{"x": f, "y": residual}, f.Shape...)
}

// slice is x[from:to] along axis, the other axes whole.
func (m *builder) slice(x *mlpackage.Var, axis, from, to int) *mlpackage.Var {
	r := len(x.Shape)
	begin, end := make([]int32, r), make([]int32, r)
	bm, em := make([]bool, r), make([]bool, r)
	for a := range r {
		bm[a], em[a] = a != axis, a != axis
	}
	begin[axis], end[axis] = int32(from), int32(to)
	shape := append([]int{}, x.Shape...)
	shape[axis] = to - from
	return m.Op("slice_by_index", map[string]any{"x": x, "begin": m.B.Ints(begin...), "end": m.B.Ints(end...),
		"begin_mask": m.Bools(bm...), "end_mask": m.Bools(em...)}, shape...)
}

func (m *builder) reshape(x *mlpackage.Var, shape ...int) *mlpackage.Var {
	s := make([]int32, len(shape))
	for i, v := range shape {
		s[i] = int32(v)
	}
	return m.Op("reshape", map[string]any{"x": x, "shape": m.B.Ints(s...)}, shape...)
}

// swap12 transposes axes 1 and 2 of a rank 4 tensor: (1, seq, heads, d) and (1, heads, seq, d).
func (m *builder) swap12(x *mlpackage.Var) *mlpackage.Var {
	s := x.Shape
	return m.Op("transpose", map[string]any{"x": x, "perm": m.B.Ints(0, 2, 1, 3)}, s[0], s[2], s[1], s[3])
}

func (m *builder) const16(name string, shape []int, vals []float32) *mlpackage.Var {
	return m.F16s(name, shape, vals)
}

// paged reports whether layer i's attention runs on the host between the parts (Options.Paged).
func (m *builder) paged(i int) bool {
	return m.opts.Paged > 0 && m.opts.Cache > 0 && !m.cfg.IsLinear(i)
}

// beforeAttention is the start of layer i, whose attention the host runs (llm.Paged): it outputs
// the queries attn_q_N, the step's keys and values kv_k_N_new and kv_v_N_new, and the output gate
// attn_gate_N for the next part.
func (m *builder) beforeAttention(i int, h *mlpackage.Var) {
	p := fmt.Sprintf("%slayers.%d.", m.cfg.prefix(), i)
	q, k, v, gate := m.qkvGate(i, m.norm(h, p+"input_layernorm.weight"))
	m.output(q, fmt.Sprintf("attn_q_%d", i))
	m.output(k, fmt.Sprintf("kv_k_%d_new", i))
	m.output(v, fmt.Sprintf("kv_v_%d_new", i))
	m.output(gate, fmt.Sprintf("attn_gate_%d", i))
}

// afterAttention is the rest of layer i from its attention's output, attn_o_N (1, heads, seq,
// head_dim), with the gate the part before output and x the hidden state entering the layer.
func (m *builder) afterAttention(i int, x *mlpackage.Var) *mlpackage.Var {
	s, nh, hd := m.seq, m.cfg.NumHeads, m.cfg.HeadDim
	o := m.input(fmt.Sprintf("attn_o_%d", i), 1, nh, s, hd)
	gate := m.input(fmt.Sprintf("attn_gate_%d", i), 1, s, nh, hd)
	return m.layerRest(i, m.attnOutput(i, o, gate), x)
}

// attention is Qwen3_5Attention: grouped query attention whose q_proj gives each head its
// queries and an output gate side by side, QK norms (zero centered) over each head, RoPE over the
// first RotaryDim of each head (rotate_half within them), causal softmax attention, and the
// output times sigmoid of the gate, channel by channel.
func (m *builder) attention(i int, h *mlpackage.Var) *mlpackage.Var {
	q, k, v, gate := m.qkvGate(i, h)
	nkv, hd := m.cfg.NumKVHeads, m.cfg.HeadDim
	s := m.seq
	var o *mlpackage.Var
	if m.opts.Cache > 0 {
		t := m.opts.Cache
		m.output(k, fmt.Sprintf("kv_k_%d_new", i))
		m.output(v, fmt.Sprintf("kv_v_%d_new", i))
		cat := func(name string, x *mlpackage.Var) *mlpackage.Var {
			c := m.input(name, 1, nkv, t, hd)
			return m.Op("concat", map[string]any{"values": []*mlpackage.Var{c, x}, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, nkv, t+s, hd)
		}
		k, v = cat(fmt.Sprintf("kv_k_%d", i), k), cat(fmt.Sprintf("kv_v_%d", i), v)
		mask := m.input("mask_full_attention", 1, 1, s, t+s)
		o = m.SDPAGrouped(q, k, v, 1/math.Sqrt(float64(hd)), func() *mlpackage.Var { return mask })
	} else {
		o = m.Attention(q, k, v, "mask_full_attention", 0)
	}
	return m.attnOutput(i, o, gate)
}

// qkvGate is the attention's queries (1, heads, seq, head_dim), keys and values (1, kv_heads,
// seq, head_dim) with the QK norms and RoPE, and the output gate (1, seq, heads, head_dim), from
// the normed hidden state.
func (m *builder) qkvGate(i int, h *mlpackage.Var) (q, k, v, gate *mlpackage.Var) {
	p := fmt.Sprintf("%slayers.%d.self_attn.", m.cfg.prefix(), i)
	s, nh, nkv, hd := m.seq, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim
	rot := m.cfg.RotaryDim()
	var cos, sin *mlpackage.Var
	if m.opts.Cache > 0 {
		cos, sin = m.input("rope_cos_full_attention", 1, s, 1, rot), m.input("rope_sin_full_attention", 1, s, 1, rot)
	} else {
		cos, sin = m.RopeTables("full_attention", m.cfg.RopeInvFreq(), s, rot)
	}
	rope := func(x *mlpackage.Var) *mlpackage.Var {
		if rot == hd {
			return m.ApplyRope(x, cos, sin)
		}
		r := m.ApplyRope(m.slice(x, 3, 0, rot), cos, sin)
		pass := m.slice(x, 3, rot, hd)
		return m.Op("concat", map[string]any{"values": []*mlpackage.Var{r, pass}, "axis": m.B.ScalarI32(3), "interleave": m.B.ScalarBool(false)}, x.Shape...)
	}

	qg := m.Linear(h, p+"q_proj.weight", "", 2*nh*hd)
	qg = m.reshape(qg, 1, s, nh, 2*hd)
	q = m.slice(qg, 3, 0, hd)
	gate = m.slice(qg, 3, hd, 2*hd)
	q = m.swap12(rope(m.norm(q, p+"q_norm.weight")))

	k = m.reshape(m.Linear(h, p+"k_proj.weight", "", nkv*hd), 1, s, nkv, hd)
	k = m.swap12(rope(m.norm(k, p+"k_norm.weight")))
	v = m.swap12(m.reshape(m.Linear(h, p+"v_proj.weight", "", nkv*hd), 1, s, nkv, hd))
	return q, k, v, gate
}

// attnOutput is the attention's output o (1, heads, seq, head_dim) times sigmoid of the gate
// (1, seq, heads, head_dim), through o_proj.
func (m *builder) attnOutput(i int, o, gate *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("%slayers.%d.self_attn.", m.cfg.prefix(), i)
	s, nh, hd := m.seq, m.cfg.NumHeads, m.cfg.HeadDim
	o = m.swap12(o) // (1, s, nh, hd)
	o = m.Op("mul", map[string]any{"x": o, "y": m.Op("sigmoid", map[string]any{"x": gate}, gate.Shape...)}, o.Shape...)
	o = m.reshape(o, 1, s, nh*hd)
	return m.Linear(o, p+"o_proj.weight", "", m.d)
}

// gatedDeltaNet is Qwen3_5GatedDeltaNet: the projections of the queries, keys and values go
// through a depthwise causal convolution and silu; the queries and keys are L2 normalized (the
// queries also divided by the square root of their size) and each key head serves consecutive
// value heads (repeat_interleave). The gated delta rule runs over them with the decay
// exp(-exp(A_log) * softplus(a + dt_bias)) and the write strength sigmoid(b), and its output goes
// through an RMS norm per head gated by silu(z) to out_proj.
func (m *builder) gatedDeltaNet(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("%slayers.%d.linear_attn.", m.cfg.prefix(), i)
	c := m.cfg
	s, hk, hv, dk, dv := m.seq, c.LinearKeyHeads, c.LinearValueHeads, c.LinearKeyDim, c.LinearValueDim
	kd, vd := c.keyDim(), c.valueDim()

	qkv := m.causalConv(i, m.Linear(h, p+"in_proj_qkv.weight", "", c.convDim()))
	z := m.Linear(h, p+"in_proj_z.weight", "", vd)
	b := m.Linear(h, p+"in_proj_b.weight", "", hv)
	a := m.Linear(h, p+"in_proj_a.weight", "", hv)

	// l2norm(x) is x / sqrt(sum x^2 + eps): an RMS norm with epsilon eps / n times 1 / sqrt(n)
	l2 := func(x *mlpackage.Var, name string, scale float64) *mlpackage.Var {
		n := x.Shape[len(x.Shape)-1]
		w := make([]float32, n)
		for j := range w {
			w[j] = float32(scale / math.Sqrt(float64(n)))
		}
		return m.RMSNormWith(x, name, w, 1e-6/float64(n), 1)
	}
	repeat := func(x *mlpackage.Var) *mlpackage.Var { // (1, s, hk, dk) -> (1, s, hv, dk)
		if r := hv / hk; r > 1 {
			x = m.reshape(x, 1, s, hk, 1, dk)
			x = m.Op("tile", map[string]any{"x": x, "reps": m.B.Ints(1, 1, 1, int32(r), 1)}, 1, s, hk, r, dk)
			x = m.reshape(x, 1, s, hv, dk)
		}
		return x
	}
	// the queries are scaled up by oScale: o = q S is read only by the gated norm, which does not
	// change with its input's scale, and at their own size (1/sqrt(dk) times a unit vector) the
	// products in q S are around 1e-5, under fp16's smallest normal value, which the Neural
	// Engine does not keep: o was 6% off on an M3 and the norm made that 17%
	q := l2(m.reshape(m.slice(qkv, 2, 0, kd), 1, s, hk, dk), p+"l2_q", oScale/math.Sqrt(float64(dk)))
	k := l2(m.reshape(m.slice(qkv, 2, kd, 2*kd), 1, s, hk, dk), p+"l2_k", 1)
	v := m.reshape(m.slice(qkv, 2, 2*kd, 2*kd+vd), 1, s, hv, dv)
	q, k, v = m.swap12(repeat(q)), m.swap12(repeat(k)), m.swap12(v) // (1, hv, s, d)

	// beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias), both (1, s, hv); with a cache
	// the padding after the step's tokens gets beta 0 and g 0, which leaves the state as it is
	beta := m.Op("sigmoid", map[string]any{"x": b}, b.Shape...)
	alog, dt := m.Get(p+"A_log", hv), m.Get(p+"dt_bias", hv)
	negA := make([]float32, hv)
	for j, v := range alog {
		negA[j] = float32(-math.Exp(float64(v)))
	}
	g := m.Op("add", map[string]any{"x": a, "y": m.const16(p+"dt_bias", []int{1, 1, hv}, dt)}, a.Shape...)
	g = m.softplus(g)
	g = m.Op("mul", map[string]any{"x": g, "y": m.const16(p+"A_log", []int{1, 1, hv}, negA)}, g.Shape...)
	if m.opts.Cache > 0 {
		valid := m.input("step_valid", 1, s, 1)
		beta = m.Op("mul", map[string]any{"x": beta, "y": valid}, beta.Shape...)
		g = m.Op("mul", map[string]any{"x": g, "y": valid}, g.Shape...)
	}
	decay := m.Op("exp", map[string]any{"x": g}, g.Shape...)
	perHead := func(x *mlpackage.Var) *mlpackage.Var { // (1, s, hv) -> (1, hv, s, 1)
		x = m.Op("transpose", map[string]any{"x": x, "perm": m.B.Ints(0, 2, 1)}, 1, hv, s)
		return m.reshape(x, 1, hv, s, 1)
	}

	var state *mlpackage.Var
	if m.opts.Cache > 0 {
		state = m.input(fmt.Sprintf("st_rec_%d", i), 1, hv, dk, dv)
	}
	o, state := m.deltaRule(q, k, v, perHead(decay), perHead(beta), state)
	if m.opts.Cache > 0 {
		m.output(state, fmt.Sprintf("st_rec_%d_new", i))
	}

	// Qwen3_5RMSNormGated over each head, norm(o) * w * silu(z), of o / oScale, normalized as o / oNormIn with eps grown by
	// (oScale / oNormIn)^2. The division is a multiply of its own, not the norm's inScale, which
	// would divide the norm's weight by oNormIn into fp16, below its normal values for weights
	// under 0.004
	o = m.Op("mul", map[string]any{"x": m.swap12(o), "y": m.B.ScalarF16(1.0 / oNormIn)}, 1, s, hv, dv)
	r := float64(oScale / oNormIn)
	o = m.RMSNorm(o, p+"norm.weight", dv, c.NormEps*r*r, 1) // (1, s, hv, dv)
	zg := m.reshape(z, 1, s, hv, dv)
	zg = m.silu(zg)
	o = m.Op("mul", map[string]any{"x": o, "y": zg}, o.Shape...)
	return m.Linear(m.reshape(o, 1, s, vd), p+"out_proj.weight", "", m.d)
}

// oScale is how much the Gated DeltaNet's queries, and so its output o = q S, are scaled up
// (gatedDeltaNet), and oNormIn by how much o is scaled down before the gated norm squares it:
// the squares of o * 16 neither overflow (o at most a few hundred) nor matter where they
// underflow (eps * 256 dominates there, as eps does at o's own size).
const (
	oScale  = 1024
	oNormIn = 64
)

// softplus is log(1 + exp(x)) written as relu(x) + log(1 + exp(-|x|)), which neither overflows
// nor underflows in fp16. The Neural Engine's own softplus gave 0 for inputs where it should
// not (an M3: the decay exp(-exp(A_log) softplus(a + dt_bias)) of Qwen3.5-4B's Gated DeltaNet
// came out 1 where the CPU gave 0.003 to 0.13, and the model lost track of long prompts).
func (m *builder) softplus(x *mlpackage.Var) *mlpackage.Var {
	pos := m.Op("relu", map[string]any{"x": x}, x.Shape...)
	neg := m.Op("mul", map[string]any{"x": m.Op("abs", map[string]any{"x": x}, x.Shape...), "y": m.B.ScalarF16(-1)}, x.Shape...)
	tail := m.Op("add", map[string]any{"x": m.Op("exp", map[string]any{"x": neg}, x.Shape...), "y": m.B.ScalarF16(1)}, x.Shape...)
	tail = m.Op("log", map[string]any{"x": tail, "epsilon": m.B.ScalarF16(0)}, x.Shape...) // tail >= 1
	return m.Op("add", map[string]any{"x": pos, "y": tail}, x.Shape...)
}

// causalConv is the depthwise causal convolution of the Gated DeltaNet over x (1, seq, channels)
// followed by silu, as ConvKernel shifted multiply adds over the sequence padded on the left: by
// zeros, or with a cache by the last rows before the step (win_conv_N, output with the step's
// rows as win_conv_N_new).
func (m *builder) causalConv(i int, x *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("%slayers.%d.linear_attn.", m.cfg.prefix(), i)
	s, ch, k := m.seq, x.Shape[2], m.cfg.ConvKernel
	cw := m.Get(p+"conv1d.weight", ch, 1, k)
	padded := x
	if k > 1 {
		if m.opts.Cache > 0 {
			win := m.input(fmt.Sprintf("win_conv_%d", i), 1, k-1, ch)
			padded = m.Op("concat", map[string]any{"values": []*mlpackage.Var{win, x}, "axis": m.B.ScalarI32(1), "interleave": m.B.ScalarBool(false)}, 1, s+k-1, ch)
			m.output(padded, fmt.Sprintf("win_conv_%d_new", i))
		} else {
			padded = m.Op("pad", map[string]any{"x": x, "pad": m.B.Ints(int32(k-1), 0, 0, 0), "mode": m.Str("constant"), "constant_val": m.B.ScalarF16(0)}, 1, s+k-1, ch)
		}
	}
	// y[t] = sum_j w[j] * x[t + j - (k-1)]: tap j reads the padded sequence shifted by j
	var y *mlpackage.Var
	for j := 0; j < k; j++ {
		tap := make([]float32, ch)
		for c := range tap {
			tap[c] = cw[c*k+j]
		}
		term := m.Op("mul", map[string]any{"x": m.slice(padded, 1, j, j+s), "y": m.const16(fmt.Sprintf("%sconv1d.weight.tap%d", p, j), []int{1, 1, ch}, tap)}, 1, s, ch)
		if y == nil {
			y = term
		} else {
			y = m.Op("add", map[string]any{"x": y, "y": term}, 1, s, ch)
		}
	}
	return m.silu(y)
}

// silu is x / (1 + exp(-x)). The Neural Engine's own silu is about 1.4% off over the
// convolution's outputs of Qwen3.5-4B (an M3), and much more on the small values of its tail,
// which the L2 norm of the keys then scales up to unit rows. Past x = -11, exp(-x) overflows fp16
// to inf and the quotient is 0 (silu is under 2e-4 there).
func (m *builder) silu(x *mlpackage.Var) *mlpackage.Var {
	neg := m.Op("mul", map[string]any{"x": x, "y": m.B.ScalarF16(-1)}, x.Shape...)
	den := m.Op("add", map[string]any{"x": m.Op("exp", map[string]any{"x": neg}, x.Shape...), "y": m.B.ScalarF16(1)}, x.Shape...)
	return m.Op("real_div", map[string]any{"x": x, "y": den}, x.Shape...)
}

// deltaRule is the gated delta rule (torch_recurrent_gated_delta_rule) over q, k (1, heads, seq,
// dk), v (1, heads, seq, dv), decay = exp(g) and beta (1, heads, seq, 1), one token after the
// other from the state S (1, heads, dk, dv), nil for zeros:
//
//	S = S * decay_t;  S = S + k_t^T ((v_t - k_t S) * beta_t);  o_t = q_t S
//
// It returns the outputs (1, heads, seq, dv) and the state after the last token.
func (m *builder) deltaRule(q, k, v, decay, beta, state *mlpackage.Var) (o, st *mlpackage.Var) {
	hv, s, dk, dv := q.Shape[1], q.Shape[2], q.Shape[3], v.Shape[3]
	mm := func(x, y *mlpackage.Var, tx bool, shape ...int) *mlpackage.Var {
		return m.Op("matmul", map[string]any{"x": x, "y": y, "transpose_x": m.B.ScalarBool(tx), "transpose_y": m.B.ScalarBool(false)}, shape...)
	}
	var outs []*mlpackage.Var
	for t := 0; t < s; t++ {
		qt, kt, vt := m.slice(q, 2, t, t+1), m.slice(k, 2, t, t+1), m.slice(v, 2, t, t+1)
		bt := m.slice(beta, 2, t, t+1)
		var delta *mlpackage.Var
		if state == nil { // S = 0: nothing to decay or read
			delta = m.Op("mul", map[string]any{"x": vt, "y": bt}, 1, hv, 1, dv)
		} else {
			state = m.Op("mul", map[string]any{"x": state, "y": m.slice(decay, 2, t, t+1)}, 1, hv, dk, dv)
			mem := mm(kt, state, false, 1, hv, 1, dv)
			delta = m.Op("sub", map[string]any{"x": vt, "y": mem}, 1, hv, 1, dv)
			delta = m.Op("mul", map[string]any{"x": delta, "y": bt}, 1, hv, 1, dv)
		}
		upd := mm(kt, delta, true, 1, hv, dk, dv)
		if state == nil {
			state = upd
		} else {
			state = m.Op("add", map[string]any{"x": state, "y": upd}, 1, hv, dk, dv)
		}
		outs = append(outs, mm(qt, state, false, 1, hv, 1, dv))
	}
	if len(outs) == 1 {
		return outs[0], state
	}
	return m.Op("concat", map[string]any{"values": outs, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, hv, s, dv), state
}

// StepInputs gives the inputs of a step of valid tokens from pos (llm.Cached): the causal mask
// and rotary tables of the full attention layers, and step_valid. Query t (position pos+t) sees
// the cached positions before pos and the step's tokens up to itself.
func (c *Config) StepInputs(pos, valid, q, cache int) map[string][]float32 {
	in := c.PagedStepInputs(pos, valid, q, cache)
	mask := make([]float32, q*(cache+q))
	for t := 0; t < q; t++ {
		row := mask[t*(cache+q):]
		for j := 0; j < cache+q; j++ {
			if j < cache && j >= pos || j >= cache && j-cache > t {
				row[j] = -1e4
			}
		}
	}
	in["mask_full_attention"] = mask
	return in
}

// PagedStepInputs is StepInputs for a program split at its attention (llm.Paged): without the
// mask of the full attention layers, which the host's attention makes.
func (c *Config) PagedStepInputs(pos, valid, q, cache int) map[string][]float32 {
	ok := make([]float32, q)
	for t := range min(valid, q) {
		ok[t] = 1
	}
	cos, sin := blocks.RopeValues(c.RopeInvFreq(), pos, q, c.RotaryDim())
	return map[string][]float32{
		"rope_cos_full_attention": cos,
		"rope_sin_full_attention": sin,
		"step_valid":              ok,
	}
}

// PagedLayers are the full attention layers: split at them (llm.Options.Paged), their attention
// runs on the host's cache in chunks, so that a step costs in proportion to the conversation.
func (c *Config) PagedLayers() []int { return c.FullLayers() }

// PagedShape is the full attention's heads, kv heads and head size (llm.Paged), its scores
// scaled by 1/sqrt(head_dim).
func (c *Config) PagedShape() paged.Shape {
	return paged.Shape{Heads: c.NumHeads, KVHeads: c.NumKVHeads, HeadDim: c.HeadDim}
}

// PagedKV is layer itself: every full attention layer has its own keys and values.
func (c *Config) PagedKV(layer int) int { return layer }
