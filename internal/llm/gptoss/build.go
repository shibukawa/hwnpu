package gptoss

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

type builder struct {
	*blocks.Graph
	cfg     *Config
	opts    llm.Options
	seq, d  int
	ins     map[string]*mlpackage.Var
	inList  []mlpackage.Feature
	outList []*mlpackage.Var
}

func (m *builder) input(name string, shape ...int) *mlpackage.Var {
	if v, ok := m.ins[name]; ok {
		return v
	}
	v := m.B.Input(name, mlpackage.Float16, shape...)
	m.ins[name] = v
	m.inList = append(m.inList, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: shape})
	return v
}

func (m *builder) output(v *mlpackage.Var, name string) {
	m.outList = append(m.outList, m.B.NamedOp("identity", name, map[string]any{"x": v}, mlpackage.Float16, v.Shape...))
}

// build writes a part (llm.PlanRouted): every layer ends with its experts, so a part is one
// layer's step between two of the host's: from the experts' output of layer Start (moe_o_N added
// to x) to the router of layer Start+1, or to the attention of layer Start+1 when the host attends
// for it (Paged; then a part of no layers, NoLayers, goes on from the attention's output attn_o_N
// to the router). The first part ({0, NoLayers}) is layer 0 up to its router; the last ends with
// the final norm and the logits. A sliding layer attends in the program over a ring of its window
// (kv_k_N, kv_v_N, mask_sliding_attention) with its sink; a part before a full attention layer
// outputs that layer's sinks (attn_sink_N) for the host's merge.
func build(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = cfg.Format()
	}
	// The attention's projections stay fp16 unless a rule of the caller's says otherwise (a later
	// FormatFor rule wins): gpt-oss's attention does not bear tables of 16 rows. With them in lut8
	// the KL from the float32 reference on plain text is 0.29 (the answer of a conversation
	// 0.0088), in fp16 0.036 (0.0003), for a step about 20 % slower on an M3; every other rounding
	// of the Neural Engine's path together adds less than 0.04 (docs/improvements.md 7.5).
	opts.FormatFor = append([]string{"self_attn=fp16"}, opts.FormatFor...)
	noLayers := opts.Layers == llm.NoLayers
	first := noLayers && opts.Start == 0 // layer 0 up to its router
	afterAttn := noLayers && opts.Start > 0
	if !noLayers && (opts.Layers <= 0 || opts.Start+opts.Layers > cfg.NumLayers) {
		opts.Layers = cfg.NumLayers - opts.Start
	}
	if opts.Cache > 0 && opts.Paged <= 0 {
		return nil, nil, fmt.Errorf("gptoss: a program with a KV cache is split at its full attention (Paged)")
	}
	if opts.LastRow && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("gptoss: the logits of the last row alone (LastRow) are of a program with a KV cache")
	}
	start, end := opts.Start, opts.Start+opts.Layers
	paged := func(i int) bool { return opts.Paged > 0 && !cfg.IsSliding(i) }
	if afterAttn && !paged(start) {
		return nil, nil, fmt.Errorf("gptoss: a part of no layers from layer %d is the rest of a full attention layer the host attends for", start)
	}
	m := &builder{Graph: &blocks.Graph{Name: "gptoss", B: b, W: w, Opts: opts}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize, ins: map[string]*mlpackage.Var{}}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		h := x
		var y *mlpackage.Var
		switch {
		case first:
			h = m.operator(0, h)
			m.moeHead(0, h)
			y = m.y(h)
		case afterAttn:
			h = m.Op("add", map[string]any{"x": m.attnOutput(start, m.input(fmt.Sprintf("attn_o_%d", start), 1, cfg.NumHeads, m.seq, cfg.HeadDim)), "y": h}, h.Shape...)
			m.moeHead(start, h)
			y = m.y(h)
		default:
			h = m.Op("add", map[string]any{"x": h, "y": m.input(fmt.Sprintf("moe_o_%d", start), 1, m.seq, m.d)}, h.Shape...)
			if end != start+1 {
				m.Fail("the part of layers %d to %d holds a layer's experts: a part is one layer's step (llm.PlanRouted)", start, end-1)
			}
			switch {
			case end < cfg.NumLayers && paged(end):
				m.beforeAttention(end, h)
				y = m.y(h)
			case end < cfg.NumLayers:
				h = m.operator(end, h)
				m.moeHead(end, h)
				y = m.y(h)
			case opts.NoHead:
				y = m.y(h)
			default:
				if opts.LastRow {
					h = m.LastRow(h, m.input("last", 1, 1, m.seq))
				}
				h = m.RMSNorm(h, "model.norm.weight", m.d, cfg.NormEps, 1)
				if opts.Hidden {
					y = m.y(h)
				} else {
					y = m.Linear(h, cfg.Head(), "y", cfg.VocabSize)
				}
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

func (m *builder) y(h *mlpackage.Var) *mlpackage.Var {
	return m.B.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
}

// moeHead is the end of a part before layer i's experts: the post attention norm's output moe_x_N
// and the router's logits moe_logits_N, its weight in fp16 (llm.Routed).
func (m *builder) moeHead(i int, h *mlpackage.Var) {
	p := fmt.Sprintf("model.layers.%d.", i)
	f := m.RMSNorm(h, p+"post_attention_layernorm.weight", m.d, m.cfg.NormEps, 1)
	m.output(f, fmt.Sprintf("moe_x_%d", i))
	e := m.cfg.NumExperts
	wv, err := m.B.F16s(p+"mlp.router.weight", []int{e, m.d}, m.Get(p+"mlp.router.weight", e, m.d))
	if err != nil {
		m.Fail("%v", err)
	}
	bv, err := m.B.F16s(p+"mlp.router.bias", []int{e}, m.Get(p+"mlp.router.bias", e))
	if err != nil {
		m.Fail("%v", err)
	}
	m.output(m.Op("linear", map[string]any{"x": f, "weight": wv, "bias": bv}, 1, m.seq, e), fmt.Sprintf("moe_logits_%d", i))
}

// operator is the first half of a layer the program attends for: the norm, the attention, the
// residual.
func (m *builder) operator(i int, h *mlpackage.Var) *mlpackage.Var {
	n := m.RMSNorm(h, fmt.Sprintf("model.layers.%d.input_layernorm.weight", i), m.d, m.cfg.NormEps, 1)
	return m.Op("add", map[string]any{"x": m.attention(i, n), "y": h}, h.Shape...)
}

// beforeAttention ends a part before layer i's attention, which the host runs: the queries, the
// step's keys and values, and the sinks.
func (m *builder) beforeAttention(i int, h *mlpackage.Var) {
	n := m.RMSNorm(h, fmt.Sprintf("model.layers.%d.input_layernorm.weight", i), m.d, m.cfg.NormEps, 1)
	q, k, v := m.qkv(i, n)
	m.output(q, fmt.Sprintf("attn_q_%d", i))
	m.output(k, fmt.Sprintf("kv_k_%d_new", i))
	m.output(v, fmt.Sprintf("kv_v_%d_new", i))
	nh := m.cfg.NumHeads
	name := fmt.Sprintf("model.layers.%d.self_attn.sinks", i)
	sv, err := m.B.F16s(name, []int{1, nh}, m.Get(name, nh))
	if err != nil {
		m.Fail("%v", err)
	}
	m.output(sv, fmt.Sprintf("attn_sink_%d", i))
}

// rope is the rotary tables (1, seq, 1, head_dim): YaRN's, scaled by its attention factor.
func (m *builder) rope() (cos, sin *mlpackage.Var) {
	hd := m.cfg.HeadDim
	if m.opts.Cache > 0 {
		return m.input("rope_cos", 1, m.seq, 1, hd), m.input("rope_sin", 1, m.seq, 1, hd)
	}
	c, s := m.cfg.ropeValues(0, m.seq)
	shape := []int{1, m.seq, 1, hd}
	cv, err := m.B.F16s("rope_cos", shape, c)
	if err == nil {
		var sv *mlpackage.Var
		if sv, err = m.B.F16s("rope_sin", shape, s); err == nil {
			return cv, sv
		}
	}
	m.Fail("%v", err)
	return nil, nil
}

// ropeValues are the rotary tables of n positions from start, scaled by YaRN's attention factor.
func (c *Config) ropeValues(start, n int) (cos, sin []float32) {
	inv, f := c.RopeInvFreq()
	cos, sin = blocks.RopeValues(inv, start, n, c.HeadDim)
	for i := range cos {
		cos[i] *= float32(f)
		sin[i] *= float32(f)
	}
	return cos, sin
}

// qkv is the attention's queries (1, heads, seq, head_dim) and keys and values (1, kv_heads, seq,
// head_dim), the projections with their biases, RoPE on the queries and keys.
func (m *builder) qkv(i int, h *mlpackage.Var) (q, k, v *mlpackage.Var) {
	p := fmt.Sprintf("model.layers.%d.self_attn.", i)
	s, hd := m.seq, m.cfg.HeadDim
	cos, sin := m.rope()
	heads := func(name string, n int, rope bool) *mlpackage.Var {
		v := m.LinearBias(h, p+name+".weight", p+name+".bias", "", n*hd)
		v = m.Op("reshape", map[string]any{"x": v, "shape": m.B.Ints(1, int32(s), int32(n), int32(hd))}, 1, s, n, hd)
		if rope {
			v = m.ApplyRope(v, cos, sin)
		}
		return m.Op("transpose", map[string]any{"x": v, "perm": m.B.Ints(0, 2, 1, 3)}, 1, n, s, hd)
	}
	return heads("q_proj", m.cfg.NumHeads, true), heads("k_proj", m.cfg.NumKVHeads, true), heads("v_proj", m.cfg.NumKVHeads, false)
}

// attnOutput is the output projection, with its bias, of the attention's o (1, heads, seq,
// head_dim).
func (m *builder) attnOutput(i int, o *mlpackage.Var) *mlpackage.Var {
	s, nh, hd := m.seq, m.cfg.NumHeads, m.cfg.HeadDim
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, s, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(s), int32(nh*hd))}, 1, s, nh*hd)
	p := fmt.Sprintf("model.layers.%d.self_attn.", i)
	return m.LinearBias(o, p+"o_proj.weight", p+"o_proj.bias", "", m.d)
}

// attention is layer i's attention in the program: over a ring of the window and the step's own
// keys for a sliding layer with a cache, over the step's tokens alone without one (causal, and
// windowed for a sliding layer).
func (m *builder) attention(i int, h *mlpackage.Var) *mlpackage.Var {
	s, nkv, hd := m.seq, m.cfg.NumKVHeads, m.cfg.HeadDim
	q, k, v := m.qkv(i, h)
	var mask *mlpackage.Var
	if m.opts.Cache > 0 {
		if !m.cfg.IsSliding(i) {
			m.Fail("layer %d's full attention is the host's in a program with a cache", i)
		}
		m.output(k, fmt.Sprintf("kv_k_%d_new", i))
		m.output(v, fmt.Sprintf("kv_v_%d_new", i))
		t := m.cfg.cacheLen(i, m.opts.Cache)
		cat := func(name string, x *mlpackage.Var) *mlpackage.Var {
			c := m.input(name, 1, nkv, t, hd)
			return m.Op("concat", map[string]any{"values": []*mlpackage.Var{c, x}, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, nkv, t+s, hd)
		}
		k, v = cat(fmt.Sprintf("kv_k_%d", i), k), cat(fmt.Sprintf("kv_v_%d", i), v)
		mask = m.input("mask_sliding_attention", 1, 1, s, t+s)
	} else {
		window := 0
		name := "mask_full_attention"
		if m.cfg.IsSliding(i) {
			window, name = m.cfg.SlidingWindow, "mask_sliding_attention"
		}
		mask = m.CausalMask(name, s, window)
	}
	return m.attnOutput(i, m.sdpaSink(i, q, k, v, mask))
}

// sdpaSink is grouped attention whose softmax takes each head's sink besides the keys' scores
// (gpt-oss): a column of the sink's logit added to the scores, dropped after the softmax.
func (m *builder) sdpaSink(i int, q, k, v, mask *mlpackage.Var) *mlpackage.Var {
	nh, s, hd := q.Shape[1], q.Shape[2], q.Shape[3]
	nkv, sk := k.Shape[1], k.Shape[2]
	rep := nh / nkv
	qg := m.Op("reshape", map[string]any{"x": q, "shape": m.B.Ints(1, int32(nkv), int32(rep*s), int32(hd))}, 1, nkv, rep*s, hd)
	sc := m.Op("matmul", map[string]any{"x": qg, "y": k, "transpose_x": m.B.ScalarBool(false), "transpose_y": m.B.ScalarBool(true)}, 1, nkv, rep*s, sk)
	sc = m.Op("mul", map[string]any{"x": sc, "y": m.B.ScalarF16(float32(1 / math.Sqrt(float64(hd))))}, sc.Shape...)
	sc = m.Op("reshape", map[string]any{"x": sc, "shape": m.B.Ints(1, int32(nh), int32(s), int32(sk))}, 1, nh, s, sk)
	sc = m.Op("add", map[string]any{"x": sc, "y": mask}, 1, nh, s, sk)
	name := fmt.Sprintf("model.layers.%d.self_attn.sinks", i)
	sinks := m.Get(name, nh)
	col := make([]float32, nh*s)
	for h := range nh {
		for t := range s {
			col[h*s+t] = sinks[h]
		}
	}
	cv, err := m.B.F16s(name, []int{1, nh, s, 1}, col)
	if err != nil {
		m.Fail("%v", err)
	}
	sc = m.Op("concat", map[string]any{"values": []*mlpackage.Var{sc, cv}, "axis": m.B.ScalarI32(3), "interleave": m.B.ScalarBool(false)}, 1, nh, s, sk+1)
	probs := m.Op("softmax", map[string]any{"x": sc, "axis": m.B.ScalarI32(-1)}, 1, nh, s, sk+1)
	probs = m.Op("slice_by_index", map[string]any{"x": probs, "begin": m.B.Ints(0, 0, 0, 0), "end": m.B.Ints(1, int32(nh), int32(s), int32(sk)),
		"begin_mask": m.Bools(true, true, true, true), "end_mask": m.Bools(true, true, true, false)}, 1, nh, s, sk)
	probs = m.Op("reshape", map[string]any{"x": probs, "shape": m.B.Ints(1, int32(nkv), int32(rep*s), int32(sk))}, 1, nkv, rep*s, sk)
	o := m.Op("matmul", map[string]any{"x": probs, "y": v, "transpose_x": m.B.ScalarBool(false), "transpose_y": m.B.ScalarBool(false)}, 1, nkv, rep*s, hd)
	return m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(nh), int32(s), int32(hd))}, 1, nh, s, hd)
}

// StepInputs gives the rotary tables and the sliding layers' mask of a step of valid tokens from
// pos (llm.Cached). The sliding layers' caches are rings of the window: query t (position pos+t)
// sees ring slot j holding the latest position written to it, and the step's tokens up to t,
// those less than the window behind it. The full layers are the host's (Paged): their masks are
// its attention's.
func (c *Config) StepInputs(pos, valid, q, cache int) map[string][]float32 {
	out := map[string][]float32{}
	out["rope_cos"], out["rope_sin"] = c.ropeValues(pos, q)
	keys := c.SlidingWindow
	if cache < keys {
		keys = cache
	}
	ring := keys < cache
	mask := make([]float32, q*(keys+q))
	for t := 0; t < q; t++ {
		p := pos + t
		row := mask[t*(keys+q):]
		for j := 0; j < keys+q; j++ {
			var key int
			var visible bool
			switch {
			case j >= keys:
				key = pos + j - keys
				visible = j-keys <= t
			case ring:
				key = pos - 1 - ((pos-1-j)%keys+keys)%keys
				visible = key >= 0
			default:
				key = j
				visible = j < pos
			}
			if p-key >= c.SlidingWindow {
				visible = false
			}
			if !visible {
				row[j] = -1e4
			}
		}
	}
	out["mask_sliding_attention"] = mask
	return out
}

// PagedStepInputs is StepInputs: the inputs of the sliding layers and the rotary tables (the
// full layers' masks are the host's).
func (c *Config) PagedStepInputs(pos, valid, q, cache int) map[string][]float32 {
	return c.StepInputs(pos, valid, q, cache)
}
