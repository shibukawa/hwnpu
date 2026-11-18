package lfm2moe

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

// build writes the part of layers [Start, Start+Layers) (all when Layers is 0): a part starting at
// a mixture of experts layer begins with its experts' output (moe_o_N, added to x), and one ending
// where such a layer begins ends with that layer's operator and router (llm.Routed); its y is then
// the hidden state the experts' output is added to. The last part ends with the final norm and the
// logits (of the last row alone with LastRow). With Cache, the KV caches of the attention layers
// (kv_k_N, kv_v_N), the convolutions' windows (win_conv_N) and the masks and rotary tables
// (mask_full_attention, rope_cos_full_attention, rope_sin_full_attention) are inputs, as
// llm.Cached describes.
func build(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = cfg.Format()
	}
	if opts.Layers <= 0 || opts.Start+opts.Layers > cfg.NumLayers {
		opts.Layers = cfg.NumLayers - opts.Start
	}
	if opts.Paged > 0 {
		return nil, nil, fmt.Errorf("lfm2moe: the model is cut at its experts and not split at its attention (-paged)")
	}
	if opts.LastRow && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("lfm2moe: the logits of the last row alone (LastRow) are of a program with a KV cache")
	}
	start, end := opts.Start, opts.Start+opts.Layers
	m := &builder{Graph: &blocks.Graph{Name: "lfm2moe", B: b, W: w, Opts: opts}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize, ins: map[string]*mlpackage.Var{}}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		h := x
		i := start
		if start > 0 && cfg.IsRouted(start) {
			h = m.Op("add", map[string]any{"x": h, "y": m.input(fmt.Sprintf("moe_o_%d", start), 1, m.seq, m.d)}, h.Shape...)
			i++
		}
		for ; i < end; i++ {
			if cfg.IsRouted(i) {
				m.Fail("layer %d's experts are inside the part of layers %d to %d: a part ends where they begin (llm.PlanRouted)", i, start, end-1)
			}
			h = m.dense(i, m.operator(i, h))
		}
		var y *mlpackage.Var
		switch {
		case end < cfg.NumLayers:
			if !cfg.IsRouted(end) {
				m.Fail("the part of layers %d to %d ends at a dense layer", start, end-1)
			}
			h = m.operator(end, h)
			p := fmt.Sprintf("model.layers.%d.", end)
			f := m.RMSNorm(h, p+"ffn_norm.weight", m.d, cfg.NormEps, 1)
			m.output(f, fmt.Sprintf("moe_x_%d", end))
			m.output(m.router(f, p+"feed_forward.gate.weight"), fmt.Sprintf("moe_logits_%d", end))
			y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		case opts.NoHead:
			y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		default:
			if opts.LastRow {
				h = m.LastRow(h, m.input("last", 1, 1, m.seq))
			}
			h = m.RMSNorm(h, "model.embedding_norm.weight", m.d, cfg.NormEps, 1)
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

// router is the router's logits (1, seq, experts), its weight in fp16: the choice of the experts
// turns on small differences of the scores.
func (m *builder) router(f *mlpackage.Var, name string) *mlpackage.Var {
	e := m.cfg.NumExperts
	wv, err := m.B.F16s(name, []int{e, m.d}, m.Get(name, e, m.d))
	if err != nil {
		m.Fail("%v", err)
	}
	return m.Op("linear", map[string]any{"x": f, "weight": wv}, 1, m.seq, e)
}

// operator is the first half of layer i: the norm, the convolution or attention, the residual.
func (m *builder) operator(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("model.layers.%d.", i)
	n := m.RMSNorm(h, p+"operator_norm.weight", m.d, m.cfg.NormEps, 1)
	if m.cfg.IsAttention(i) {
		n = m.attention(i, n)
	} else {
		n = m.shortConv(i, n)
	}
	return m.Op("add", map[string]any{"x": n, "y": h}, h.Shape...)
}

// dense is the second half of a dense layer: the norm, the SwiGLU, the residual.
func (m *builder) dense(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("model.layers.%d.", i)
	f := m.RMSNorm(h, p+"ffn_norm.weight", m.d, m.cfg.NormEps, 1)
	f = m.SwiGLU(f, p+"feed_forward.w1.weight", p+"feed_forward.w3.weight", p+"feed_forward.w2.weight", m.cfg.IntermediateSize, m.d, 1)
	return m.Op("add", map[string]any{"x": f, "y": h}, h.Shape...)
}

// shortConv is LFM2's gated short convolution as K shifted multiply adds (lfm3's taps): in_proj
// splits into B, C and x; the causal depthwise convolution runs over B*x, C gates it.
func (m *builder) shortConv(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("model.layers.%d.conv.", i)
	d, s, k := m.d, m.seq, m.cfg.ConvKernel
	bcx := m.Linear(h, p+"in_proj.weight", "", 3*d)
	cw := m.Get(p+"conv.weight", d, 1, k)
	part := func(j int) *mlpackage.Var {
		return m.Op("slice_by_index", map[string]any{"x": bcx, "begin": m.B.Ints(0, 0, int32(j*d)), "end": m.B.Ints(1, int32(s), int32((j+1)*d)),
			"begin_mask": m.Bools(true, true, false), "end_mask": m.Bools(true, true, false)}, 1, s, d)
	}
	bx := m.Op("mul", map[string]any{"x": part(0), "y": part(2)}, 1, s, d)
	var padded *mlpackage.Var
	if m.opts.Cache > 0 {
		win := m.input(fmt.Sprintf("win_conv_%d", i), 1, k-1, d)
		padded = m.Op("concat", map[string]any{"values": []*mlpackage.Var{win, bx}, "axis": m.B.ScalarI32(1), "interleave": m.B.ScalarBool(false)}, 1, s+k-1, d)
		m.output(padded, fmt.Sprintf("win_conv_%d_new", i))
	} else {
		padded = m.Op("pad", map[string]any{"x": bx, "pad": m.B.Ints(int32(k-1), 0, 0, 0), "mode": m.Str("constant"), "constant_val": m.B.ScalarF16(0)}, 1, s+k-1, d)
	}
	var y *mlpackage.Var
	for j := 0; j < k; j++ {
		tap := make([]float32, d)
		for c := 0; c < d; c++ {
			tap[c] = cw[c*k+j]
		}
		wv, err := m.B.F16s(fmt.Sprintf("%sconv.weight.tap%d", p, j), []int{1, 1, d}, tap)
		if err != nil {
			m.Fail("%v", err)
		}
		seg := m.Op("slice_by_index", map[string]any{"x": padded, "begin": m.B.Ints(0, int32(j), 0), "end": m.B.Ints(1, int32(j+s), int32(d)),
			"begin_mask": m.Bools(true, false, true), "end_mask": m.Bools(true, false, true)}, 1, s, d)
		term := m.Op("mul", map[string]any{"x": seg, "y": wv}, 1, s, d)
		if y == nil {
			y = term
		} else {
			y = m.Op("add", map[string]any{"x": y, "y": term}, 1, s, d)
		}
	}
	y = m.Op("mul", map[string]any{"x": part(1), "y": y}, 1, s, d)
	return m.Linear(y, p+"out_proj.weight", "", d)
}

// attention is grouped query attention with QK RMS norms and RoPE, causal.
func (m *builder) attention(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("model.layers.%d.self_attn.", i)
	s, nh, nkv, hd := m.seq, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim
	const lt = "full_attention"
	var cos, sin *mlpackage.Var
	if m.opts.Cache > 0 {
		cos, sin = m.input("rope_cos_"+lt, 1, s, 1, hd), m.input("rope_sin_"+lt, 1, s, 1, hd)
	} else {
		cos, sin = m.RopeTables(lt, m.cfg.RopeInvFreq(), s, hd)
	}
	heads := func(name, norm string, n int) *mlpackage.Var {
		v := m.Linear(h, p+name, "", n*hd)
		v = m.Op("reshape", map[string]any{"x": v, "shape": m.B.Ints(1, int32(s), int32(n), int32(hd))}, 1, s, n, hd)
		if norm != "" {
			v = m.RMSNorm(v, p+norm, hd, m.cfg.NormEps, 1)
			v = m.ApplyRope(v, cos, sin)
		}
		return m.Op("transpose", map[string]any{"x": v, "perm": m.B.Ints(0, 2, 1, 3)}, 1, n, s, hd)
	}
	q := heads("q_proj.weight", "q_layernorm.weight", nh)
	k := heads("k_proj.weight", "k_layernorm.weight", nkv)
	v := heads("v_proj.weight", "", nkv)
	var o *mlpackage.Var
	if m.opts.Cache > 0 {
		m.output(k, fmt.Sprintf("kv_k_%d_new", i))
		m.output(v, fmt.Sprintf("kv_v_%d_new", i))
		t := m.opts.Cache
		if lens, chunked := llm.CacheChunks(t); chunked {
			var ks, vs, masks []*mlpackage.Var
			for j, l := range lens {
				ks = append(ks, m.input(fmt.Sprintf("kv_k_%d_c%d", i, j), 1, nkv, l, hd))
				vs = append(vs, m.input(fmt.Sprintf("kv_v_%d_c%d", i, j), 1, nkv, l, hd))
				masks = append(masks, m.input(fmt.Sprintf("mask_%s_c%d", lt, j), 1, 1, s, l))
			}
			masks = append(masks, m.input("mask_"+lt+"_self", 1, 1, s, s))
			o = m.ChunkedSDPAGrouped(q, append(ks, k), append(vs, v), masks)
		} else {
			cat := func(name string, x *mlpackage.Var) *mlpackage.Var {
				c := m.input(name, 1, nkv, t, hd)
				return m.Op("concat", map[string]any{"values": []*mlpackage.Var{c, x}, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, nkv, t+s, hd)
			}
			k, v = cat(fmt.Sprintf("kv_k_%d", i), k), cat(fmt.Sprintf("kv_v_%d", i), v)
			mask := m.input("mask_"+lt, 1, 1, s, t+s)
			o = m.SDPAGrouped(q, k, v, 1/math.Sqrt(float64(hd)), func() *mlpackage.Var { return mask })
		}
	} else {
		o = m.Attention(q, k, v, "mask_"+lt, 0)
	}
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, s, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(s), int32(nh*hd))}, 1, s, nh*hd)
	return m.Linear(o, p+"out_proj.weight", "", m.d)
}

// StepInputs gives the masks and rotary tables of a step of valid tokens from pos (llm.Cached):
// query t sees the cached positions before pos and the step's tokens up to t.
func (c *Config) StepInputs(pos, valid, q, cache int) map[string][]float32 {
	const lt = "full_attention"
	out := map[string][]float32{}
	mask := make([]float32, q*(cache+q))
	for t := 0; t < q; t++ {
		row := mask[t*(cache+q):]
		for j := 0; j < cache+q; j++ {
			visible := j < pos
			if j >= cache {
				visible = j-cache <= t
			}
			if !visible {
				row[j] = -1e4
			}
		}
	}
	if lens, chunked := llm.CacheChunks(cache); chunked {
		part := func(name string, from, n int) {
			v := make([]float32, 0, q*n)
			for t := 0; t < q; t++ {
				v = append(v, mask[t*(cache+q)+from:t*(cache+q)+from+n]...)
			}
			out[name] = v
		}
		off := 0
		for j, l := range lens {
			part(fmt.Sprintf("mask_%s_c%d", lt, j), off, l)
			off += l
		}
		part("mask_"+lt+"_self", cache, q)
	} else {
		out["mask_"+lt] = mask
	}
	out["rope_cos_"+lt], out["rope_sin_"+lt] = blocks.RopeValues(c.RopeInvFreq(), pos, q, c.HeadDim)
	return out
}
