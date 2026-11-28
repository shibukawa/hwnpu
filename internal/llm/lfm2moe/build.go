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
	cfg    *Config
	opts   llm.Options
	seq, d int
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
	paged := func(i int) bool { return opts.Paged > 0 && cfg.IsAttention(i) }
	first, afterAttn, err := blocks.NormalizeRoutedOptions("lfm2moe", &opts, cfg.NumLayers, cfg.Format(), nil, cfg.IsRouted, paged)
	if err != nil {
		return nil, nil, err
	}
	m := &builder{Graph: &blocks.Graph{Name: "lfm2moe", B: b, W: w, Opts: opts}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize}
	return blocks.BuildRouted(m, first, afterAttn, m.seq, m.d, cfg.NumLayers, cfg.VocabSize, cfg.Head())
}

func (m *builder) IsRouted(i int) bool { return m.cfg.IsRouted(i) }
func (m *builder) IsPaged(i int) bool  { return m.opts.Paged > 0 && m.cfg.IsAttention(i) }

func (m *builder) FinalNorm(h *mlpackage.Var) *mlpackage.Var {
	return m.RMSNorm(h, "model.embedding_norm.weight", m.d, m.cfg.NormEps, 1)
}

func (m *builder) BeforeAttention(i int, h *mlpackage.Var) {
	p := fmt.Sprintf("model.layers.%d.", i)
	q, k, v := m.qkv(i, m.RMSNorm(h, p+"operator_norm.weight", m.d, m.cfg.NormEps, 1))
	m.EmitPagedQKV(i, q, k, v)
}

func (m *builder) AfterAttention(i int) *mlpackage.Var {
	return m.attnOutput(i, m.Input(fmt.Sprintf("attn_o_%d", i), 1, m.cfg.NumHeads, m.seq, m.cfg.HeadDim))
}

// MoEHead is the end of a part before layer i's experts: the ffn norm's output moe_x_N and the
// router's logits moe_logits_N (llm.Routed).
func (m *builder) MoEHead(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("model.layers.%d.", i)
	f := m.RMSNorm(h, p+"ffn_norm.weight", m.d, m.cfg.NormEps, 1)
	m.EmitMoERouter(i, f, p+"feed_forward.gate.weight", "", m.cfg.NumExperts)
	return h
}

// Mixer is the first half of layer i: the norm, the convolution or attention, the residual.
func (m *builder) Mixer(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("model.layers.%d.", i)
	n := m.RMSNorm(h, p+"operator_norm.weight", m.d, m.cfg.NormEps, 1)
	if m.cfg.IsAttention(i) {
		n = m.attention(i, n)
	} else {
		n = m.shortConv(i, n)
	}
	return m.Op("add", map[string]any{"x": n, "y": h}, h.Shape...)
}

// DenseFFN is the second half of a dense layer: the norm, the SwiGLU, the residual.
func (m *builder) DenseFFN(i int, h *mlpackage.Var) *mlpackage.Var {
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
		win := m.Input(fmt.Sprintf("win_conv_%d", i), 1, k-1, d)
		padded = m.Op("concat", map[string]any{"values": []*mlpackage.Var{win, bx}, "axis": m.B.ScalarI32(1), "interleave": m.B.ScalarBool(false)}, 1, s+k-1, d)
		m.Output(padded, fmt.Sprintf("win_conv_%d_new", i))
	} else {
		padded = m.Op("pad", map[string]any{"x": bx, "pad": m.B.Ints(int32(k-1), 0, 0, 0), "mode": m.Str("constant"), "constant_val": m.B.ScalarF16(0)}, 1, s+k-1, d)
	}
	var y *mlpackage.Var
	for j := 0; j < k; j++ {
		tap := make([]float32, d)
		for c := 0; c < d; c++ {
			tap[c] = cw[c*k+j]
		}
		wv := m.F16s(fmt.Sprintf("%sconv.weight.tap%d", p, j), []int{1, 1, d}, tap)
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

// qkv is the attention's queries (1, heads, seq, head_dim) and keys and values (1, kv_heads, seq,
// head_dim) from the normed hidden state, with the QK norms and RoPE.
func (m *builder) qkv(i int, h *mlpackage.Var) (q, k, v *mlpackage.Var) {
	p := fmt.Sprintf("model.layers.%d.self_attn.", i)
	s, nh, nkv, hd := m.seq, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim
	const lt = "full_attention"
	var cos, sin *mlpackage.Var
	if m.opts.Cache > 0 {
		cos, sin = m.Input("rope_cos_"+lt, 1, s, 1, hd), m.Input("rope_sin_"+lt, 1, s, 1, hd)
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
	return heads("q_proj.weight", "q_layernorm.weight", nh), heads("k_proj.weight", "k_layernorm.weight", nkv), heads("v_proj.weight", "", nkv)
}

// attnOutput is the output projection of the attention's o (1, heads, seq, head_dim).
func (m *builder) attnOutput(i int, o *mlpackage.Var) *mlpackage.Var {
	s, nh, hd := m.seq, m.cfg.NumHeads, m.cfg.HeadDim
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, s, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(s), int32(nh*hd))}, 1, s, nh*hd)
	return m.Linear(o, fmt.Sprintf("model.layers.%d.self_attn.out_proj.weight", i), "", m.d)
}

// attention is grouped query attention with QK RMS norms and RoPE, causal, over the cache in the
// program.
func (m *builder) attention(i int, h *mlpackage.Var) *mlpackage.Var {
	s, nkv, hd := m.seq, m.cfg.NumKVHeads, m.cfg.HeadDim
	const lt = "full_attention"
	q, k, v := m.qkv(i, h)
	var o *mlpackage.Var
	if m.opts.Cache > 0 {
		m.Output(k, fmt.Sprintf("kv_k_%d_new", i))
		m.Output(v, fmt.Sprintf("kv_v_%d_new", i))
		t := m.opts.Cache
		if lens, chunked := llm.CacheChunks(t); chunked {
			var ks, vs, masks []*mlpackage.Var
			for j, l := range lens {
				ks = append(ks, m.Input(fmt.Sprintf("kv_k_%d_c%d", i, j), 1, nkv, l, hd))
				vs = append(vs, m.Input(fmt.Sprintf("kv_v_%d_c%d", i, j), 1, nkv, l, hd))
				masks = append(masks, m.Input(fmt.Sprintf("mask_%s_c%d", lt, j), 1, 1, s, l))
			}
			masks = append(masks, m.Input("mask_"+lt+"_self", 1, 1, s, s))
			o = m.ChunkedSDPAGrouped(q, append(ks, k), append(vs, v), masks)
		} else {
			cat := func(name string, x *mlpackage.Var) *mlpackage.Var {
				c := m.Input(name, 1, nkv, t, hd)
				return m.Op("concat", map[string]any{"values": []*mlpackage.Var{c, x}, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, nkv, t+s, hd)
			}
			k, v = cat(fmt.Sprintf("kv_k_%d", i), k), cat(fmt.Sprintf("kv_v_%d", i), v)
			mask := m.Input("mask_"+lt, 1, 1, s, t+s)
			o = m.SDPAGrouped(q, k, v, 1/math.Sqrt(float64(hd)), func() *mlpackage.Var { return mask })
		}
	} else {
		o = m.Attention(q, k, v, "mask_"+lt, 0)
	}
	return m.attnOutput(i, o)
}

// StepInputs gives the masks and rotary tables of a step of valid tokens from pos (llm.Cached):
// query t sees the cached positions before pos and the step's tokens up to t.
func (c *Config) StepInputs(pos, valid, q, cache int) map[string][]float32 {
	const lt = "full_attention"
	out := map[string][]float32{}
	if cache != pagedCache {
		blocks.PutStepMask(out, lt, pos, q, cache, 0)
	}
	out["rope_cos_"+lt], out["rope_sin_"+lt] = blocks.RopeValues(c.RopeInvFreq(), pos, q, c.HeadDim)
	return out
}

// pagedCache marks StepInputs' call from PagedStepInputs.
const pagedCache = -1

// PagedStepInputs is StepInputs for a program split at its attention (llm.Paged): the rotary
// tables, the masks being the host's attention's.
func (c *Config) PagedStepInputs(pos, valid, q, cache int) map[string][]float32 {
	return c.StepInputs(pos, valid, q, pagedCache)
}
