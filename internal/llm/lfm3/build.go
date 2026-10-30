package lfm3

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Options select what to build: the common options and LFM3's own settings.
type Options struct {
	llm.Options
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
	// HiddenOut makes a program with a KV cache output h, the final hidden state (1, seq,
	// hidden), besides the logits y: LFM3-Audio's speech output samples from it
	// (internal/llm/decode reads it as Session.Hidden).
	HiddenOut bool
}

type builder struct {
	*blocks.Graph
	cfg  *Config
	opts Options
	seq  int
	d    int
	// With a KV cache (opts.Cache): the inputs and outputs besides x and y.
	ins     map[string]*mlpackage.Var
	inList  []mlpackage.Feature
	outList []*mlpackage.Var
	// padBias is the additive padding mask of a d1 model's bidirectional layers, made once.
	padBias *mlpackage.Var
}

func newBuilder(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts Options) *builder {
	return &builder{Graph: &blocks.Graph{Name: "lfm3", B: b, W: w, Opts: opts.Options}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize, ins: map[string]*mlpackage.Var{}}
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
// embeddings of shape (1, seq, hidden): the embedding lookup stays on the host because Core ML
// runs gather on the CPU. The output is y: logits (1, seq, vocab) or a hidden state (1, seq,
// hidden). It returns the input and output descriptions. A program with a KV cache has more of
// both; build it with BuildCached.
func Build(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts Options) (in, out mlpackage.Feature, err error) {
	if opts.Cache > 0 {
		return in, out, fmt.Errorf("lfm3: a program with a KV cache has more inputs and outputs; use BuildCached")
	}
	ins, outs, err := BuildCached(b, cfg, w, opts)
	if err != nil {
		return in, out, err
	}
	return ins[0], outs[0], nil
}

// BuildCached is Build for any Options, returning every input and output: x and y first, then,
// with opts.Cache, the KV cache of each attention layer (kv_k_N, kv_v_N; a sliding layer's is a
// ring of SlidingWindow when that is shorter), the convolution window of each conv layer
// (win_conv_N), and the masks and rotary tables of each attention type (mask_TYPE, rope_cos_TYPE,
// rope_sin_TYPE) that StepInputs fills, as llm.Cached describes.
func BuildCached(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts Options) (ins, outs []mlpackage.Feature, err error) {
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
		return nil, nil, fmt.Errorf("lfm3: PostScale needs sandwich norms to absorb the scale")
	}
	if opts.Cache > 0 && opts.ConvMode != "taps" {
		return nil, nil, fmt.Errorf("lfm3: a KV cache needs conv mode taps")
	}
	if cfg.D1 && opts.Cache > 0 {
		return nil, nil, fmt.Errorf("lfm3: a d1 model reads the whole prompt at once; it has no KV cache")
	}
	if cfg.D1 {
		opts.Hidden = true
	}
	m := newBuilder(b, cfg, w, opts)
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		h := x
		for i := opts.Start; i < opts.Start+opts.Layers; i++ {
			h = m.layer(i, h)
		}
		var y *mlpackage.Var
		switch {
		case opts.Start+opts.Layers < cfg.NumLayers || opts.NoHead:
			y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		default:
			h = m.rmsNorm(h, cfg.prefix()+"embedding_norm.weight", m.d)
			if opts.Hidden {
				y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
			} else {
				y = m.Linear(h, cfg.Head(), "y", cfg.VocabSize)
				if opts.HiddenOut && opts.Cache > 0 {
					m.output(h, "h")
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

// rmsNorm is blocks.RMSNorm with LFM3's epsilon; inScale is the producer's PostScale, if any.
func (m *builder) rmsNorm(x *mlpackage.Var, name string, n int, inScale ...float32) *mlpackage.Var {
	k := float32(1)
	if len(inScale) > 0 {
		k = inScale[0]
	}
	return m.RMSNorm(x, name, n, m.cfg.NormEps, k)
}

func (m *builder) layer(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("%slayers.%d.", m.cfg.prefix(), i)
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
	h = m.Op("add", map[string]any{"x": h, "y": residual}, h.Shape...)
	residual = h
	f := m.rmsNorm(h, p+"ffn_norm.weight", m.d)
	f = m.SwiGLU(f, p+"feed_forward.w1.weight", p+"feed_forward.w3.weight", p+"feed_forward.w2.weight", m.cfg.IntermediateSize, m.d, m.opts.PostScale)
	if m.cfg.UseSandwichNorm {
		f = m.rmsNorm(f, p+"post_ffn_norm.weight", m.d, m.opts.PostScale)
	}
	return m.Op("add", map[string]any{"x": f, "y": residual}, f.Shape...)
}

// shortConv is the gated short convolution: in_proj splits into B, C and x; the depthwise causal
// convolution of width K runs over B*x along the sequence, and C gates the result. In "taps"
// mode it stays in the (1, seq, d) layout: the convolution is K shifted multiply adds over a
// left padded sequence, with no transposes.
func (m *builder) shortConv(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("%slayers.%d.conv.", m.cfg.prefix(), i)
	d, s, k := m.d, m.seq, m.cfg.ConvKernel
	bcx := m.Linear(h, p+"in_proj.weight", "", 3*d) // (1, s, 3d)
	cw := m.Get(p+"conv.weight", d, 1, k)
	var y, cg *mlpackage.Var
	switch m.opts.ConvMode {
	case "conv":
		t := m.Op("transpose", map[string]any{"x": bcx, "perm": m.B.Ints(0, 2, 1)}, 1, 3*d, s) // (1, 3d, s)
		part := func(j int) *mlpackage.Var {
			return m.Op("slice_by_index", map[string]any{"x": t, "begin": m.B.Ints(0, int32(j*d), 0), "end": m.B.Ints(1, int32((j+1)*d), int32(s)),
				"begin_mask": m.Bools(true, false, true), "end_mask": m.Bools(true, false, true)}, 1, d, s)
		}
		bg, xg := part(0), part(2)
		cg = part(1)
		bx := m.Op("mul", map[string]any{"x": bg, "y": xg}, 1, d, s)
		padded := m.Op("pad", map[string]any{"x": bx, "pad": m.B.Ints(int32(k-1), 0), "mode": m.Str("constant"), "constant_val": m.B.ScalarF16(0)}, 1, d, s+k-1)
		wv, err := m.B.F16s(p+"conv.weight", []int{d, 1, k}, cw)
		if err != nil {
			m.Fail("%v", err)
		}
		conv := map[string]any{"x": padded, "weight": wv, "strides": m.B.Ints(1), "pad_type": m.Str("valid"), "pad": m.B.Ints(0, 0),
			"dilations": m.B.Ints(1), "groups": m.B.ScalarI32(int32(d))}
		if m.cfg.ConvBias {
			bias, err := m.B.F16s(p+"conv.bias", []int{d}, m.Get(p+"conv.bias", d))
			if err != nil {
				m.Fail("%v", err)
			}
			conv["bias"] = bias
		}
		y = m.Op("conv", conv, 1, d, s)
		y = m.Op("mul", map[string]any{"x": cg, "y": y}, 1, d, s)
		y = m.Op("transpose", map[string]any{"x": y, "perm": m.B.Ints(0, 2, 1)}, 1, s, d)
	case "taps":
		part := func(j int) *mlpackage.Var {
			return m.Op("slice_by_index", map[string]any{"x": bcx, "begin": m.B.Ints(0, 0, int32(j*d)), "end": m.B.Ints(1, int32(s), int32((j+1)*d)),
				"begin_mask": m.Bools(true, true, false), "end_mask": m.Bools(true, true, false)}, 1, s, d)
		}
		bg, xg := part(0), part(2)
		cg = part(1)
		bx := m.Op("mul", map[string]any{"x": bg, "y": xg}, 1, s, d)
		var padded *mlpackage.Var
		if m.opts.Cache > 0 {
			// the last k-1 rows of B*x before the step instead of zeros
			win := m.input(fmt.Sprintf("win_conv_%d", i), 1, k-1, d)
			padded = m.Op("concat", map[string]any{"values": []*mlpackage.Var{win, bx}, "axis": m.B.ScalarI32(1), "interleave": m.B.ScalarBool(false)}, 1, s+k-1, d)
			m.output(padded, fmt.Sprintf("win_conv_%d_new", i))
		} else {
			// pad the sequence axis on the left; pad covers the last two axes
			padded = m.Op("pad", map[string]any{"x": bx, "pad": m.B.Ints(int32(k-1), 0, 0, 0), "mode": m.Str("constant"), "constant_val": m.B.ScalarF16(0)}, 1, s+k-1, d)
		}
		// y[t] = sum_j w[j] * bx[t + j - (k-1)]: tap j reads the padded sequence shifted by j
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
		if m.cfg.ConvBias {
			bias, err := m.B.F16s(p+"conv.bias", []int{1, 1, d}, m.Get(p+"conv.bias", d))
			if err != nil {
				m.Fail("%v", err)
			}
			y = m.Op("add", map[string]any{"x": y, "y": bias}, 1, s, d)
		}
		y = m.Op("mul", map[string]any{"x": cg, "y": y}, 1, s, d)
	default:
		m.Fail("unknown conv mode %q", m.opts.ConvMode)
	}
	return m.LinearS(y, p+"out_proj.weight", "", d, m.opts.PostScale)
}

// attention is grouped query attention with QK RMS norm, partial RoPE, a causal (and for sliding
// layers windowed) mask and a sigmoid gate per head. A d1 model's full attention layers are
// bidirectional instead: only the padding is masked, by the program input mask (1, seq), 1 for
// the prompt's tokens and 0 for the padding after them.
func (m *builder) attention(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("%slayers.%d.self_attn.", m.cfg.prefix(), i)
	s, nh, nkv, hd := m.seq, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim
	layerType := m.cfg.LayerTypes[i]
	var cos, sin *mlpackage.Var
	if m.opts.Cache > 0 {
		cos, sin = m.input("rope_cos_"+layerType, 1, s, 1, hd), m.input("rope_sin_"+layerType, 1, s, 1, hd)
	} else {
		cos, sin = m.RopeTables(layerType, m.cfg.RopeInvFreq(layerType), s, hd)
	}

	q := m.Linear(h, p+"q_proj.weight", "", nh*hd)
	q = m.Op("reshape", map[string]any{"x": q, "shape": m.B.Ints(1, int32(s), int32(nh), int32(hd))}, 1, s, nh, hd)
	q = m.rmsNorm(q, p+"q_norm.weight", hd)
	q = m.ApplyRope(q, cos, sin)
	q = m.Op("transpose", map[string]any{"x": q, "perm": m.B.Ints(0, 2, 1, 3)}, 1, nh, s, hd)

	k := m.Linear(h, p+"k_proj.weight", "", nkv*hd)
	k = m.Op("reshape", map[string]any{"x": k, "shape": m.B.Ints(1, int32(s), int32(nkv), int32(hd))}, 1, s, nkv, hd)
	k = m.rmsNorm(k, p+"k_norm.weight", hd)
	k = m.ApplyRope(k, cos, sin)
	k = m.Op("transpose", map[string]any{"x": k, "perm": m.B.Ints(0, 2, 1, 3)}, 1, nkv, s, hd)

	v := m.Linear(h, p+"v_proj.weight", "", nkv*hd)
	v = m.Op("reshape", map[string]any{"x": v, "shape": m.B.Ints(1, int32(s), int32(nkv), int32(hd))}, 1, s, nkv, hd)
	v = m.Op("transpose", map[string]any{"x": v, "perm": m.B.Ints(0, 2, 1, 3)}, 1, nkv, s, hd)

	var o *mlpackage.Var
	if m.opts.Cache > 0 {
		// the cached keys and values, then the step's own. A sliding layer only ever sees the
		// last SlidingWindow positions, so its cache is a ring of that many (cacheLen).
		t := m.cfg.cacheLen(layerType, m.opts.Cache)
		m.output(k, fmt.Sprintf("kv_k_%d_new", i))
		m.output(v, fmt.Sprintf("kv_v_%d_new", i))
		if lens, chunked := llm.CacheChunks(t); chunked {
			// a cache over the Neural Engine's dimension limit: in chunks, each attended on
			// its own, then the step's tokens with a mask of their own
			var ks, vs, masks []*mlpackage.Var
			for j, l := range lens {
				ks = append(ks, m.input(fmt.Sprintf("kv_k_%d_c%d", i, j), 1, nkv, l, hd))
				vs = append(vs, m.input(fmt.Sprintf("kv_v_%d_c%d", i, j), 1, nkv, l, hd))
				masks = append(masks, m.input(fmt.Sprintf("mask_%s_c%d", layerType, j), 1, 1, s, l))
			}
			masks = append(masks, m.input("mask_"+layerType+"_self", 1, 1, s, s))
			o = m.ChunkedSDPA(q, append(ks, k), append(vs, v), masks)
		} else {
			cat := func(name string, x *mlpackage.Var) *mlpackage.Var {
				c := m.input(name, 1, nkv, t, hd)
				return m.Op("concat", map[string]any{"values": []*mlpackage.Var{c, x}, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, nkv, t+s, hd)
			}
			k, v = cat(fmt.Sprintf("kv_k_%d", i), k), cat(fmt.Sprintf("kv_v_%d", i), v)
			mask := m.input("mask_"+layerType, 1, 1, s, t+s)
			o = m.SDPA(q, k, v, func() *mlpackage.Var { return mask })
		}
	} else if m.cfg.D1 && layerType == "full_attention" {
		if m.padBias == nil {
			m.padBias = m.MaskBias(m.input("mask", 1, s))
		}
		o = m.SDPA(q, k, v, func() *mlpackage.Var { return m.padBias })
	} else {
		window := 0
		if layerType == "sliding_attention" {
			window = m.cfg.SlidingWindow
		}
		o = m.Attention(q, k, v, "mask_"+layerType, window)
	}
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, s, nh, hd)

	gate := m.Linear(h, p+"gate_proj.weight", "", nh)
	gate = m.Op("sigmoid", map[string]any{"x": gate}, 1, s, nh)
	gate = m.Op("reshape", map[string]any{"x": gate, "shape": m.B.Ints(1, int32(s), int32(nh), 1)}, 1, s, nh, 1)
	o = m.Op("mul", map[string]any{"x": o, "y": gate}, 1, s, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(s), int32(nh*hd))}, 1, s, nh*hd)
	return m.LinearS(o, p+"o_proj.weight", "", m.d, m.opts.PostScale)
}

// cacheLen is the length of the KV cache of a layer of the given type in a program built with
// Options.Cache: the whole cache, or for a sliding layer the window when that is smaller. A cache
// shorter than the longest is a ring the host writes at position mod length (internal/llm/decode).
func (c *Config) cacheLen(layerType string, cache int) int {
	if layerType == "sliding_attention" && c.SlidingWindow > 0 && c.SlidingWindow < cache {
		return c.SlidingWindow
	}
	return cache
}

// StepInputs gives the masks and rotary tables of a step of valid tokens from pos (llm.Cached).
// Query t (position pos+t) sees cached position j < pos and step token u <= t, and in a sliding
// layer only those less than the window behind it. A sliding layer's cache is a ring when the
// window is shorter than the cache: slot j holds the latest position before pos congruent to j.
// A cache in chunks (llm.CacheChunks) has the mask in parts: mask_TYPE_cJ over each chunk and
// mask_TYPE_self over the step's tokens.
func (c *Config) StepInputs(pos, valid, q, cache int) map[string][]float32 {
	out := map[string][]float32{}
	for _, lt := range c.LayerTypes {
		if _, done := out["rope_cos_"+lt]; done || lt == "conv" {
			continue
		}
		window := 0
		if lt == "sliding_attention" {
			window = c.SlidingWindow
		}
		keys := c.cacheLen(lt, cache)
		ring := keys < cache
		mask := make([]float32, q*(keys+q))
		for t := 0; t < q; t++ {
			p := pos + t
			row := mask[t*(keys+q):]
			for j := 0; j < keys+q; j++ {
				var key int
				var visible bool
				switch {
				case j >= keys: // the step's own tokens
					key = pos + j - keys
					visible = j-keys <= t
				case ring: // the latest position written to slot j, if any
					key = pos - 1 - ((pos-1-j)%keys+keys)%keys
					visible = key >= 0
				default:
					key = j
					visible = j < pos
				}
				if window > 0 && p-key >= window {
					visible = false
				}
				if !visible {
					row[j] = -1e4
				}
			}
		}
		if lens, chunked := llm.CacheChunks(keys); chunked {
			part := func(name string, from, n int) {
				v := make([]float32, 0, q*n)
				for t := 0; t < q; t++ {
					v = append(v, mask[t*(keys+q)+from:t*(keys+q)+from+n]...)
				}
				out[name] = v
			}
			off := 0
			for j, l := range lens {
				part(fmt.Sprintf("mask_%s_c%d", lt, j), off, l)
				off += l
			}
			part("mask_"+lt+"_self", keys, q)
		} else {
			out["mask_"+lt] = mask
		}
		out["rope_cos_"+lt], out["rope_sin_"+lt] = blocks.RopeValues(c.RopeInvFreq(lt), pos, q, c.HeadDim)
	}
	return out
}
