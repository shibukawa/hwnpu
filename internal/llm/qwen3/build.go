package qwen3

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

type builder struct {
	*blocks.Graph
	cfg  *Config
	opts llm.Options
	seq  int
	d    int
	end  int // one past the last layer built

	ins     map[string]*mlpackage.Var
	inList  []mlpackage.Feature
	outList []*mlpackage.Var

	cos, sin, mask *mlpackage.Var
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

// Build writes the program for the model into b and returns its inputs and outputs, x and y
// first. x is the fp16 token embeddings (1, seq, hidden), the rows of the table as they are, or
// the hidden state entering layer Start; y is the logits (1, seq, vocab) or a hidden state. With
// opts.Cache, the KV cache of each layer (kv_k_N, kv_v_N), the mask and the rotary tables (mask,
// rope_cos, rope_sin) are inputs and the step's keys and values outputs, as llm.Cached
// describes; with opts.Paged as well, every layer's attention runs on the host between the parts
// (llm.Paged); the first part then has no layers (llm.NoLayers) and outputs layer 0's.
func Build(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = "lut4"
	}
	switch {
	case opts.Layers == llm.NoLayers:
		if opts.Start != 0 || opts.Paged <= 0 {
			return nil, nil, fmt.Errorf("qwen3: no layers (NoLayers) is the part before layer 0's attention of a program split at it (Paged)")
		}
		opts.Layers = 0
	case opts.Layers <= 0 || opts.Start+opts.Layers > cfg.NumLayers:
		opts.Layers = cfg.NumLayers - opts.Start
	}
	if opts.Paged > 0 && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("qwen3: a program split at its attention (Paged) needs a KV cache")
	}
	if opts.LastRow && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("qwen3: the logits of the last row alone (LastRow) are of a program with a KV cache")
	}
	m := &builder{Graph: &blocks.Graph{Name: "qwen3", B: b, W: w, Opts: opts}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize,
		end: opts.Start + opts.Layers, ins: map[string]*mlpackage.Var{}}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		// Nothing comes between the embedding lookup and layer 0.
		h := x
		for i := opts.Start; i < m.end; i++ {
			if i == opts.Start && m.paged() {
				h = m.afterAttention(i, h) // the host attended between the parts, before layer 0 too
				continue
			}
			h = m.layer(i, h)
		}
		if m.end < cfg.NumLayers && m.paged() {
			m.beforeAttention(m.end, h) // the host attends next; y is the hidden state entering end
		}
		var y *mlpackage.Var
		if m.end < cfg.NumLayers || opts.NoHead {
			y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		} else {
			if opts.LastRow {
				h = m.LastRow(h, m.input("last", 1, 1, m.seq))
			}
			y = m.head(h)
		}
		outs = []mlpackage.Feature{{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}}
		for _, v := range m.outList {
			outs = append(outs, mlpackage.Feature{Name: v.Name, DType: mlpackage.Float16, Shape: v.Shape})
		}
		ins = append(ins, m.inList...)
	})
	return ins, outs, err
}

// head is the final norm and lm_head (Qwen3ForCausalLM), the embeddings when they are tied.
func (m *builder) head(h *mlpackage.Var) *mlpackage.Var {
	return m.Linear(m.norm(h, "model.norm.weight", m.d), m.cfg.head(), "y", m.cfg.VocabSize)
}

// normScale is how much the norms scale their input down before taking the squares, which
// overflow fp16 past 256: Qwen3-0.6B's residual reaches 8141 from layer 2 on. The scaled input is
// what is normalized, so the norm does not change (with the epsilon scaled to match), and the
// weights stay as they are (Gemma 4's norm). The norm before layer 0 reads the token embeddings
// (Qwen3-0.6B's up to about 0.1), whose squares scaled down would fall under fp16's smallest
// values, so it takes them as they are (scale 1).
const normScale = 64

// norm is Qwen3RMSNorm over the last axis (size n): x normalized, times the named weight.
func (m *builder) norm(x *mlpackage.Var, name string, n int) *mlpackage.Var {
	return m.normScaled(x, name, n, normScale)
}

// normScaled is norm with the input scaled down by scale before its squares.
func (m *builder) normScaled(x *mlpackage.Var, name string, n int, scale float64) *mlpackage.Var {
	msShape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), 1)
	xs := x
	if scale != 1 {
		xs = m.Op("mul", map[string]any{"x": x, "y": m.B.ScalarF16(float32(1 / scale))}, x.Shape...)
	}
	sq := m.Op("mul", map[string]any{"x": xs, "y": xs}, x.Shape...)
	ms := m.Op("reduce_mean", map[string]any{"x": sq, "axes": m.B.Ints(-1), "keep_dims": m.B.ScalarBool(true)}, msShape...)
	r := m.Op("rsqrt", map[string]any{"x": ms, "epsilon": m.B.ScalarF16(float32(m.cfg.NormEps / (scale * scale)))}, msShape...)
	y := m.Op("mul", map[string]any{"x": xs, "y": r}, x.Shape...)
	wshape := make([]int, len(x.Shape))
	for i := range wshape {
		wshape[i] = 1
	}
	wshape[len(wshape)-1] = n
	return m.Op("mul", map[string]any{"x": y, "y": m.F16s(name, wshape, m.Get(name, n))}, x.Shape...)
}

// inputNorm is layer i's input_layernorm: on the token embeddings as they are before layer 0
// (normScale).
func (m *builder) inputNorm(i int, h *mlpackage.Var) *mlpackage.Var {
	scale := float64(normScale)
	if i == 0 {
		scale = 1
	}
	return m.normScaled(h, layerPrefix(i)+"input_layernorm.weight", m.d, scale)
}

// paged reports whether the layers' attention runs on the host between the parts (Options.Paged).
func (m *builder) paged() bool { return m.opts.Paged > 0 && m.opts.Cache > 0 }

// layer is Qwen3DecoderLayer: the attention and the MLP, each after a norm and added to the
// residual.
func (m *builder) layer(i int, h *mlpackage.Var) *mlpackage.Var {
	if m.paged() {
		m.Fail("layer %d's attention runs on the host: a part must start or end at it", i)
	}
	return m.layerRest(i, m.attention(i, m.inputNorm(i, h)), h)
}

// beforeAttention is the start of layer i, whose attention the host runs (llm.Paged): it outputs
// the queries attn_q_N and the step's keys and values kv_k_N_new and kv_v_N_new.
func (m *builder) beforeAttention(i int, h *mlpackage.Var) {
	q, k, v := m.qkv(i, m.inputNorm(i, h))
	m.output(q, fmt.Sprintf("attn_q_%d", i))
	m.output(k, fmt.Sprintf("kv_k_%d_new", i))
	m.output(v, fmt.Sprintf("kv_v_%d_new", i))
}

// afterAttention is the rest of layer i from its attention's output with the tokens first,
// attn_ot_N (1, seq, heads, head_dim), which the host lays out (llm.Paged), with x the hidden
// state entering the layer.
func (m *builder) afterAttention(i int, x *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	o := m.input(fmt.Sprintf("attn_ot_%d", i), 1, m.seq, c.NumHeads, c.HeadDim)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(m.seq), int32(c.NumHeads*c.HeadDim))}, 1, m.seq, c.NumHeads*c.HeadDim)
	return m.layerRest(i, m.Linear(o, layerPrefix(i)+"self_attn.o_proj.weight", "", m.d), x)
}

// layerRest is a layer after its attention (a, o_proj's output), given the hidden state entering
// the layer.
func (m *builder) layerRest(i int, a, residual *mlpackage.Var) *mlpackage.Var {
	h := m.Op("add", map[string]any{"x": residual, "y": a}, residual.Shape...)
	f := m.mlp(i, m.norm(h, layerPrefix(i)+"post_attention_layernorm.weight", m.d))
	return m.Op("add", map[string]any{"x": h, "y": f}, h.Shape...)
}

// mlp is Qwen3MLP: down_proj(silu(gate_proj(x)) * up_proj(x)).
func (m *builder) mlp(i int, h *mlpackage.Var) *mlpackage.Var {
	p := layerPrefix(i) + "mlp."
	f := m.cfg.IntermediateSize
	gate := m.SiLU(m.Linear(h, p+"gate_proj.weight", "", f))
	up := m.Linear(h, p+"up_proj.weight", "", f)
	g := m.Op("mul", map[string]any{"x": gate, "y": up}, gate.Shape...)
	return m.Linear(g, p+"down_proj.weight", "", m.d)
}

// ropeTables returns the cos and sin tables (1, seq, 1, head_dim): constants for positions
// 0..seq-1, or inputs that StepInputs fills with a KV cache.
func (m *builder) ropeTables() (cos, sin *mlpackage.Var) {
	if m.cos == nil {
		hd := m.cfg.HeadDim
		if m.opts.Cache > 0 {
			m.cos, m.sin = m.input("rope_cos", 1, m.seq, 1, hd), m.input("rope_sin", 1, m.seq, 1, hd)
		} else {
			m.cos, m.sin = m.RopeTables("rope", m.cfg.RopeInvFreq(), m.seq, hd)
		}
	}
	return m.cos, m.sin
}

// attnMask returns the additive mask over keys keys: causal without a cache, an input that
// StepInputs fills with one.
func (m *builder) attnMask(keys int) *mlpackage.Var {
	if m.mask == nil {
		if m.opts.Cache > 0 {
			m.mask = m.input("mask", 1, 1, m.seq, keys)
		} else {
			m.mask = m.CausalMask("mask", m.seq, 0)
		}
	}
	return m.mask
}

// qkv is layer i's queries (1, heads, seq, head_dim) and the step's keys and values (1, kv heads,
// seq, head_dim): the queries and keys normed per head (q_norm, k_norm) and turned by RoPE.
func (m *builder) qkv(i int, h *mlpackage.Var) (q, k, v *mlpackage.Var) {
	c := m.cfg
	p := layerPrefix(i) + "self_attn."
	hd, s := c.HeadDim, m.seq
	heads := func(name string, n int) *mlpackage.Var {
		t := m.Linear(h, p+name+"_proj.weight", "", n*hd)
		return m.Op("reshape", map[string]any{"x": t, "shape": m.B.Ints(1, int32(s), int32(n), int32(hd))}, 1, s, n, hd)
	}
	toHeads := func(t *mlpackage.Var) *mlpackage.Var {
		sh := t.Shape
		return m.Op("transpose", map[string]any{"x": t, "perm": m.B.Ints(0, 2, 1, 3)}, sh[0], sh[2], sh[1], sh[3])
	}
	cos, sin := m.ropeTables()
	q = toHeads(m.ApplyRope(m.norm(heads("q", c.NumHeads), p+"q_norm.weight", hd), cos, sin))
	k = toHeads(m.ApplyRope(m.norm(heads("k", c.NumKVHeads), p+"k_norm.weight", hd), cos, sin))
	v = toHeads(heads("v", c.NumKVHeads))
	return q, k, v
}

// attention is Qwen3Attention: queries, keys and values, the cached keys and values before the
// step's with a cache, grouped causal attention scaled by 1/sqrt(head_dim), and o_proj.
func (m *builder) attention(i int, h *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	nh, nkv, hd, s := c.NumHeads, c.NumKVHeads, c.HeadDim, m.seq
	q, k, v := m.qkv(i, h)
	keys := s
	if t := m.opts.Cache; t > 0 {
		keys += t
		m.output(k, fmt.Sprintf("kv_k_%d_new", i))
		m.output(v, fmt.Sprintf("kv_v_%d_new", i))
		cat := func(name string, x *mlpackage.Var) *mlpackage.Var {
			in := m.input(name, 1, nkv, t, hd)
			return m.Op("concat", map[string]any{"values": []*mlpackage.Var{in, x}, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, nkv, t+s, hd)
		}
		k, v = cat(fmt.Sprintf("kv_k_%d", i), k), cat(fmt.Sprintf("kv_v_%d", i), v)
	}
	o := m.SDPAGrouped(q, k, v, c.scoreScale(), func() *mlpackage.Var { return m.attnMask(keys) })
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, s, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(s), int32(nh*hd))}, 1, s, nh*hd)
	return m.Linear(o, layerPrefix(i)+"self_attn.o_proj.weight", "", m.d)
}
