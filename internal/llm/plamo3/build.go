package plamo3

import (
	"fmt"
	"slices"

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

	rope  map[bool][2]*mlpackage.Var // cos and sin of the full and the sliding layers
	masks map[bool]*mlpackage.Var
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
// opts.Cache, the KV cache of each layer (kv_k_N, kv_v_N; a ring of the window for a sliding layer
// when that is shorter) and the masks and rotary tables of the sliding and full layers
// (mask_sliding, rope_cos_full, ...) are inputs and the step's keys and values outputs, as
// llm.Cached describes; with opts.Paged as well, the full layers' attention runs on the host
// between the parts (llm.Paged).
func Build(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = "lut4"
	}
	if opts.Layers <= 0 || opts.Start+opts.Layers > cfg.NumLayers {
		opts.Layers = cfg.NumLayers - opts.Start
	}
	if opts.Paged > 0 && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("plamo3: a program split at its attention (Paged) needs a KV cache")
	}
	m := &builder{Graph: &blocks.Graph{Name: "plamo3", B: b, W: w, Opts: opts}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize,
		end: opts.Start + opts.Layers, ins: map[string]*mlpackage.Var{}, rope: map[bool][2]*mlpackage.Var{}, masks: map[bool]*mlpackage.Var{}}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		// Nothing comes between the embedding lookup and layer 0 (no scale_embedding).
		h := x
		for i := opts.Start; i < m.end; i++ {
			if i == opts.Start && i > 0 && m.paged(i) {
				h = m.afterAttention(i, h) // the host attended between the parts
				continue
			}
			h = m.layer(i, h)
		}
		if m.end < cfg.NumLayers && m.paged(m.end) {
			m.beforeAttention(m.end, h) // the host attends next; y is the hidden state entering end
		}
		var y *mlpackage.Var
		if m.end < cfg.NumLayers || opts.NoHead {
			y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		} else {
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

// head is the final norm and lm_head (Plamo3ForCausalLM), cut to the vocabulary when lm_head
// has rows past it (the vocabulary padded to 16).
func (m *builder) head(h *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	h = m.norm(h, "model.norm.weight", m.d, offsetPre)
	if c.headRows() == c.VocabSize || c.TieWordEmbed {
		return m.Linear(h, c.head(), "y", c.VocabSize)
	}
	y := m.LinearRows(h, c.head(), 0, c.VocabSize)
	return m.B.NamedOp("identity", "y", map[string]any{"x": y}, mlpackage.Float16, y.Shape...)
}

// normScale is how much the norms scale their input down before taking the squares, which
// overflow fp16 past 256. The scaled input is what is normalized, so the norm does not change
// (with the epsilon scaled to match), and the weights stay as they are (Gemma 4's norm).
const normScale = 64

// norm is Plamo3RMSNorm over the last axis (size n): x normalized, times offset + the named
// weight.
func (m *builder) norm(x *mlpackage.Var, name string, n int, offset float32) *mlpackage.Var {
	msShape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), 1)
	xs := m.Op("mul", map[string]any{"x": x, "y": m.B.ScalarF16(1.0 / normScale)}, x.Shape...)
	sq := m.Op("mul", map[string]any{"x": xs, "y": xs}, x.Shape...)
	ms := m.Op("reduce_mean", map[string]any{"x": sq, "axes": m.B.Ints(-1), "keep_dims": m.B.ScalarBool(true)}, msShape...)
	r := m.Op("rsqrt", map[string]any{"x": ms, "epsilon": m.B.ScalarF16(float32(m.cfg.NormEps / (normScale * normScale)))}, msShape...)
	y := m.Op("mul", map[string]any{"x": xs, "y": r}, x.Shape...)
	w := slices.Clone(m.Get(name, n))
	for i := range w {
		w[i] += offset
	}
	wshape := make([]int, len(x.Shape))
	for i := range wshape {
		wshape[i] = 1
	}
	wshape[len(wshape)-1] = n
	return m.Op("mul", map[string]any{"x": y, "y": m.F16s(name, wshape, w)}, x.Shape...)
}

// layer is Plamo3DecoderLayer: the attention and the MLP, each between two norms and added to the
// residual.
func (m *builder) layer(i int, h *mlpackage.Var) *mlpackage.Var {
	if m.paged(i) {
		m.Fail("layer %d's attention runs on the host: a part must start or end at it", i)
	}
	return m.layerRest(i, m.attention(i, m.norm(h, layerPrefix(i)+"pre_mixer_norm.weight", m.d, offsetPre)), h)
}

// paged reports whether layer i's attention runs on the host between the parts (Options.Paged).
func (m *builder) paged(i int) bool {
	return m.opts.Paged > 0 && m.opts.Cache > 0 && m.cfg.full(i)
}

// beforeAttention is the start of layer i, whose attention the host runs (llm.Paged): it outputs
// the queries attn_q_N and the step's keys and values kv_k_N_new and kv_v_N_new.
func (m *builder) beforeAttention(i int, h *mlpackage.Var) {
	h = m.norm(h, layerPrefix(i)+"pre_mixer_norm.weight", m.d, offsetPre)
	q, k, v := m.qkv(i, h)
	m.output(q, fmt.Sprintf("attn_q_%d", i))
	m.output(k, fmt.Sprintf("kv_k_%d_new", i))
	m.output(v, fmt.Sprintf("kv_v_%d_new", i))
}

// afterAttention is the rest of layer i from its attention's output with the tokens first,
// attn_ot_N (1, seq, heads, head_dim), which the host lays out (llm.Paged), with x the hidden
// state entering the layer. Taking attn_o_N (heads first) and transposing it, Core ML put the
// transpose that begins the last part (layer 23 and the head) on the CPU on an M3, though not the
// same transpose at the start of the parts before.
func (m *builder) afterAttention(i int, x *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	o := m.input(fmt.Sprintf("attn_ot_%d", i), 1, m.seq, c.NumHeads, c.HeadDim)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(m.seq), int32(c.NumHeads*c.HeadDim))}, 1, m.seq, c.NumHeads*c.HeadDim)
	return m.layerRest(i, m.Linear(o, layerPrefix(i)+"mixer.o_proj.weight", "", m.d), x)
}

// layerRest is a layer after its attention (a, before post_mixer_norm), given the hidden state
// entering the layer.
func (m *builder) layerRest(i int, a, residual *mlpackage.Var) *mlpackage.Var {
	p := layerPrefix(i)
	a = m.norm(a, p+"post_mixer_norm.weight", m.d, offsetPostAttn)
	h := m.Op("add", map[string]any{"x": residual, "y": a}, residual.Shape...)
	f := m.mlp(i, m.norm(h, p+"pre_mlp_norm.weight", m.d, offsetPre))
	f = m.norm(f, p+"post_mlp_norm.weight", m.d, offsetPostMLP)
	return m.Op("add", map[string]any{"x": h, "y": f}, h.Shape...)
}

// mlp is MLP: gate_up_proj, whose first half is the gate and second half the up projection
// (_swiglu), as two linear layers, then down_proj.
func (m *builder) mlp(i int, h *mlpackage.Var) *mlpackage.Var {
	p := layerPrefix(i) + "mlp."
	f := m.cfg.IntermediateSize
	gate := m.LinearRows(h, p+"gate_up_proj.weight", 0, f)
	up := m.LinearRows(h, p+"gate_up_proj.weight", f, 2*f)
	gate = m.SiLU(gate)
	g := m.Op("mul", map[string]any{"x": gate, "y": up}, gate.Shape...)
	return m.Linear(g, p+"down_proj.weight", "", m.d)
}

// ropeTables returns the cos and sin tables (1, seq, 1, head_dim) of the full or the sliding
// layers: constants for positions 0..seq-1, or inputs that StepInputs fills with a KV cache.
// YaRN's concentration is not in them: the attention scales the scores by its square
// (Config.scoreScale), which keeps the cached keys free of it.
func (m *builder) ropeTables(full bool) (cos, sin *mlpackage.Var) {
	if t, ok := m.rope[full]; ok {
		return t[0], t[1]
	}
	lt, hd := layerType(full), m.cfg.HeadDim
	if m.opts.Cache > 0 {
		cos, sin = m.input("rope_cos_"+lt, 1, m.seq, 1, hd), m.input("rope_sin_"+lt, 1, m.seq, 1, hd)
	} else {
		cos, sin = m.RopeTables(lt, m.cfg.RopeInvFreq(full), m.seq, hd)
	}
	m.rope[full] = [2]*mlpackage.Var{cos, sin}
	return cos, sin
}

// mask returns the additive mask of the full or the sliding layers over keys keys: causal (and
// windowed for a sliding layer) without a cache, an input that StepInputs fills with one.
func (m *builder) mask(full bool, keys int) *mlpackage.Var {
	if v, ok := m.masks[full]; ok {
		return v
	}
	var v *mlpackage.Var
	name := "mask_" + layerType(full)
	if m.opts.Cache > 0 {
		v = m.input(name, 1, 1, m.seq, keys)
	} else {
		window := 0
		if !full {
			window = m.cfg.Window
		}
		v = m.CausalMask(name, m.seq, window)
	}
	m.masks[full] = v
	return v
}

// qkv is layer i's queries (1, heads, seq, head_dim) and the step's keys and values (1, kv heads,
// seq, head_dim): the rows of qkv_proj for each, the queries and keys normed per head and turned
// by RoPE.
func (m *builder) qkv(i int, h *mlpackage.Var) (q, k, v *mlpackage.Var) {
	c := m.cfg
	p := layerPrefix(i) + "mixer."
	nh, nkv, hd, s := c.NumHeads, c.NumKVHeads, c.HeadDim, m.seq
	heads := func(begin, n int) *mlpackage.Var {
		t := m.LinearRows(h, p+"qkv_proj.weight", begin, begin+n*hd)
		return m.Op("reshape", map[string]any{"x": t, "shape": m.B.Ints(1, int32(s), int32(n), int32(hd))}, 1, s, n, hd)
	}
	toHeads := func(t *mlpackage.Var) *mlpackage.Var {
		sh := t.Shape
		return m.Op("transpose", map[string]any{"x": t, "perm": m.B.Ints(0, 2, 1, 3)}, sh[0], sh[2], sh[1], sh[3])
	}
	cos, sin := m.ropeTables(c.full(i))
	q = toHeads(m.ApplyRope(m.norm(heads(0, nh), p+"q_norm.weight", hd, offsetPre), cos, sin))
	k = toHeads(m.ApplyRope(m.norm(heads(nh*hd, nkv), p+"k_norm.weight", hd, offsetPre), cos, sin))
	v = toHeads(heads((nh+nkv)*hd, nkv))
	return q, k, v
}

// attention is Attention: queries, keys and values, the cached keys and values before the step's
// with a cache, grouped attention scaled by 1/sqrt(head_dim) (times YaRN's concentration squared
// on a full layer), causal and on a sliding layer windowed, and o_proj.
func (m *builder) attention(i int, h *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	full := c.full(i)
	nkv, hd, s := c.NumKVHeads, c.HeadDim, m.seq
	q, k, v := m.qkv(i, h)
	keys := s
	if m.opts.Cache > 0 {
		t := c.cacheLen(full, m.opts.Cache)
		keys += t
		m.output(k, fmt.Sprintf("kv_k_%d_new", i))
		m.output(v, fmt.Sprintf("kv_v_%d_new", i))
		cat := func(name string, x *mlpackage.Var) *mlpackage.Var {
			in := m.input(name, 1, nkv, t, hd)
			return m.Op("concat", map[string]any{"values": []*mlpackage.Var{in, x}, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, nkv, t+s, hd)
		}
		k, v = cat(fmt.Sprintf("kv_k_%d", i), k), cat(fmt.Sprintf("kv_v_%d", i), v)
	}
	o := m.SDPAGrouped(q, k, v, c.scoreScale(full), func() *mlpackage.Var { return m.mask(full, keys) })
	return m.attnOutput(i, o)
}

// attnOutput is o_proj of the attention's output o (1, heads, seq, head_dim).
func (m *builder) attnOutput(i int, o *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	nh, hd := c.NumHeads, c.HeadDim
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, m.seq, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(m.seq), int32(nh*hd))}, 1, m.seq, nh*hd)
	return m.Linear(o, layerPrefix(i)+"mixer.o_proj.weight", "", m.d)
}
