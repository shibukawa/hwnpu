package ouro

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
	// consts are the weights already in the program by name: the passes of a layer share them
	consts map[string]*mlpackage.Var

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
// first. The layers are those of every pass (Config.Layers). x is the fp16 token embeddings (1,
// seq, hidden), the rows of the table as they are, or the hidden state entering layer Start; y is
// the logits (1, seq, vocab) or a hidden state. With opts.Cache, the KV cache of each layer of each
// pass (kv_k_N, kv_v_N, N of Layers), the mask and the rotary tables (mask, rope_cos, rope_sin) are
// inputs and the step's keys and values outputs, as llm.Cached describes; with opts.Paged as well,
// every layer's attention runs on the host between the parts (llm.Paged); the first part then has
// no layers (llm.NoLayers) and outputs layer 0's.
func Build(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = "lut4"
	}
	n := cfg.Layers()
	switch {
	case opts.Layers == llm.NoLayers:
		if opts.Start != 0 || opts.Paged <= 0 {
			return nil, nil, fmt.Errorf("ouro: no layers (NoLayers) is the part before layer 0's attention of a program split at it (Paged)")
		}
		opts.Layers = 0
	case opts.Layers <= 0 || opts.Start+opts.Layers > n:
		opts.Layers = n - opts.Start
	}
	if opts.Paged > 0 && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("ouro: a program split at its attention (Paged) needs a KV cache")
	}
	if opts.LastRow && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("ouro: the logits of the last row alone (LastRow) are of a program with a KV cache")
	}
	m := &builder{Graph: &blocks.Graph{Name: "ouro", B: b, W: w, Opts: opts}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize,
		end: opts.Start + opts.Layers, ins: map[string]*mlpackage.Var{}, consts: map[string]*mlpackage.Var{}}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		// Nothing comes between the embedding lookup and layer 0.
		h := x
		for i := opts.Start; i < m.end; i++ {
			if i == opts.Start && m.paged() {
				h = m.afterAttention(i, h) // the host attended between the parts, before layer 0 too
			} else {
				h = m.layer(i, h)
			}
			if cfg.endsPass(i) && i < n-1 {
				h = m.passNorm(h) // the next pass starts from the normed state
			}
		}
		if m.end < n && m.paged() {
			m.beforeAttention(m.end, h) // the host attends next; y is the hidden state entering end
		}
		var y *mlpackage.Var
		if m.end < n || opts.NoHead {
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

// head is the final norm of the last pass and lm_head (OuroForCausalLM), the embeddings when they
// are tied.
func (m *builder) head(h *mlpackage.Var) *mlpackage.Var {
	return m.Linear(m.passNorm(h), m.cfg.head(), "y", m.cfg.VocabSize)
}

// passNorm is model.norm, which ends every pass.
func (m *builder) passNorm(h *mlpackage.Var) *mlpackage.Var {
	return m.norm(h, "model.norm.weight", residualScale)
}

// linear is nn.Linear without a bias, its weight held once in the program (consts): the passes
// of a layer read the same constant. A weight in lutNr (blocks.RowScaledBits) is not shared.
func (m *builder) linear(x *mlpackage.Var, name string, outDim int) *mlpackage.Var {
	if blocks.RowScaledBits(m.Opts.FormatOf(name)) != 0 {
		return m.Linear(x, name, "", outDim)
	}
	in := x.Shape[len(x.Shape)-1]
	w, ok := m.consts[name]
	if !ok {
		w = m.Weight(name, outDim, in, 1)
		m.consts[name] = w
	}
	shape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), outDim)
	return m.Op("linear", map[string]any{"x": x, "weight": w}, shape...)
}

// slice is blocks.MatrixSlice held once in the program (consts).
func (m *builder) slice(name string, rows, cols, r0, r1, c0, c1 int) *mlpackage.Var {
	key := fmt.Sprintf("%s[%d:%d,%d:%d]", name, r0, r1, c0, c1)
	w, ok := m.consts[key]
	if !ok {
		w = m.MatrixSlice(name, rows, cols, r0, r1, c0, c1)
		m.consts[key] = w
	}
	return w
}

// residualScale is how much the norms of the residual (input_layernorm, post_attention_layernorm
// and model.norm) scale their input down before taking the squares, which overflow fp16 past 256,
// and their mean past a root mean square of 256: Ouro-1.4B-Thinking's residual reaches 212, its
// rows' root mean square 4.7. The scaled input is what is normalized, so the norm does not change
// (with the epsilon scaled to match), and the weights stay as they are (Gemma 4's norm). Not 64 as
// Qwen3's: the rows of a root mean square of 0.35 (the normed state starting a pass) would be
// squared under fp16's smallest normal value. The norms after the attention and the MLP
// (input_layernorm_2, post_attention_layernorm_2) read at most 54 and 25 and rows down to a root
// mean square of 0.0002, so they take their input as it is, as does the norm before the first
// layer, which reads the token embeddings (up to 0.1).
const residualScale = 4

// norm is OuroRMSNorm over the last axis: x normalized, times the named weight (held once), x
// scaled down by scale before its squares.
func (m *builder) norm(x *mlpackage.Var, name string, scale float64) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	msShape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), 1)
	xs := x
	if scale != 1 {
		xs = m.Op("mul", map[string]any{"x": x, "y": m.B.ScalarF16(float32(1 / scale))}, x.Shape...)
	}
	sq := m.Op("mul", map[string]any{"x": xs, "y": xs}, x.Shape...)
	ms := m.Op("reduce_mean", map[string]any{"x": sq, "axes": m.B.Ints(-1), "keep_dims": m.B.ScalarBool(true)}, msShape...)
	r := m.Op("rsqrt", map[string]any{"x": ms, "epsilon": m.B.ScalarF16(float32(m.cfg.NormEps / (scale * scale)))}, msShape...)
	y := m.Op("mul", map[string]any{"x": xs, "y": r}, x.Shape...)
	w, ok := m.consts[name]
	if !ok {
		wshape := make([]int, len(x.Shape))
		for i := range wshape {
			wshape[i] = 1
		}
		wshape[len(wshape)-1] = n
		w = m.F16s(name, wshape, m.Get(name, n))
		m.consts[name] = w
	}
	return m.Op("mul", map[string]any{"x": y, "y": w}, x.Shape...)
}

// inputNorm is layer i's input_layernorm: on the token embeddings as they are before the first
// layer (residualScale).
func (m *builder) inputNorm(i int, h *mlpackage.Var) *mlpackage.Var {
	scale := float64(residualScale)
	if i == 0 {
		scale = 1
	}
	return m.norm(h, m.cfg.layerPrefix(i)+"input_layernorm.weight", scale)
}

// paged reports whether the layers' attention runs on the host between the parts (Options.Paged).
func (m *builder) paged() bool { return m.opts.Paged > 0 && m.opts.Cache > 0 }

// layer is OuroDecoderLayer: the attention and the MLP, each between two norms and added to the
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
	return m.layerRest(i, m.linear(o, c.layerPrefix(i)+"self_attn.o_proj.weight", m.d), x)
}

// layerRest is a layer after its attention (a, o_proj's output), given the hidden state entering
// the layer: input_layernorm_2 on the attention's output, added; post_attention_layernorm, the
// MLP, post_attention_layernorm_2, added.
func (m *builder) layerRest(i int, a, residual *mlpackage.Var) *mlpackage.Var {
	p := m.cfg.layerPrefix(i)
	h := m.Op("add", map[string]any{"x": residual, "y": m.norm(a, p+"input_layernorm_2.weight", 1)}, residual.Shape...)
	f := m.mlp(i, m.norm(h, p+"post_attention_layernorm.weight", residualScale))
	return m.Op("add", map[string]any{"x": h, "y": m.norm(f, p+"post_attention_layernorm_2.weight", 1)}, h.Shape...)
}

// mlpSplit is the widest intermediate the MLP computes in one piece: the Neural Engine's axis
// limit. A variable for the tests.
var mlpSplit = llm.NeuralEngineMaxDim

// mlp is OuroMLP: down_proj(silu(gate_proj(x)) * up_proj(x)). An intermediate over mlpSplit is
// computed in pieces of whole 32 columns, as Qwen3's.
func (m *builder) mlp(i int, h *mlpackage.Var) *mlpackage.Var {
	p := m.cfg.layerPrefix(i) + "mlp."
	f, d := m.cfg.IntermediateSize, m.d
	if f <= mlpSplit {
		gate := m.SiLU(m.linear(h, p+"gate_proj.weight", f))
		up := m.linear(h, p+"up_proj.weight", f)
		g := m.Op("mul", map[string]any{"x": gate, "y": up}, gate.Shape...)
		return m.linear(g, p+"down_proj.weight", d)
	}
	n := (f + mlpSplit - 1) / mlpSplit
	piece := (f/n + 31) / 32 * 32
	sh := append([]int{}, h.Shape[:len(h.Shape)-1]...)
	var sum *mlpackage.Var
	for a := 0; a < f; a += piece {
		b := min(a+piece, f)
		ps := append(append([]int{}, sh...), b-a)
		gate := m.Op("linear", map[string]any{"x": h, "weight": m.slice(p+"gate_proj.weight", f, d, a, b, 0, d)}, ps...)
		up := m.Op("linear", map[string]any{"x": h, "weight": m.slice(p+"up_proj.weight", f, d, a, b, 0, d)}, ps...)
		g := m.Op("mul", map[string]any{"x": m.SiLU(gate), "y": up}, ps...)
		y := m.Op("linear", map[string]any{"x": g, "weight": m.slice(p+"down_proj.weight", d, f, 0, d, a, b)}, h.Shape...)
		if sum == nil {
			sum = y
		} else {
			sum = m.Op("add", map[string]any{"x": sum, "y": y}, h.Shape...)
		}
	}
	return sum
}

// ropeTables returns the cos and sin tables (1, seq, 1, head_dim): constants for positions
// 0..seq-1, or inputs that StepInputs fills with a KV cache. Every pass turns by the same.
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
// seq, head_dim), the queries and keys turned by RoPE.
func (m *builder) qkv(i int, h *mlpackage.Var) (q, k, v *mlpackage.Var) {
	c := m.cfg
	p := c.layerPrefix(i) + "self_attn."
	hd, s := c.HeadDim, m.seq
	heads := func(name string, n int) *mlpackage.Var {
		t := m.linear(h, p+name+"_proj.weight", n*hd)
		return m.Op("reshape", map[string]any{"x": t, "shape": m.B.Ints(1, int32(s), int32(n), int32(hd))}, 1, s, n, hd)
	}
	toHeads := func(t *mlpackage.Var) *mlpackage.Var {
		sh := t.Shape
		return m.Op("transpose", map[string]any{"x": t, "perm": m.B.Ints(0, 2, 1, 3)}, sh[0], sh[2], sh[1], sh[3])
	}
	cos, sin := m.ropeTables()
	q = toHeads(m.ApplyRope(heads("q", c.NumHeads), cos, sin))
	k = toHeads(m.ApplyRope(heads("k", c.NumKVHeads), cos, sin))
	v = toHeads(heads("v", c.NumKVHeads))
	return q, k, v
}

// attention is OuroAttention: queries, keys and values, the cached keys and values of this pass
// of the layer before the step's with a cache, grouped causal attention scaled by
// 1/sqrt(head_dim), and o_proj.
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
	return m.linear(o, c.layerPrefix(i)+"self_attn.o_proj.weight", m.d)
}
