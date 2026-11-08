package gemma4

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Options select what to build. Gemma 4 has no settings of its own yet.
type Options struct {
	llm.Options
	// FoldedLogits leaves y as (1, seq, rows, vocab/rows) when the vocabulary is wider than the
	// Neural Engine takes elementwise (head): the reshape back to (1, seq, vocab) is the one
	// operation of the program the CPU runs otherwise. decode reads either.
	FoldedLogits bool
	// HostSoftcap leaves the soft cap of the logits to the host (-logits host, the default): y is
	// the output layer's (1, seq, vocab), all on the Neural Engine, and the host applies
	// tanh(y/cap)*cap to the logits it uses (llm.LogitSoftcap).
	HostSoftcap bool
}

type builder struct {
	*blocks.Graph
	cfg  *Config
	opts Options
	seq  int
	d    int
	end  int // one past the last layer built
	// rows is the rows the layers being built compute: seq, or 1 from the first layer that reuses
	// earlier keys and values on with LastRow (lastFrom), whose rows only the last token's logits
	// need
	rows int

	ins     map[string]*mlpackage.Var
	inList  []mlpackage.Feature
	outList []*mlpackage.Var

	rope   map[string][2]*mlpackage.Var // cos and sin by layer type
	masks  map[string]*mlpackage.Var    // by layer type
	shared map[string][2]*mlpackage.Var // keys and values the shared layers reuse, by layer type
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
// first. x is the fp16 token embeddings (1, seq, hidden), the rows of the table as they are
// (the program scales them), or the hidden state entering layer Start; y is the logits (1, seq,
// vocab) or a hidden state. The other inputs and outputs:
//
//   - ple (1, seq, layers*per-layer dim): the rows of the per-layer embeddings, which the host
//     looks up (llm.TokenInputs), in the program that starts at layer 0
//   - per_layer_inputs, the same shape: each layer's input from the per-layer embeddings, which
//     the program starting at layer 0 computes and outputs for the later parts of a split model,
//     and those take as an input
//   - shared_k_TYPE and shared_v_TYPE (1, kv heads, keys, head_dim): the keys and values of the
//     last layer of each type before the shared layers, output by the part that computes them
//     when shared layers come after it, and taken by a later part with shared layers
//   - with opts.Cache, the KV cache of each layer that has its own (kv_k_N, kv_v_N, a ring of
//     the window for a sliding layer when that is shorter) and the masks and rotary tables of
//     each layer type (mask_TYPE, rope_cos_TYPE, rope_sin_TYPE), as llm.Cached describes
//
// The keys of the shared values are the step's tokens without a cache, and the cache followed
// by them with one.
func Build(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts Options) (ins, outs []mlpackage.Feature, err error) {
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = "lut4"
	}
	if opts.Layers <= 0 || opts.Start+opts.Layers > cfg.NumLayers {
		opts.Layers = cfg.NumLayers - opts.Start
	}
	if opts.LastRow && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("gemma4: the logits of the last row alone (LastRow) are of a program with a KV cache")
	}
	if opts.Paged > 0 && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("gemma4: a program split at its attention (Paged) needs a KV cache")
	}
	m := &builder{Graph: &blocks.Graph{Name: "gemma4", B: b, W: w, Opts: opts.Options}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize,
		end: opts.Start + opts.Layers, ins: map[string]*mlpackage.Var{}, rope: map[string][2]*mlpackage.Var{}, masks: map[string]*mlpackage.Var{},
		shared: map[string][2]*mlpackage.Var{}}
	m.rows = m.seq
	from := m.lastFrom()
	if opts.Start > 0 && opts.Start >= from {
		m.rows = 1 // the part before gives the last token's row
	}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.rows, m.d)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		h := x
		var ple *mlpackage.Var
		if opts.Layers > 0 {
			h, ple = m.embedding(x)
		}
		if m.rows == 1 && ple != nil {
			ple = m.last(ple) // per_layer_inputs, of every row
		}
		for i := opts.Start; i < m.end; i++ {
			if i == from && i > opts.Start {
				h, ple = m.toLast(h, ple)
			}
			if i == opts.Start && i > 0 && m.paged(i) {
				h = m.afterAttention(i, h, ple) // the host attended between the parts
				continue
			}
			h = m.layer(i, h, ple)
		}
		if m.end == from && m.end > opts.Start {
			h, ple = m.toLast(h, ple) // the state entering the first layer of the last row alone
		}
		if m.end < cfg.NumLayers && m.paged(m.end) {
			m.beforeAttention(m.end, h) // the host attends next; y is the hidden state entering end
		}
		var y *mlpackage.Var
		if m.end < cfg.NumLayers || opts.NoHead {
			y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		} else {
			if opts.LastRow && m.rows > 1 {
				h = m.last(h)
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

// lastFrom is the first layer that computes the last token's row alone: with LastRow and a KV
// cache, the first that reuses earlier keys and values. A prompt's step leaves behind the cache
// and the logits after its last token; those layers write no cache, so the other rows of them go
// nowhere (design.md 16.80 on AMD, 16.83). Past the last layer without.
func (m *builder) lastFrom() int {
	if m.opts.LastRow && m.opts.Cache > 0 && m.cfg.KVShared > 0 {
		return m.cfg.firstShared()
	}
	return m.cfg.NumLayers + 1
}

// last is the row of v (1, seq, n) of the step's last token, which the input last (1, 1, seq),
// one-hot, picks: (1, 1, n).
func (m *builder) last(v *mlpackage.Var) *mlpackage.Var {
	return m.LastRow(v, m.input("last", 1, 1, m.seq))
}

// toLast takes the last token's rows of the hidden state and the per-layer inputs: the layers
// built from here on compute that row alone.
func (m *builder) toLast(h, ple *mlpackage.Var) (*mlpackage.Var, *mlpackage.Var) {
	h = m.last(h)
	if ple != nil {
		ple = m.last(ple)
	}
	m.rows = 1
	return h, ple
}

// lastRow4 is last for v (1, 1, seq, n) or (1, seq, 1, n), a mask or a rotary table: the last
// token's row, (1, 1, 1, n).
func (m *builder) lastRow4(v *mlpackage.Var) *mlpackage.Var {
	n := v.Shape[3]
	flat := m.Op("reshape", map[string]any{"x": v, "shape": m.B.Ints(1, int32(m.seq), int32(n))}, 1, m.seq, n)
	return m.Op("reshape", map[string]any{"x": m.last(flat), "shape": m.B.Ints(1, 1, 1, int32(n))}, 1, 1, 1, n)
}

// embedding is what comes before the first layer built: at layer 0, the scaling of the token
// embeddings (Gemma4TextScaledWordEmbedding) and the per-layer inputs; at a later layer, the
// per-layer inputs taken from the part that computed them. It returns the hidden state and the
// per-layer inputs (nil without per-layer embeddings).
func (m *builder) embedding(x *mlpackage.Var) (h, ple *mlpackage.Var) {
	c := m.cfg
	width := c.NumLayers * c.PerLayerDim
	if m.opts.Start > 0 {
		if c.PerLayerDim > 0 {
			ple = m.input("per_layer_inputs", 1, m.seq, width)
		}
		return x, ple
	}
	// The float32 reference multiplies by sqrt(hidden) exactly; bf16 inference rounds it first.
	h = m.Op("mul", map[string]any{"x": x, "y": m.B.ScalarF16(float32(math.Sqrt(float64(m.d))))}, x.Shape...)
	if c.PerLayerDim > 0 {
		ple = m.perLayerInputs(h)
		if m.end < c.NumLayers {
			m.output(ple, "per_layer_inputs")
		}
	}
	return h, ple
}

// perLayerInputs is get_per_layer_inputs and project_per_layer_inputs: the per-layer
// embeddings of the tokens (looked up by the host, scaled by sqrt of their width) added to a
// projection of the scaled token embeddings, normalized per layer, and both scaled by 1/sqrt(2).
// The result is (1, seq, layers*per-layer dim), layer i's input in the i-th slice.
func (m *builder) perLayerInputs(h *mlpackage.Var) *mlpackage.Var {
	c, p := m.cfg, m.cfg.Prefix
	n, pd, s := c.NumLayers, c.PerLayerDim, m.seq
	proj := m.LinearS(h, p+"per_layer_model_projection.weight", "", n*pd, float32(1/math.Sqrt(float64(m.d))))
	proj = m.Op("reshape", map[string]any{"x": proj, "shape": m.B.Ints(1, int32(s), int32(n), int32(pd))}, 1, s, n, pd)
	// the projection is small (RMS about 0.3 in E2B): no scaling down before the squares
	proj = m.RMSNorm(proj, p+"per_layer_projection_norm.weight", pd, c.NormEps, 1)
	tok := m.input("ple", 1, s, n*pd)
	tok = m.Op("reshape", map[string]any{"x": tok, "shape": m.B.Ints(1, int32(s), int32(n), int32(pd))}, 1, s, n, pd)
	tok = m.Op("mul", map[string]any{"x": tok, "y": m.B.ScalarF16(float32(math.Sqrt(float64(pd))))}, tok.Shape...)
	sum := m.Op("add", map[string]any{"x": proj, "y": tok}, tok.Shape...)
	sum = m.Op("mul", map[string]any{"x": sum, "y": m.B.ScalarF16(float32(1 / math.Sqrt2))}, sum.Shape...)
	return m.Op("reshape", map[string]any{"x": sum, "shape": m.B.Ints(1, int32(s), int32(n*pd))}, 1, s, n*pd)
}

// head is the final norm, the output layer tied to the embeddings and the soft cap of the logits.
// By default (Options.HostSoftcap) the host applies the soft cap: it is monotonic, so sampling
// needs its values for the few most likely tokens alone, and the program is all on the Neural
// Engine. On an M1 the output layer's part took 2.3 ms less a step (docs/design.md 16.113).
// In the program instead: the Neural Engine takes the output layer's 262144 wide axis but not
// elementwise operations over it (they go to the CPU), so the soft cap runs on the logits folded
// into rows of at most llm.NeuralEngineMaxDim. The reshape back to (1, seq, vocab) still goes to
// the CPU (flat); leaving y folded (Options.FoldedLogits) puts everything on the Neural Engine
// but was 5 to 12% slower on an M3 (16.43).
func (m *builder) head(h *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	seq := h.Shape[1] // 1 for the last row alone (LastRow)
	h = m.norm(h, c.Prefix+"norm.weight", m.d)
	if c.Softcap == nil || m.opts.HostSoftcap {
		return m.Linear(h, c.Embedding(), "y", c.VocabSize)
	}
	limit := float32(*c.Softcap)
	y := m.Linear(h, c.Embedding(), "", c.VocabSize)
	rows := 1
	for c.VocabSize/rows > llm.NeuralEngineMaxDim || c.VocabSize%rows != 0 {
		rows++
	}
	if rows > 1 {
		y = m.Op("reshape", map[string]any{"x": y, "shape": m.B.Ints(1, int32(seq), int32(rows), int32(c.VocabSize/rows))}, 1, seq, rows, c.VocabSize/rows)
	}
	y = m.Op("mul", map[string]any{"x": y, "y": m.B.ScalarF16(1 / limit)}, y.Shape...)
	y = m.Op("tanh", map[string]any{"x": y}, y.Shape...)
	if rows == 1 {
		return m.B.NamedOp("mul", "y", map[string]any{"x": y, "y": m.B.ScalarF16(limit)}, mlpackage.Float16, y.Shape...)
	}
	if m.opts.FoldedLogits {
		return m.B.NamedOp("mul", "y", map[string]any{"x": y, "y": m.B.ScalarF16(limit)}, mlpackage.Float16, y.Shape...)
	}
	y = m.Op("mul", map[string]any{"x": y, "y": m.B.ScalarF16(limit)}, y.Shape...)
	return m.B.NamedOp("reshape", "y", map[string]any{"x": y, "shape": m.B.Ints(1, int32(seq), int32(c.VocabSize))}, mlpackage.Float16, 1, seq, c.VocabSize)
}

// normScale is how much the norms scale their input down before taking the squares. The square
// of anything over 256 overflows fp16, and E2B goes far over: its layer 0 multiplies the
// normalized input by weights of up to 270, the projections of that into queries and keys are
// normalized again, and its residual reaches about 1900 before layer_scalar (0.018) brings it back
// to 33. The scaled input is what is normalized, so the norm does not change (with the epsilon
// scaled to match) and its weight is used as it is: dividing the weight instead (blocks.RMSNorm's
// inScale) would make E2B's smallest norm weights (0.16) fp16 subnormals.
const normScale = 64

// norm is Gemma4RMSNorm over the last axis (size n), with the squares taken of x/normScale, times
// the named weight, or without one for an empty name (with_scale=False).
func (m *builder) norm(x *mlpackage.Var, name string, n int) *mlpackage.Var {
	msShape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), 1)
	xs := m.Op("mul", map[string]any{"x": x, "y": m.B.ScalarF16(1.0 / normScale)}, x.Shape...)
	sq := m.Op("mul", map[string]any{"x": xs, "y": xs}, x.Shape...)
	ms := m.Op("reduce_mean", map[string]any{"x": sq, "axes": m.B.Ints(-1), "keep_dims": m.B.ScalarBool(true)}, msShape...)
	r := m.Op("rsqrt", map[string]any{"x": ms, "epsilon": m.B.ScalarF16(float32(m.cfg.NormEps / (normScale * normScale)))}, msShape...)
	y := m.Op("mul", map[string]any{"x": xs, "y": r}, x.Shape...)
	if name == "" {
		return y
	}
	wshape := make([]int, len(x.Shape))
	for i := range wshape {
		wshape[i] = 1
	}
	wshape[len(wshape)-1] = n
	gamma := m.F16s(name, wshape, m.Get(name, n))
	return m.Op("mul", map[string]any{"x": y, "y": gamma}, x.Shape...)
}

// layer is Gemma4TextDecoderLayer: attention and the MLP, each between two norms and added to
// the residual; then the per-layer input gated by the hidden state, projected, normalized and
// added; and the whole scaled by layer_scalar.
func (m *builder) layer(i int, h, ple *mlpackage.Var) *mlpackage.Var {
	if m.paged(i) {
		m.Fail("layer %d's attention runs on the host: a part must start or end at it", i)
	}
	p := fmt.Sprintf("%slayers.%d.", m.cfg.Prefix, i)
	return m.layerRest(i, m.attention(i, m.norm(h, p+"input_layernorm.weight", m.d)), h, ple)
}

// paged reports whether layer i's attention runs on the host between the parts (Options.Paged).
func (m *builder) paged(i int) bool {
	return m.opts.Paged > 0 && m.opts.Cache > 0 && m.cfg.LayerTypes[i] == "full_attention"
}

// beforeAttention is the start of layer i, whose attention the host runs (llm.Paged): it outputs
// the queries attn_q_N and, unless the layer reuses an earlier layer's keys and values, the step's
// keys and values kv_k_N_new and kv_v_N_new.
func (m *builder) beforeAttention(i int, h *mlpackage.Var) {
	p := fmt.Sprintf("%slayers.%d.", m.cfg.Prefix, i)
	h = m.norm(h, p+"input_layernorm.weight", m.d)
	m.output(m.query(i, h), fmt.Sprintf("attn_q_%d", i))
	if !m.cfg.shared(i) {
		k, v := m.keysValues(i, h)
		m.output(k, fmt.Sprintf("kv_k_%d_new", i))
		m.output(v, fmt.Sprintf("kv_v_%d_new", i))
	}
}

// afterAttention is the rest of layer i from its attention's output, attn_o_N (1, heads, seq,
// head_dim), with x the hidden state entering the layer.
func (m *builder) afterAttention(i int, x, ple *mlpackage.Var) *mlpackage.Var {
	o := m.input(fmt.Sprintf("attn_o_%d", i), 1, m.cfg.NumHeads, m.rows, m.cfg.headDim(m.cfg.LayerTypes[i]))
	return m.layerRest(i, m.attnOutput(i, o), x, ple)
}

// layerRest is a layer after its attention (a, before the norm that follows it), given the
// hidden state entering the layer.
func (m *builder) layerRest(i int, a, residual, ple *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	p := fmt.Sprintf("%slayers.%d.", c.Prefix, i)
	a = m.norm(a, p+"post_attention_layernorm.weight", m.d)
	h := m.Op("add", map[string]any{"x": residual, "y": a}, residual.Shape...)

	residual = h
	f := m.mlp(i, m.norm(h, p+"pre_feedforward_layernorm.weight", m.d))
	f = m.norm(f, p+"post_feedforward_layernorm.weight", m.d)
	h = m.Op("add", map[string]any{"x": residual, "y": f}, h.Shape...)

	if ple != nil {
		residual = h
		pd := c.PerLayerDim
		g := m.Linear(h, p+"per_layer_input_gate.weight", "", pd)
		g = m.Activation(g, c.HiddenAct)
		mine := m.Op("slice_by_index", map[string]any{"x": ple, "begin": m.B.Ints(0, 0, int32(i*pd)), "end": m.B.Ints(1, int32(m.rows), int32((i+1)*pd)),
			"begin_mask": m.Bools(true, true, false), "end_mask": m.Bools(true, true, false)}, 1, m.rows, pd)
		g = m.Op("mul", map[string]any{"x": g, "y": mine}, g.Shape...)
		g = m.Linear(g, p+"per_layer_projection.weight", "", m.d)
		g = m.norm(g, p+"post_per_layer_input_norm.weight", m.d)
		h = m.Op("add", map[string]any{"x": residual, "y": g}, h.Shape...)
	}
	if k := m.Get(p+"layer_scalar", 1)[0]; k != 1 {
		h = m.Op("mul", map[string]any{"x": h, "y": m.B.ScalarF16(k)}, h.Shape...)
	}
	return h
}

// mlp is Gemma4TextMLP: down(act(gate(x)) * up(x)).
func (m *builder) mlp(i int, h *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("%slayers.%d.mlp.", m.cfg.Prefix, i)
	f := m.cfg.mlpWidth(i)
	g := m.Activation(m.Linear(h, p+"gate_proj.weight", "", f), m.cfg.HiddenAct)
	u := m.Linear(h, p+"up_proj.weight", "", f)
	g = m.Op("mul", map[string]any{"x": g, "y": u}, g.Shape...)
	return m.Linear(g, p+"down_proj.weight", "", m.d)
}

// ropeTables returns the cos and sin tables (1, seq, 1, head_dim) of a layer type: constants
// for positions 0..seq-1, or inputs that StepInputs fills with a KV cache.
func (m *builder) ropeTables(layerType string) (cos, sin *mlpackage.Var) {
	if m.rows == 1 && m.seq > 1 {
		// the last token's rows of the tables
		if t, ok := m.rope[layerType+"/last"]; ok {
			return t[0], t[1]
		}
		m.rows = m.seq
		cos, sin = m.ropeTables(layerType)
		m.rows = 1
		cos, sin = m.lastRow4(cos), m.lastRow4(sin)
		m.rope[layerType+"/last"] = [2]*mlpackage.Var{cos, sin}
		return cos, sin
	}
	if t, ok := m.rope[layerType]; ok {
		return t[0], t[1]
	}
	hd := m.cfg.headDim(layerType)
	if m.opts.Cache > 0 {
		cos, sin = m.input("rope_cos_"+layerType, 1, m.seq, 1, hd), m.input("rope_sin_"+layerType, 1, m.seq, 1, hd)
	} else {
		cos, sin = m.RopeTables(layerType, m.cfg.RopeInvFreq(layerType), m.seq, hd)
	}
	m.rope[layerType] = [2]*mlpackage.Var{cos, sin}
	return cos, sin
}

// mask returns the additive mask of a layer type over keys keys: causal (and windowed for a
// sliding layer) without a cache, an input that StepInputs fills with one.
func (m *builder) mask(layerType string, keys int) *mlpackage.Var {
	if m.rows == 1 && m.seq > 1 {
		// the last token's row of the mask
		if v, ok := m.masks[layerType+"/last"]; ok {
			return v
		}
		m.rows = m.seq
		v := m.mask(layerType, keys)
		m.rows = 1
		v = m.lastRow4(v)
		m.masks[layerType+"/last"] = v
		return v
	}
	if v, ok := m.masks[layerType]; ok {
		return v
	}
	var v *mlpackage.Var
	if m.opts.Cache > 0 {
		v = m.input("mask_"+layerType, 1, 1, m.seq, keys)
	} else {
		window := 0
		if layerType == "sliding_attention" {
			window = m.cfg.SlidingWindow
		}
		v = m.CausalMask("mask_"+layerType, m.seq, window)
	}
	m.masks[layerType] = v
	return v
}

// heads reshapes a projection (1, seq, n*hd) to (1, seq, n, hd).
func (m *builder) heads(x *mlpackage.Var, n, hd int) *mlpackage.Var {
	return m.Op("reshape", map[string]any{"x": x, "shape": m.B.Ints(1, int32(m.rows), int32(n), int32(hd))}, 1, m.rows, n, hd)
}

// toHeads transposes (1, seq, n, hd) to (1, n, seq, hd).
func (m *builder) toHeads(x *mlpackage.Var) *mlpackage.Var {
	sh := x.Shape
	return m.Op("transpose", map[string]any{"x": x, "perm": m.B.Ints(0, 2, 1, 3)}, sh[0], sh[2], sh[1], sh[3])
}

// attention is Gemma4TextAttention: queries with a norm and RoPE, keys with a norm and RoPE and
// values with a norm without weight, or for a shared layer the keys and values of the last
// earlier layer of its type; attention with no score scaling (the norms take its place), causal
// and for a sliding layer windowed; and o_proj.
func (m *builder) attention(i int, h *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	lt := c.LayerTypes[i]
	s, nkv, hd := m.seq, c.NumKVHeads, c.headDim(lt)
	q := m.query(i, h)

	keys := s
	if m.opts.Cache > 0 {
		keys += c.cacheLen(lt, m.opts.Cache)
	}
	var k, v *mlpackage.Var
	if c.shared(i) {
		kv, ok := m.shared[lt]
		if !ok {
			// computed by an earlier part
			kv = [2]*mlpackage.Var{m.input(sharedName("k", lt), 1, nkv, keys, hd), m.input(sharedName("v", lt), 1, nkv, keys, hd)}
			m.shared[lt] = kv
		}
		k, v = kv[0], kv[1]
	} else {
		k, v = m.keysValues(i, h)
		if m.opts.Cache > 0 {
			// the cached keys and values, then the step's own
			t := c.cacheLen(lt, m.opts.Cache)
			m.output(k, fmt.Sprintf("kv_k_%d_new", i))
			m.output(v, fmt.Sprintf("kv_v_%d_new", i))
			cat := func(name string, x *mlpackage.Var) *mlpackage.Var {
				in := m.input(name, 1, nkv, t, hd)
				return m.Op("concat", map[string]any{"values": []*mlpackage.Var{in, x}, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, nkv, t+s, hd)
			}
			k, v = cat(fmt.Sprintf("kv_k_%d", i), k), cat(fmt.Sprintf("kv_v_%d", i), v)
		}
		if c.KVShared > 0 && i == c.source(lt) {
			m.shared[lt] = [2]*mlpackage.Var{k, v}
			if m.sharedAfter(lt) {
				m.output(k, sharedName("k", lt))
				m.output(v, sharedName("v", lt))
			}
		}
	}
	o := m.SDPAGrouped(q, k, v, 1, func() *mlpackage.Var { return m.mask(lt, keys) })
	return m.attnOutput(i, o)
}

// query is layer i's queries (1, heads, seq, head_dim): the projection with its norm and RoPE.
func (m *builder) query(i int, h *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	p := fmt.Sprintf("%slayers.%d.self_attn.", c.Prefix, i)
	lt := c.LayerTypes[i]
	nh, hd := c.NumHeads, c.headDim(lt)
	cos, sin := m.ropeTables(lt)
	q := m.heads(m.Linear(h, p+"q_proj.weight", "", nh*hd), nh, hd)
	return m.toHeads(m.ApplyRope(m.norm(q, p+"q_norm.weight", hd), cos, sin))
}

// keysValues is layer i's keys and values of the step's tokens (1, kv heads, seq, head_dim): the
// keys with a norm and RoPE, the values with a norm without weight.
func (m *builder) keysValues(i int, h *mlpackage.Var) (k, v *mlpackage.Var) {
	c := m.cfg
	p := fmt.Sprintf("%slayers.%d.self_attn.", c.Prefix, i)
	lt := c.LayerTypes[i]
	nkv, hd := c.NumKVHeads, c.headDim(lt)
	cos, sin := m.ropeTables(lt)
	k = m.heads(m.Linear(h, p+"k_proj.weight", "", nkv*hd), nkv, hd)
	k = m.toHeads(m.ApplyRope(m.norm(k, p+"k_norm.weight", hd), cos, sin))
	v = m.heads(m.Linear(h, p+"v_proj.weight", "", nkv*hd), nkv, hd)
	return k, m.toHeads(m.norm(v, "", hd))
}

// attnOutput is o_proj of the attention's output o (1, heads, seq, head_dim).
func (m *builder) attnOutput(i int, o *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	nh, hd := c.NumHeads, c.headDim(c.LayerTypes[i])
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, m.rows, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(m.rows), int32(nh*hd))}, 1, m.rows, nh*hd)
	return m.Linear(o, fmt.Sprintf("%slayers.%d.self_attn.o_proj.weight", c.Prefix, i), "", m.d)
}

// sharedAfter reports whether a shared layer of the type comes after the layers built.
func (m *builder) sharedAfter(layerType string) bool {
	for j := max(m.end, m.cfg.firstShared()); j < m.cfg.NumLayers; j++ {
		if m.cfg.LayerTypes[j] == layerType {
			return true
		}
	}
	return false
}
