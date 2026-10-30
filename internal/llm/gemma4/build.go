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
}

type builder struct {
	*blocks.Graph
	cfg  *Config
	opts Options
	seq  int
	d    int
	end  int // one past the last layer built

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
	m := &builder{Graph: &blocks.Graph{Name: "gemma4", B: b, W: w, Opts: opts.Options}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize,
		end: opts.Start + opts.Layers, ins: map[string]*mlpackage.Var{}, rope: map[string][2]*mlpackage.Var{}, masks: map[string]*mlpackage.Var{},
		shared: map[string][2]*mlpackage.Var{}}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		h := x
		var ple *mlpackage.Var
		if opts.Layers > 0 {
			h, ple = m.embedding(x)
		}
		for i := opts.Start; i < m.end; i++ {
			h = m.layer(i, h, ple)
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
// The Neural Engine takes the output layer's 262144 wide axis but not elementwise operations over
// it (they go to the CPU), so the soft cap runs on the logits folded into rows of at most
// llm.NeuralEngineMaxDim. The reshape back to (1, seq, vocab) still goes to the CPU; leaving y
// folded puts everything on the Neural Engine but made a token four times slower (137 ms on the
// M3, against 33), reading the folded output back.
func (m *builder) head(h *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	h = m.norm(h, c.Prefix+"norm.weight", m.d)
	if c.Softcap == nil {
		return m.Linear(h, c.Embedding(), "y", c.VocabSize)
	}
	limit := float32(*c.Softcap)
	y := m.Linear(h, c.Embedding(), "", c.VocabSize)
	rows := 1
	for c.VocabSize/rows > llm.NeuralEngineMaxDim || c.VocabSize%rows != 0 {
		rows++
	}
	if rows > 1 {
		y = m.Op("reshape", map[string]any{"x": y, "shape": m.B.Ints(1, int32(m.seq), int32(rows), int32(c.VocabSize/rows))}, 1, m.seq, rows, c.VocabSize/rows)
	}
	y = m.Op("mul", map[string]any{"x": y, "y": m.B.ScalarF16(1 / limit)}, y.Shape...)
	y = m.Op("tanh", map[string]any{"x": y}, y.Shape...)
	if rows == 1 {
		return m.B.NamedOp("mul", "y", map[string]any{"x": y, "y": m.B.ScalarF16(limit)}, mlpackage.Float16, y.Shape...)
	}
	y = m.Op("mul", map[string]any{"x": y, "y": m.B.ScalarF16(limit)}, y.Shape...)
	return m.B.NamedOp("reshape", "y", map[string]any{"x": y, "shape": m.B.Ints(1, int32(m.seq), int32(c.VocabSize))}, mlpackage.Float16, 1, m.seq, c.VocabSize)
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
	c := m.cfg
	p := fmt.Sprintf("%slayers.%d.", c.Prefix, i)
	residual := h
	a := m.attention(i, m.norm(h, p+"input_layernorm.weight", m.d))
	a = m.norm(a, p+"post_attention_layernorm.weight", m.d)
	h = m.Op("add", map[string]any{"x": residual, "y": a}, h.Shape...)

	residual = h
	f := m.mlp(i, m.norm(h, p+"pre_feedforward_layernorm.weight", m.d))
	f = m.norm(f, p+"post_feedforward_layernorm.weight", m.d)
	h = m.Op("add", map[string]any{"x": residual, "y": f}, h.Shape...)

	if ple != nil {
		residual = h
		pd := c.PerLayerDim
		g := m.Linear(h, p+"per_layer_input_gate.weight", "", pd)
		g = m.Activation(g, c.HiddenAct)
		mine := m.Op("slice_by_index", map[string]any{"x": ple, "begin": m.B.Ints(0, 0, int32(i*pd)), "end": m.B.Ints(1, int32(m.seq), int32((i+1)*pd)),
			"begin_mask": m.Bools(true, true, false), "end_mask": m.Bools(true, true, false)}, 1, m.seq, pd)
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
	return m.Op("reshape", map[string]any{"x": x, "shape": m.B.Ints(1, int32(m.seq), int32(n), int32(hd))}, 1, m.seq, n, hd)
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
	p := fmt.Sprintf("%slayers.%d.self_attn.", c.Prefix, i)
	lt := c.LayerTypes[i]
	s, nh, nkv, hd := m.seq, c.NumHeads, c.NumKVHeads, c.headDim(lt)
	cos, sin := m.ropeTables(lt)

	q := m.heads(m.Linear(h, p+"q_proj.weight", "", nh*hd), nh, hd)
	q = m.toHeads(m.ApplyRope(m.norm(q, p+"q_norm.weight", hd), cos, sin))

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
		k = m.heads(m.Linear(h, p+"k_proj.weight", "", nkv*hd), nkv, hd)
		k = m.toHeads(m.ApplyRope(m.norm(k, p+"k_norm.weight", hd), cos, sin))
		v = m.heads(m.Linear(h, p+"v_proj.weight", "", nkv*hd), nkv, hd)
		v = m.toHeads(m.norm(v, "", hd))
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
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, s, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(s), int32(nh*hd))}, 1, s, nh*hd)
	return m.Linear(o, p+"o_proj.weight", "", m.d)
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
