package gptoss

import (
	"fmt"
	"slices"
	"strings"

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
	ex   *experts

	ins     map[string]*mlpackage.Var
	inList  []mlpackage.Feature
	outList []*mlpackage.Var

	cos, sin *mlpackage.Var
	masks    map[bool]*mlpackage.Var
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

// withDefaults gives the options gpt-oss's defaults: the attention's projections in attn-format
// (8 bit tables) unless FormatFor names them, and the router in fp16 (its 32 logits decide which
// experts weigh in).
func withDefaults(o llm.Options) llm.Options {
	if o.Group == 0 {
		o.Group = 16
	}
	if o.Format == "" {
		o.Format = "lut4"
	}
	if !slices.ContainsFunc(o.FormatFor, func(r string) bool { return strings.HasPrefix(r, "self_attn=") }) {
		if f := Arch.Param(o, "attn-format"); f != "" && o.Format != "fp16" {
			o.FormatFor = append([]string{"self_attn=" + f}, o.FormatFor...)
		}
	}
	o.KeepFP16 = append(slices.Clone(o.KeepFP16), "mlp.router")
	return o
}

// Build writes the program for the model into b and returns its inputs and outputs, x and y
// first. x is the fp16 token embeddings (1, seq, hidden), the rows of the table as they are, or
// the hidden state entering layer Start; y is the logits (1, seq, vocab) or a hidden state. With
// opts.Cache, the KV cache of each layer (kv_k_N, kv_v_N; a ring of the window for a sliding layer
// when that is shorter), the masks of the sliding and full layers (mask_sliding, mask_full) and
// the rotary tables (rope_cos, rope_sin) are inputs and the step's keys and values outputs, as
// llm.Cached describes; with opts.Paged as well, the full layers' attention runs on the host
// between the parts (llm.Paged).
func Build(b *mlpackage.Builder, cfg *Config, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	opts = withDefaults(opts)
	switch {
	case opts.Layers == llm.NoLayers:
		return nil, nil, fmt.Errorf("gptoss: layer 0 is a sliding layer, whose attention is in the program: no part ends before it")
	case opts.Layers <= 0 || opts.Start+opts.Layers > cfg.NumLayers:
		opts.Layers = cfg.NumLayers - opts.Start
	}
	if opts.Paged > 0 && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("gptoss: a program split at its attention (Paged) needs a KV cache")
	}
	if opts.LastRow && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("gptoss: the logits of the last row alone (LastRow) are of a program with a KV cache")
	}
	m := &builder{Graph: &blocks.Graph{Name: "gptoss", B: b, W: w, Opts: opts}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize,
		end: opts.Start + opts.Layers, ex: newExperts(cfg, w), ins: map[string]*mlpackage.Var{}, masks: map[bool]*mlpackage.Var{}}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		// Nothing comes between the embedding lookup and layer 0.
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
			if opts.LastRow {
				h = m.LastRow(h, m.input("last", 1, 1, m.seq))
			}
			y = m.Linear(m.norm(h, "model.norm.weight", normScale), "lm_head.weight", "y", cfg.VocabSize)
		}
		outs = []mlpackage.Feature{{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}}
		for _, v := range m.outList {
			outs = append(outs, mlpackage.Feature{Name: v.Name, DType: mlpackage.Float16, Shape: v.Shape})
		}
		ins = append(ins, m.inList...)
	})
	return ins, outs, err
}

// normScale is how much the norms scale their input down before taking the squares, which
// overflow fp16 past 256. The scaled input is what is normalized, so the norm does not change
// (with the epsilon scaled to match), and the weights stay as they are. The norm before layer 0
// reads the token embeddings, whose squares scaled down would fall under fp16's smallest values,
// so it takes them as they are (scale 1).
const normScale = 64

// norm is GptOssRMSNorm over the hidden axis: x normalized, times the named weight, with x scaled
// down by scale before its squares.
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
	return m.norm(h, layerPrefix(i)+"input_layernorm.weight", scale)
}

// paged reports whether layer i's attention runs on the host between the parts (Options.Paged).
func (m *builder) paged(i int) bool {
	return m.opts.Paged > 0 && m.opts.Cache > 0 && m.cfg.full(i)
}

// layer is GptOssDecoderLayer: the attention and the experts, each after a norm and added to the
// residual.
func (m *builder) layer(i int, h *mlpackage.Var) *mlpackage.Var {
	if m.paged(i) {
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
// attn_ot_N (1, seq, heads, head_dim), which the host lays out with the sinks in (llm.Paged), with
// x the hidden state entering the layer.
func (m *builder) afterAttention(i int, x *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	o := m.input(fmt.Sprintf("attn_ot_%d", i), 1, m.seq, c.NumHeads, c.HeadDim)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(m.seq), int32(c.NumHeads*c.HeadDim))}, 1, m.seq, c.NumHeads*c.HeadDim)
	p := layerPrefix(i) + "self_attn."
	return m.layerRest(i, m.LinearBias(o, p+"o_proj.weight", p+"o_proj.bias", "", m.d), x)
}

// layerRest is a layer after its attention (a, o_proj's output), given the hidden state entering
// the layer.
func (m *builder) layerRest(i int, a, residual *mlpackage.Var) *mlpackage.Var {
	h := m.Op("add", map[string]any{"x": residual, "y": a}, residual.Shape...)
	f := m.moe(i, m.norm(h, layerPrefix(i)+"post_attention_layernorm.weight", normScale))
	return m.Op("add", map[string]any{"x": h, "y": f}, h.Shape...)
}

// ropeTables returns the cos and sin tables (1, seq, 1, head_dim), the same for every layer:
// constants for positions 0..seq-1, or inputs that StepInputs fills with a KV cache. YaRN's
// concentration is not in them: the attention scales the scores by its square (scoreScale).
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
// seq, head_dim): the projections with their biases, the queries and keys turned by RoPE.
func (m *builder) qkv(i int, h *mlpackage.Var) (q, k, v *mlpackage.Var) {
	c := m.cfg
	p := layerPrefix(i) + "self_attn."
	hd, s := c.HeadDim, m.seq
	heads := func(name string, n int) *mlpackage.Var {
		t := m.LinearBias(h, p+name+"_proj.weight", p+name+"_proj.bias", "", n*hd)
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

// attention is GptOssAttention: queries, keys and values, the cached keys and values before the
// step's with a cache, grouped attention with the layer's sinks, causal and on a sliding layer
// windowed, and o_proj.
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
	o := m.sdpaSinks(i, q, k, v, m.mask(full, keys))
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, s, c.NumHeads, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(s), int32(c.NumHeads*hd))}, 1, s, c.NumHeads*hd)
	p := layerPrefix(i) + "self_attn."
	return m.LinearBias(o, p+"o_proj.weight", p+"o_proj.bias", "", m.d)
}

// sdpaSinks is grouped attention (blocks.SDPAGrouped) with layer i's sinks: the scores of each
// query head get one more column, the head's sink logit, before the softmax, which is dropped
// after it, so that the sink takes a share of the weight and adds nothing (eager_attention_forward
// of modeling_gpt_oss.py).
func (m *builder) sdpaSinks(i int, q, k, v, mask *mlpackage.Var) *mlpackage.Var {
	nh, s, hd := q.Shape[1], q.Shape[2], q.Shape[3]
	nkv, sk := k.Shape[1], k.Shape[2]
	rep := nh / nkv
	qg := m.Op("reshape", map[string]any{"x": q, "shape": m.B.Ints(1, int32(nkv), int32(rep*s), int32(hd))}, 1, nkv, rep*s, hd)
	scores := m.Op("matmul", map[string]any{"x": qg, "y": k, "transpose_x": m.B.ScalarBool(false), "transpose_y": m.B.ScalarBool(true)}, 1, nkv, rep*s, sk)
	scores = m.Op("mul", map[string]any{"x": scores, "y": m.B.ScalarF16(float32(m.cfg.scoreScale()))}, scores.Shape...)
	scores = m.Op("reshape", map[string]any{"x": scores, "shape": m.B.Ints(1, int32(nh), int32(s), int32(sk))}, 1, nh, s, sk)
	scores = m.Op("add", map[string]any{"x": scores, "y": mask}, 1, nh, s, sk)
	name := layerPrefix(i) + "self_attn.sinks"
	sinks := m.Get(name, nh)
	col := make([]float32, nh*s)
	for h := range nh {
		for t := range s {
			col[h*s+t] = sinks[h]
		}
	}
	all := m.Op("concat", map[string]any{"values": []*mlpackage.Var{scores, m.F16s(name, []int{1, nh, s, 1}, col)}, "axis": m.B.ScalarI32(3), "interleave": m.B.ScalarBool(false)}, 1, nh, s, sk+1)
	probs := m.Op("softmax", map[string]any{"x": all, "axis": m.B.ScalarI32(-1)}, all.Shape...)
	probs = m.Op("slice_by_index", map[string]any{"x": probs, "begin": m.B.Ints(0, 0, 0, 0), "end": m.B.Ints(1, int32(nh), int32(s), int32(sk))}, 1, nh, s, sk)
	probs = m.Op("reshape", map[string]any{"x": probs, "shape": m.B.Ints(1, int32(nkv), int32(rep*s), int32(sk))}, 1, nkv, rep*s, sk)
	o := m.Op("matmul", map[string]any{"x": probs, "y": v, "transpose_x": m.B.ScalarBool(false), "transpose_y": m.B.ScalarBool(false)}, 1, nkv, rep*s, hd)
	return m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(nh), int32(s), int32(hd))}, 1, nh, s, hd)
}

// moe is GptOssMLP of x (1, seq, hidden), normed: the router's weights of the experts (route)
// and every expert of the layer weighed by them. The Neural Engine cannot pick the experts'
// weights at run time, so the program computes them all, an expert's share zero when it is not
// among a token's top ones. They go in groups (groupOf): the gate and the up of each expert, then
// one down over the group's weighted hidden values side by side, each expert's down with tables
// of its own (columns in blocks of the intermediate size).
func (m *builder) moe(i int, x *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	p := layerPrefix(i) + "mlp."
	nE, d, f, s := c.Experts, c.HiddenSize, c.IntermediateSize, m.seq
	r := m.LinearBias(x, p+"router.weight", p+"router.bias", "", nE)
	w := m.route(r)
	per := groupOf(nE, f)
	downs := m.Graph
	if per > 1 {
		g := *m.Graph
		g.Opts.ColBlock = f
		downs = &g
	}
	var sum *mlpackage.Var
	for g0 := 0; g0 < nE; g0 += per {
		var hs []*mlpackage.Var
		down := make([]float32, d*per*f)
		for e := g0; e < g0+per; e++ {
			ew := m.ex.get(m.Graph, i, e)
			name := fmt.Sprintf("%sexperts.%d.", p, e)
			gate := m.expertLinear(x, name+"gate", ew.gate, ew.gateBias, f, d)
			up := m.expertLinear(x, name+"up", ew.up, ew.upBias, f, d)
			h := m.glu(gate, up)
			we := m.Op("slice_by_index", map[string]any{"x": w, "begin": m.B.Ints(0, 0, int32(e)), "end": m.B.Ints(1, int32(s), int32(e+1))}, 1, s, 1)
			hs = append(hs, m.Op("mul", map[string]any{"x": h, "y": we}, h.Shape...))
			for row := 0; row < d; row++ {
				copy(down[row*per*f+(e-g0)*f:], ew.down[row*f:(row+1)*f])
			}
		}
		cat := hs[0]
		if per > 1 {
			cat = m.Op("concat", map[string]any{"values": hs, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, s, per*f)
		}
		dw, scale := m.expertMatrix(downs, fmt.Sprintf("%sexperts.%d_%d.down", p, g0, g0+per-1), down, d, per*f)
		y := m.Op("linear", map[string]any{"x": cat, "weight": dw}, 1, s, d)
		if scale != nil {
			y = m.Op("mul", map[string]any{"x": y, "y": m.F16s(fmt.Sprintf("%sexperts.%d_%d.down_scale", p, g0, g0+per-1), []int{1, 1, d}, scale)}, y.Shape...)
		}
		if sum == nil {
			sum = y
		} else {
			sum = m.Op("add", map[string]any{"x": sum, "y": y}, sum.Shape...)
		}
	}
	// the downs' biases, each times its expert's weight: w (1, seq, experts) times (experts, hidden)
	bias := m.ex.downBias(m.Graph, i) // (experts, hidden)
	bt := make([]float32, d*nE)
	for e := range nE {
		for j := range d {
			bt[j*nE+e] = bias[e*d+j]
		}
	}
	b := m.Op("linear", map[string]any{"x": w, "weight": m.F16s(p+"experts.down_bias", []int{d, nE}, bt)}, 1, s, d)
	return m.Op("add", map[string]any{"x": sum, "y": b}, sum.Shape...)
}

// expertLinear is x times an expert's weight w (rows, cols) plus its bias (expertMatrix): with
// the rows' powers of two taken out of the weight, the bias divided by them before and the
// output multiplied by them after.
func (m *builder) expertLinear(x *mlpackage.Var, name string, w, bias []float32, rows, cols int) *mlpackage.Var {
	wv, scale := m.expertMatrix(m.Graph, name, w, rows, cols)
	sh := append(slices.Clone(x.Shape[:len(x.Shape)-1]), rows)
	b := bias
	if scale != nil {
		b = make([]float32, rows)
		for i := range b {
			b[i] = bias[i] / scale[i]
		}
	}
	y := m.Op("linear", map[string]any{"x": x, "weight": wv, "bias": m.F16s(name+"_bias", []int{rows}, b)}, sh...)
	if scale != nil {
		y = m.Op("mul", map[string]any{"x": y, "y": m.F16s(name+"_scale", []int{1, 1, rows}, scale)}, sh...)
	}
	return y
}

// expertMatrix is an expert's weight w (rows, cols) as a constant of the graph g (whose Opts give
// the format and the column blocks of the tables). In lut4 or lut8 its tables are fitted exactly
// (exactTables) unless the options choose a fitting (Quant, Codes) or the values are too many,
// and with expert-tables exact-rows the rows' powers of two are taken out first: they are
// returned for the output to be multiplied by (nil for none).
func (m *builder) expertMatrix(g *blocks.Graph, name string, w []float32, rows, cols int) (*mlpackage.Var, []float32) {
	bits := map[string]int{"lut4": 4, "lut8": 8}[g.Opts.FormatOf(name)]
	mode := Arch.Param(g.Opts, "expert-tables")
	if bits == 0 || mode == "palette" || g.Opts.Quant != "" || g.Opts.Codes != "" {
		return g.Matrix(name, w, rows, cols), nil
	}
	var scale []float32
	v := w
	if mode == "exact-rows" {
		scale = rowScales(w, rows, cols)
		v = make([]float32, len(w))
		for r := 0; r < rows; r++ {
			for j := 0; j < cols; j++ {
				v[r*cols+j] = w[r*cols+j] / scale[r]
			}
		}
	} else if mode != "exact" {
		m.Fail("expert-tables %q is not exact-rows, exact or palette", mode)
	}
	colBlock := g.Opts.ColBlock
	indices, lut, ok := exactTables(v, rows, cols, g.Opts.Group, colBlock, bits)
	if !ok {
		return g.Matrix(name, w, rows, cols), nil
	}
	if colBlock == 0 {
		colBlock = cols
	}
	c, err := m.B.LUT(name, bits, []int{rows, cols}, []int{g.Opts.Group, colBlock}, indices, lut)
	if err != nil {
		m.Fail("%v", err)
	}
	return c, scale
}

// groupOf is how many experts share a down projection: the most of 4, 2 and 1 that divides the
// experts and keeps the group's intermediate values within the Neural Engine's 16384 per axis.
func groupOf(experts, inter int) int {
	for _, n := range []int{4, 2} {
		if experts%n == 0 && n*inter <= 16384 {
			return n
		}
	}
	return 1
}

// glu is GptOssExperts' activation of an expert's gate and up: (up + 1) * gate * sigmoid(1.702
// gate), the gate clamped above at swiglu_limit and the up to [-limit, limit]; the sigmoid written
// with exp and real_div (the Neural Engine's sigmoid drifts, as its SiLU: blocks.SiLU).
func (m *builder) glu(gate, up *mlpackage.Var) *mlpackage.Var {
	lim := float32(m.cfg.SwigluLimit)
	sh := gate.Shape
	gate = m.Op("minimum", map[string]any{"x": gate, "y": m.B.ScalarF16(lim)}, sh...)
	up = m.Op("clip", map[string]any{"x": up, "alpha": m.B.ScalarF16(-lim), "beta": m.B.ScalarF16(lim)}, sh...)
	neg := m.Op("mul", map[string]any{"x": gate, "y": m.B.ScalarF16(-alpha)}, sh...)
	den := m.Op("add", map[string]any{"x": m.Op("exp", map[string]any{"x": neg}, sh...), "y": m.B.ScalarF16(1)}, sh...)
	g := m.Op("real_div", map[string]any{"x": gate, "y": den}, sh...)
	u := m.Op("add", map[string]any{"x": up, "y": m.B.ScalarF16(1)}, sh...)
	return m.Op("mul", map[string]any{"x": u, "y": g}, sh...)
}

// Selecting the top logits without comparisons (the lowering has none): at each of the k rounds,
// t = min(max - logit, routeSpan) is 0 at the largest remaining logit and at least the gap to it
// elsewhere, and relu(1 - t routeScale^2) is 1 there and 0 wherever the gap is 2^-16 or more (the
// spacing of fp16's values from 2^-6 on: distinct logits of that size are never mixed). The
// chosen logit is then pushed down by routeDrop out of the next rounds' way. routeSpan keeps
// t routeScale^2 within fp16; the scale is two multiplies, 65536 being past fp16's largest value.
const (
	routeSpan  = 1.0 / 16
	routeScale = 256
	routeDrop  = 1000
)

// route is GptOssTopKRouter's weights of the experts from the router's logits r (1, seq,
// experts): the top k logits' softmax at their experts, zero elsewhere (torch.topk, then a
// softmax over the k, scattered).
func (m *builder) route(r *mlpackage.Var) *mlpackage.Var {
	sh := r.Shape
	one := []int{sh[0], sh[1], 1}
	top := func(x *mlpackage.Var) *mlpackage.Var {
		return m.Op("reduce_max", map[string]any{"x": x, "axes": m.B.Ints(-1), "keep_dims": m.B.ScalarBool(true)}, one...)
	}
	m1 := top(r)
	cur := r
	var sel *mlpackage.Var
	for k := 0; k < m.cfg.TopK; k++ {
		mk := m1
		if k > 0 {
			mk = top(cur)
		}
		t := m.Op("sub", map[string]any{"x": mk, "y": cur}, sh...)
		t = m.Op("minimum", map[string]any{"x": t, "y": m.B.ScalarF16(routeSpan)}, sh...)
		t = m.Op("mul", map[string]any{"x": t, "y": m.B.ScalarF16(-routeScale)}, sh...)
		t = m.Op("mul", map[string]any{"x": t, "y": m.B.ScalarF16(routeScale)}, sh...)
		ind := m.Op("relu", map[string]any{"x": m.Op("add", map[string]any{"x": t, "y": m.B.ScalarF16(1)}, sh...)}, sh...)
		if sel == nil {
			sel = ind
		} else {
			sel = m.Op("add", map[string]any{"x": sel, "y": ind}, sh...)
		}
		if k+1 < m.cfg.TopK {
			cur = m.Op("sub", map[string]any{"x": cur, "y": m.Op("mul", map[string]any{"x": ind, "y": m.B.ScalarF16(routeDrop)}, sh...)}, sh...)
		}
	}
	e := m.Op("exp", map[string]any{"x": m.Op("sub", map[string]any{"x": r, "y": m1}, sh...)}, sh...)
	e = m.Op("mul", map[string]any{"x": e, "y": sel}, sh...)
	sum := m.Op("reduce_sum", map[string]any{"x": e, "axes": m.B.Ints(-1), "keep_dims": m.B.ScalarBool(true)}, one...)
	return m.Op("real_div", map[string]any{"x": e, "y": sum}, sh...)
}

// layerType names the inputs of a kind of layer: "full" or "sliding".
func layerType(full bool) string {
	if full {
		return "full"
	}
	return "sliding"
}
