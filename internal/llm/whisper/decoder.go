package whisper

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch registers the text decoder with the whitelist: what model_type whisper selects.
var Arch = &llm.Arch{
	Name:          "whisper",
	Title:         "Whisper text decoder",
	ModelTypes:    []string{"whisper"},
	Architectures: []string{"WhisperForConditionalGeneration", "WhisperModel"},
	// 8 bit tables: to be measured
	Format:     "lut8",
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseDecoderConfig(raw) },
}

func init() { llm.Register(Arch) }

// Decoder is the text decoder of a Whisper checkpoint (llm.Cached).
type Decoder struct{ *Config }

// ParseDecoderConfig parses config.json for the text decoder.
func ParseDecoderConfig(raw []byte) (*Decoder, error) {
	c, err := ParseConfig(raw)
	if err != nil {
		return nil, err
	}
	return &Decoder{c}, nil
}

// Hidden is the decoder's width.
func (d *Decoder) Hidden() int { return d.DModel }

// Layers is the number of decoder layers.
func (d *Decoder) Layers() int { return d.DecoderLayers }

// Output is the logits.
func (d *Decoder) Output() (int, bool) { return d.VocabSize, true }

// Embedding names the token embedding table, which is the output layer too (proj_out is tied to
// it).
func (d *Decoder) Embedding() string { return d.decoder() + "embed_tokens.weight" }

// Tensors lists the tensors the decoder reads. The encoder's are the encoder program's
// (EncoderArch), and the cross attention's key and value projections the cross program's
// (CrossArch): they turn the encoder's output into the inputs xk_N and xv_N once per clip.
func (d *Decoder) Tensors() map[string][]int {
	dm, f, p := d.DModel, d.DecoderFFN, d.decoder()
	t := map[string][]int{
		d.Embedding():                {d.VocabSize, dm},
		p + "embed_positions.weight": {d.TargetPositions, dm},
	}
	norm(t, p+"layer_norm", dm)
	for i := 0; i < d.DecoderLayers; i++ {
		l := d.decoderLayer(i)
		attentionTensors(t, l+"self_attn.", dm)
		for _, n := range []string{"q_proj", "out_proj"} {
			t[l+"encoder_attn."+n+".weight"] = []int{dm, dm}
			t[l+"encoder_attn."+n+".bias"] = []int{dm}
		}
		norm(t, l+"self_attn_layer_norm", dm)
		norm(t, l+"encoder_attn_layer_norm", dm)
		norm(t, l+"final_layer_norm", dm)
		t[l+"fc1.weight"] = []int{f, dm}
		t[l+"fc1.bias"] = []int{f}
		t[l+"fc2.weight"] = []int{dm, f}
		t[l+"fc2.bias"] = []int{dm}
	}
	return t
}

func (d *Decoder) headDim() int { return d.DModel / d.DecoderHeads }

type decoder struct {
	*blocks.Graph
	cfg     *Config
	seq     int
	d       int
	opts    llm.Options
	ins     []mlpackage.Feature
	outs    []mlpackage.Feature
	mask    *mlpackage.Var
	inputs  map[string]*mlpackage.Var
	outList []*mlpackage.Var
}

// input declares a program input once and returns it.
func (m *decoder) input(name string, shape ...int) *mlpackage.Var {
	if v, ok := m.inputs[name]; ok {
		return v
	}
	v := m.B.Input(name, mlpackage.Float16, shape...)
	m.inputs[name] = v
	m.ins = append(m.ins, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: shape})
	return v
}

// output makes v an output of the program under name.
func (m *decoder) output(v *mlpackage.Var, name string) {
	o := m.B.NamedOp("identity", name, map[string]any{"x": v}, mlpackage.Float16, v.Shape...)
	m.outList = append(m.outList, o)
}

// Build writes the decoder without a KV cache, for opts.Seq tokens from position 0 (what
// hwnpu-dev check runs). Its inputs are x, the token embeddings (1, seq, d_model), and for each built
// layer N xk_N and xv_N, the cross attention's keys and values of the clip (1, heads, 1500,
// head_dim) that the cross program computes; its output y is the logits (1, seq, vocab), or the
// hidden state after the last built layer of a truncated program.
func (d *Decoder) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Cache > 0 {
		return nil, out, fmt.Errorf("whisper: a decoder with a KV cache is BuildCached's")
	}
	ins, outs, err := d.build(b, w, opts)
	if err != nil {
		return nil, out, err
	}
	return ins, outs[0], nil
}

// BuildCached writes the decoder for opts.Seq tokens per step with a KV cache of opts.Cache
// positions (llm.Cached): besides x and xk_N, xv_N its inputs are pos (1, seq, max_target_positions),
// the one-hot positions of the step's tokens, mask (1, 1, seq, cache + seq), and each layer's
// cache kv_k_N, kv_v_N (1, heads, cache, head_dim), whose outputs kv_k_N_new, kv_v_N_new are the
// step's keys and values.
func (d *Decoder) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	if opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("whisper: BuildCached needs a KV cache")
	}
	return d.build(b, w, opts)
}

func (d *Decoder) build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	c := d.Config
	if opts.Seq <= 0 {
		return nil, nil, fmt.Errorf("whisper: no tokens per call")
	}
	if opts.Cache > c.TargetPositions || opts.Cache == 0 && opts.Seq > c.TargetPositions {
		return nil, nil, fmt.Errorf("whisper: %d positions over the decoder's %d", max(opts.Cache, opts.Seq), c.TargetPositions)
	}
	if opts.Paged > 0 || opts.LastRow {
		return nil, nil, fmt.Errorf("whisper: the decoder is not split at its attention, nor has it the logits of the last row alone")
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = Arch.Format
	}
	nl := c.DecoderLayers
	if opts.Layers <= 0 || opts.Start+opts.Layers > nl {
		opts.Layers = nl - opts.Start
	}
	m := &decoder{Graph: &blocks.Graph{Name: "whisper", B: b, W: w, Opts: opts}, cfg: c, seq: opts.Seq, d: c.DModel, opts: opts, inputs: map[string]*mlpackage.Var{}}
	err = blocks.Run(func() {
		h := m.input("x", 1, m.seq, m.d)
		if opts.Start == 0 {
			h = m.embed(h)
		}
		for i := opts.Start; i < opts.Start+opts.Layers; i++ {
			h = m.layer(i, h)
		}
		var y *mlpackage.Var
		if opts.Start+opts.Layers < nl || opts.NoHead {
			y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		} else {
			p := c.decoder() + "layer_norm"
			h = m.LayerNorm(h, p+".weight", p+".bias", m.d, layerNormEps)
			y = m.LinearPadded(h, c.decoder()+"embed_tokens.weight", "y", c.VocabSize)
		}
		m.outs = append([]mlpackage.Feature{{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}}, m.outs...)
		for _, v := range m.outList {
			m.outs = append(m.outs, mlpackage.Feature{Name: v.Name, DType: mlpackage.Float16, Shape: v.Shape})
		}
	})
	return m.ins, m.outs, err
}

// embed is WhisperDecoder before its layers: the token embeddings (scaled by sqrt(d_model) with
// scale_embedding) plus the position embeddings, of positions 0.. without a cache (constants) and
// of the step's positions with one, picked by the one-hot input pos (gather runs on the CPU).
func (m *decoder) embed(x *mlpackage.Var) *mlpackage.Var {
	c, p := m.cfg, m.cfg.decoder()+"embed_positions.weight"
	if c.ScaleEmbedding {
		x = m.Op("mul", map[string]any{"x": x, "y": m.B.ScalarF16(float32(math.Sqrt(float64(m.d))))}, x.Shape...)
	}
	n := c.TargetPositions
	table := m.Get(p, n, m.d)
	var pos *mlpackage.Var
	if m.opts.Cache > 0 {
		sel := m.input("pos", 1, m.seq, n)
		t := m.F16s(p, []int{1, n, m.d}, table)
		pos = m.Op("matmul", map[string]any{"x": sel, "y": t, "transpose_x": m.B.ScalarBool(false), "transpose_y": m.B.ScalarBool(false)}, 1, m.seq, m.d)
	} else {
		pos = m.F16s(p, []int{1, m.seq, m.d}, table[:m.seq*m.d])
	}
	return m.Op("add", map[string]any{"x": x, "y": pos}, x.Shape...)
}

// layer is WhisperDecoderLayer: pre norm causal self attention, cross attention over the clip
// and the MLP, each with a residual.
func (m *decoder) layer(i int, h *mlpackage.Var) *mlpackage.Var {
	p, c := m.cfg.decoderLayer(i), m.cfg
	a := m.LayerNorm(h, p+"self_attn_layer_norm.weight", p+"self_attn_layer_norm.bias", m.d, layerNormEps)
	h = m.Op("add", map[string]any{"x": h, "y": m.selfAttention(i, a)}, h.Shape...)
	a = m.LayerNorm(h, p+"encoder_attn_layer_norm.weight", p+"encoder_attn_layer_norm.bias", m.d, layerNormEps)
	h = m.Op("add", map[string]any{"x": h, "y": m.crossAttention(i, a)}, h.Shape...)
	f := m.LayerNorm(h, p+"final_layer_norm.weight", p+"final_layer_norm.bias", m.d, layerNormEps)
	f = m.LinearBias(f, p+"fc1.weight", p+"fc1.bias", "", c.DecoderFFN)
	f = m.Activation(f, c.Activation)
	f = m.LinearBias(f, p+"fc2.weight", p+"fc2.bias", "", m.d)
	return m.Op("add", map[string]any{"x": h, "y": f}, h.Shape...)
}

// selfAttention is the decoder's causal self attention; with a KV cache, over the cached keys and
// values followed by the step's, which are outputs.
func (m *decoder) selfAttention(i int, x *mlpackage.Var) *mlpackage.Var {
	p := m.cfg.decoderLayer(i) + "self_attn."
	nh := m.cfg.DecoderHeads
	hd := m.d / nh
	q := scaled(m.Graph, heads(m.Graph, x, p+"q_proj.weight", p+"q_proj.bias", nh))
	k := heads(m.Graph, x, p+"k_proj.weight", "", nh)
	v := heads(m.Graph, x, p+"v_proj.weight", p+"v_proj.bias", nh)
	keys := m.seq
	if t := m.opts.Cache; t > 0 {
		m.output(k, fmt.Sprintf("kv_k_%d_new", i))
		m.output(v, fmt.Sprintf("kv_v_%d_new", i))
		cat := func(name string, x *mlpackage.Var) *mlpackage.Var {
			in := m.input(name, 1, nh, t, hd)
			return m.Op("concat", map[string]any{"values": []*mlpackage.Var{in, x}, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, nh, t+m.seq, hd)
		}
		k, v = cat(fmt.Sprintf("kv_k_%d", i), k), cat(fmt.Sprintf("kv_v_%d", i), v)
		keys += t
	}
	o := m.SDPAScale(q, k, v, 1, func() *mlpackage.Var { return m.causal(keys) })
	return merge(m.Graph, o, p)
}

// causal is the self attention's additive mask over keys: the input mask with a KV cache
// (StepInputs), else a constant one.
func (m *decoder) causal(keys int) *mlpackage.Var {
	if m.mask == nil {
		if m.opts.Cache > 0 {
			m.mask = m.input("mask", 1, 1, m.seq, keys)
		} else {
			m.mask = m.CausalMask("mask", m.seq, 0)
		}
	}
	return m.mask
}

// crossAttention is the decoder layer's attention over the clip: its queries from x, its keys
// and values the inputs xk_N and xv_N (the cross program's), every position seen.
func (m *decoder) crossAttention(i int, x *mlpackage.Var) *mlpackage.Var {
	p := m.cfg.decoderLayer(i) + "encoder_attn."
	nh, n := m.cfg.DecoderHeads, m.cfg.SourcePositions
	hd := m.d / nh
	q := scaled(m.Graph, heads(m.Graph, x, p+"q_proj.weight", p+"q_proj.bias", nh))
	k := m.input(fmt.Sprintf("xk_%d", i), 1, nh, n, hd)
	v := m.input(fmt.Sprintf("xv_%d", i), 1, nh, n, hd)
	return merge(m.Graph, m.SDPAScale(q, k, v, 1, nil), p)
}

// StepInputs gives a step's pos, the one-hot positions pos.. of its q tokens (the padding's
// clamped to the last position), and mask: a step's token t sees the cache's first pos positions
// and the step's tokens up to itself.
func (d *Decoder) StepInputs(pos, valid, q, cache int) map[string][]float32 {
	n := d.TargetPositions
	sel := make([]float32, q*n)
	for t := 0; t < q; t++ {
		sel[t*n+min(pos+t, n-1)] = 1
	}
	keys := cache + q
	mask := make([]float32, q*keys)
	for t := 0; t < q; t++ {
		for j := 0; j < keys; j++ {
			visible := j < pos || j >= cache && j-cache <= t
			if !visible {
				mask[t*keys+j] = -1e4 // as blocks.CausalMask: fp16 safe
			}
		}
	}
	return map[string][]float32{"pos": sel, "mask": mask}
}
