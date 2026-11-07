package whisper

import (
	"fmt"
	"math"
	"strconv"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// EncoderArch registers the audio encoder with the whitelist; it is chosen with -arch on the
// checkpoint's config.json, whose model_type selects the decoder (Arch).
var EncoderArch = &llm.Arch{
	Name:  "whisper-encoder",
	Title: "Whisper audio encoder",
	Params: []llm.Param{
		{Name: "qchunk", Default: "500", Usage: "queries per attention of the encoder's layers (0: all 1500 at once); the scores of 20 heads over 1500 keys are 90 MB at once, over the 64 MiB a tensor past which the Neural Engine slows down"},
	},
	// 8 bit tables, as the other encoders (bert, plamo-bi): to be measured
	Format:     "lut8",
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseEncoderConfig(raw) },
}

func init() { llm.Register(EncoderArch) }

// Encoder is the audio encoder of a Whisper checkpoint (llm.Model).
type Encoder struct{ *Config }

// ParseEncoderConfig parses config.json for the audio encoder.
func ParseEncoderConfig(raw []byte) (*Encoder, error) {
	c, err := ParseConfig(raw)
	if err != nil {
		return nil, err
	}
	return &Encoder{c}, nil
}

// Hidden is the encoder's width, of the hidden state between layers. The input of a program that
// starts at the first layer is the mel frames instead (MelBins wide).
func (e *Encoder) Hidden() int { return e.DModel }

// Layers is the number of encoder layers.
func (e *Encoder) Layers() int { return e.EncoderLayers }

// Output is the encoder's last hidden state, after its final LayerNorm.
func (e *Encoder) Output() (int, bool) { return e.DModel, false }

// Embedding is empty: the input is mel frames.
func (e *Encoder) Embedding() string { return "" }

// Tensors lists the tensors the encoder reads. The decoder's are the decoder program's (Arch),
// unused here.
func (e *Encoder) Tensors() map[string][]int {
	d, f, p := e.DModel, e.EncoderFFN, e.encoder()
	t := map[string][]int{
		p + "conv1.weight":           {d, e.MelBins, 3},
		p + "conv1.bias":             {d},
		p + "conv2.weight":           {d, d, 3},
		p + "conv2.bias":             {d},
		p + "embed_positions.weight": {e.SourcePositions, d},
	}
	norm(t, p+"layer_norm", d)
	for i := 0; i < e.EncoderLayers; i++ {
		l := e.encoderLayer(i)
		attentionTensors(t, l+"self_attn.", d)
		norm(t, l+"self_attn_layer_norm", d)
		norm(t, l+"final_layer_norm", d)
		t[l+"fc1.weight"] = []int{f, d}
		t[l+"fc1.bias"] = []int{f}
		t[l+"fc2.weight"] = []int{d, f}
		t[l+"fc2.bias"] = []int{d}
	}
	return t
}

// QChunk parses the qchunk parameter: the queries per attention, at most the positions.
func QChunk(param string, positions int) (int, error) {
	n, err := strconv.Atoi(param)
	if err != nil || n < 0 {
		return 0, fmt.Errorf("whisper: qchunk %q is not a number of queries", param)
	}
	if n == 0 || n > positions {
		n = positions
	}
	return n, nil
}

type encoder struct {
	*blocks.Graph
	cfg    *Config
	n      int // positions
	d      int
	qchunk int
}

// Build writes the encoder for 30 seconds of audio; opts.Seq is not used. When the program starts
// at the first layer its input x is the log mel frames (1, Frames, mel bins), time major, as
// Features computes them; otherwise x is the hidden state (1, positions, d_model). The output y
// is the last hidden state (1, positions, d_model), after the final LayerNorm, or the hidden state
// after the last built layer of a truncated program (and with NoHead).
func (e *Encoder) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	c := e.Config
	if opts.Cache > 0 {
		return nil, out, fmt.Errorf("whisper: the audio encoder has no KV cache")
	}
	qchunk, err := QChunk(EncoderArch.Param(opts, "qchunk"), c.SourcePositions)
	if err != nil {
		return nil, out, err
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = EncoderArch.Format
	}
	nl := c.EncoderLayers
	if opts.Layers <= 0 || opts.Start+opts.Layers > nl {
		opts.Layers = nl - opts.Start
	}
	m := &encoder{Graph: &blocks.Graph{Name: "whisper", B: b, W: w, Opts: opts}, cfg: c, n: c.SourcePositions, d: c.DModel, qchunk: qchunk}
	err = blocks.Run(func() {
		var x, h *mlpackage.Var
		if opts.Start == 0 {
			x = b.Input("x", mlpackage.Float16, 1, c.Frames(), c.MelBins)
			h = m.embed(x)
		} else {
			x = b.Input("x", mlpackage.Float16, 1, m.n, m.d)
			h = x
		}
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		for i := opts.Start; i < opts.Start+opts.Layers; i++ {
			h = m.layer(i, h)
		}
		if opts.Start+opts.Layers == nl && !opts.NoHead {
			p := c.encoder() + "layer_norm"
			h = m.LayerNorm(h, p+".weight", p+".bias", m.d, layerNormEps)
		}
		y := b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		out = mlpackage.Feature{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}
	})
	return ins, out, err
}

// taps lays the three frames a kernel 3 convolution reads side by side, (1, T, 3 C) from x
// (1, T, C) padded by a frame of zeros on each side: frames t-1, t and t+1 for each t, or with
// stride 2 frames 2t-1, 2t and 2t+1 for each t < T/2 (T even).
func (m *encoder) taps(x *mlpackage.Var, stride int) *mlpackage.Var {
	t, ch := x.Shape[1], x.Shape[2]
	padded := m.Op("pad", map[string]any{"x": x, "pad": m.B.Ints(0, 0, 1, 1, 0, 0), "mode": m.Str("constant"), "constant_val": m.B.ScalarF16(0)}, 1, t+2, ch)
	rows := func(begin, n int) *mlpackage.Var {
		return m.Op("slice_by_index", map[string]any{"x": padded, "begin": m.B.Ints(0, int32(begin), 0), "end": m.B.Ints(1, int32(begin+n), int32(ch)),
			"begin_mask": m.Bools(true, false, true), "end_mask": m.Bools(true, false, true)}, 1, n, ch)
	}
	if stride == 1 {
		return m.Op("concat", map[string]any{"values": []*mlpackage.Var{rows(0, t), rows(1, t), rows(2, t)}, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, t, 3*ch)
	}
	if t%2 != 0 {
		m.Fail("a stride 2 convolution over %d frames", t)
	}
	o := t / 2
	// padded rows 2t' and 2t'+1 side by side, then 2t'+2 (the first half of the pairs from row 2)
	pairs := m.Op("reshape", map[string]any{"x": rows(0, t), "shape": m.B.Ints(1, int32(o), int32(2*ch))}, 1, o, 2*ch)
	next := m.Op("reshape", map[string]any{"x": rows(2, t), "shape": m.B.Ints(1, int32(o), int32(2*ch))}, 1, o, 2*ch)
	next = m.Op("slice_by_index", map[string]any{"x": next, "begin": m.B.Ints(0, 0, 0), "end": m.B.Ints(1, int32(o), int32(ch)),
		"begin_mask": m.Bools(true, true, true), "end_mask": m.Bools(true, true, false)}, 1, o, ch)
	return m.Op("concat", map[string]any{"values": []*mlpackage.Var{pairs, next}, "axis": m.B.ScalarI32(2), "interleave": m.B.ScalarBool(false)}, 1, o, 3*ch)
}

// conv is a Conv1d of kernel 3 and padding 1 (weight (out, in, 3), bias) over x (1, T, in), as a
// linear over its taps: the weight laid out (out, 3 in), tap major.
func (m *encoder) conv(x *mlpackage.Var, name string, stride int) *mlpackage.Var {
	in := x.Shape[2]
	w := m.Get(name+".weight", m.d, in, 3)
	mat := make([]float32, m.d*3*in)
	for o := 0; o < m.d; o++ {
		for i := 0; i < in; i++ {
			for j := 0; j < 3; j++ {
				mat[(o*3+j)*in+i] = w[(o*in+i)*3+j]
			}
		}
	}
	tp := m.taps(x, stride)
	wv := m.F16s(name+".taps", []int{m.d, 3 * in}, mat)
	bv := m.F16s(name+".bias", []int{m.d}, m.Get(name+".bias", m.d))
	return m.Op("linear", map[string]any{"x": tp, "weight": wv, "bias": bv}, 1, tp.Shape[1], m.d)
}

// embed is WhisperEncoder before its layers: conv1 and GELU, conv2 (stride 2) and GELU, and the
// position embeddings.
func (m *encoder) embed(x *mlpackage.Var) *mlpackage.Var {
	p := m.cfg.encoder()
	h := m.Op("gelu", map[string]any{"x": m.conv(x, p+"conv1", 1), "mode": m.Str("EXACT")}, 1, x.Shape[1], m.d)
	h = m.Op("gelu", map[string]any{"x": m.conv(h, p+"conv2", 2), "mode": m.Str("EXACT")}, 1, m.n, m.d)
	pos := m.F16s(p+"embed_positions.weight", []int{1, m.n, m.d}, m.Get(p+"embed_positions.weight", m.n, m.d))
	return m.Op("add", map[string]any{"x": h, "y": pos}, 1, m.n, m.d)
}

// layer is WhisperEncoderLayer: pre norm self attention and MLP, each with a residual.
func (m *encoder) layer(i int, h *mlpackage.Var) *mlpackage.Var {
	p := m.cfg.encoderLayer(i)
	a := m.LayerNorm(h, p+"self_attn_layer_norm.weight", p+"self_attn_layer_norm.bias", m.d, layerNormEps)
	a = m.attention(p+"self_attn.", a)
	h = m.Op("add", map[string]any{"x": h, "y": a}, h.Shape...)
	f := m.LayerNorm(h, p+"final_layer_norm.weight", p+"final_layer_norm.bias", m.d, layerNormEps)
	f = m.LinearBias(f, p+"fc1.weight", p+"fc1.bias", "", m.cfg.EncoderFFN)
	f = m.Activation(f, m.cfg.Activation)
	f = m.LinearBias(f, p+"fc2.weight", p+"fc2.bias", "", m.d)
	return m.Op("add", map[string]any{"x": h, "y": f}, h.Shape...)
}

// attention is WhisperAttention over the positions (no mask): the queries scaled by
// 1/sqrt(head_dim) before the scores, as transformers does, and computed qchunk queries at a time.
func (m *encoder) attention(p string, x *mlpackage.Var) *mlpackage.Var {
	return selfAttention(m.Graph, p, x, m.cfg.EncoderHeads, m.qchunk)
}

// heads projects x (1, s, d) and splits it into heads (1, heads, s, d/heads).
func heads(g *blocks.Graph, x *mlpackage.Var, weight, bias string, nh int) *mlpackage.Var {
	s, d := x.Shape[1], x.Shape[2]
	hd := d / nh
	var t *mlpackage.Var
	if bias == "" {
		t = g.Linear(x, weight, "", d)
	} else {
		t = g.LinearBias(x, weight, bias, "", d)
	}
	t = g.Op("reshape", map[string]any{"x": t, "shape": g.B.Ints(1, int32(s), int32(nh), int32(hd))}, 1, s, nh, hd)
	return g.Op("transpose", map[string]any{"x": t, "perm": g.B.Ints(0, 2, 1, 3)}, 1, nh, s, hd)
}

// merge joins the heads of o (1, heads, s, hd) into (1, s, heads hd) and applies out_proj.
func merge(g *blocks.Graph, o *mlpackage.Var, p string) *mlpackage.Var {
	nh, s, hd := o.Shape[1], o.Shape[2], o.Shape[3]
	o = g.Op("transpose", map[string]any{"x": o, "perm": g.B.Ints(0, 2, 1, 3)}, 1, s, nh, hd)
	o = g.Op("reshape", map[string]any{"x": o, "shape": g.B.Ints(1, int32(s), int32(nh*hd))}, 1, s, nh*hd)
	return g.LinearBias(o, p+"out_proj.weight", p+"out_proj.bias", "", nh*hd)
}

// scaled multiplies the queries by 1/sqrt(head_dim).
func scaled(g *blocks.Graph, q *mlpackage.Var) *mlpackage.Var {
	return g.Op("mul", map[string]any{"x": q, "y": g.B.ScalarF16(float32(1 / math.Sqrt(float64(q.Shape[3]))))}, q.Shape...)
}

// selfAttention is an unmasked WhisperAttention of x with itself, qchunk queries at a time.
func selfAttention(g *blocks.Graph, p string, x *mlpackage.Var, nh, qchunk int) *mlpackage.Var {
	q := scaled(g, heads(g, x, p+"q_proj.weight", p+"q_proj.bias", nh))
	k := heads(g, x, p+"k_proj.weight", "", nh)
	v := heads(g, x, p+"v_proj.weight", p+"v_proj.bias", nh)
	s, hd := q.Shape[2], q.Shape[3]
	var parts []*mlpackage.Var
	for begin := 0; begin < s; begin += qchunk {
		n := min(qchunk, s-begin)
		qc := q
		if n < s {
			qc = g.Op("slice_by_index", map[string]any{"x": q, "begin": g.B.Ints(0, 0, int32(begin), 0), "end": g.B.Ints(1, int32(nh), int32(begin+n), int32(hd)),
				"begin_mask": g.Bools(true, true, false, true), "end_mask": g.Bools(true, true, false, true)}, 1, nh, n, hd)
		}
		parts = append(parts, g.SDPAScale(qc, k, v, 1, nil))
	}
	o := parts[0]
	if len(parts) > 1 {
		o = g.Op("concat", map[string]any{"values": parts, "axis": g.B.ScalarI32(2), "interleave": g.B.ScalarBool(false)}, 1, nh, s, hd)
	}
	return merge(g, o, p)
}
