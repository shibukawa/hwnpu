// Package lfm3audio converts the audio encoder of Liquid AI's LFM3-Audio (model_type lfm3_audio)
// and LFM3-Omni (lfm3_omni) into a Core ML program for clips of up to a fixed number of mel
// frames. The architecture is lfm3-audio-encoder, chosen with -arch on the checkpoint's
// config.json (whose text model is internal/llm/lfm3).
//
// The structure is llama.cpp's tools/mtmd/models/lfm3a.cpp (a clip encoded from its start, with
// no streaming state): a downsampler of three stride 2 convolutions over (time, mel) that divides
// the frames by 8 (a 3x3 convolution, then twice a depthwise 3x3 and a pointwise one, each
// followed by SiLU), a linear to the encoder's width, a causal LFM2 stack (gated short
// convolutions and attention over a sliding window, with QK norm and RoPE), the final RMS norm,
// and a projector (LayerNorm, linear, GELU, linear) to the text model's width.
package lfm3audio

import (
	"cmp"
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch registers the audio encoder with the whitelist; it is chosen with -arch.
var Arch = &llm.Arch{
	Name:  "lfm3-audio-encoder",
	Title: "LFM3 audio encoder",
	Params: []llm.Param{
		{Name: "frames", Default: "3000", Usage: "mel frames (10 ms each) of the audio encoder's program: the longest clip it takes"},
	},
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseConfig(raw) },
}

func init() { llm.Register(Arch) }

// LFM2 is the LFM2 config of the encoder's stack.
type LFM2 struct {
	HiddenSize       int      `json:"hidden_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"head_dim"`
	IntermediateSize int      `json:"intermediate_size"`
	BlockFFDim       *int     `json:"block_ff_dim"`
	AutoAdjust       bool     `json:"block_auto_adjust_ff_dim"`
	Multiplier       *float64 `json:"block_ffn_dim_multiplier"`
	MultipleOf       int      `json:"block_multiple_of"`
	NormEps          float64  `json:"norm_eps"`
	ConvKernel       int      `json:"conv_L_cache"`
	ConvBias         bool     `json:"conv_bias"`
	LayerTypes       []string `json:"layer_types"`
	RopeTheta        float64  `json:"rope_theta"`
	RopeParameters   struct {
		Theta float64 `json:"rope_theta"`
	} `json:"rope_parameters"`
	// SubsamplingChannels are the downsampler's channels (NeMo's subsampling_conv_channels).
	SubsamplingChannels int `json:"subsampling_conv_channels"`
}

// FFDim is the SwiGLU's hidden width, as LFM2 computes it (llama.cpp's lfm2_feed_forward_length).
func (c *LFM2) FFDim() int {
	ff := c.IntermediateSize
	if c.BlockFFDim != nil {
		ff = *c.BlockFFDim
	}
	if c.AutoAdjust {
		ff = 2 * ff / 3
		if c.Multiplier != nil {
			ff = int(*c.Multiplier * float64(ff))
		}
		m := max(c.MultipleOf, 1)
		ff = m * ((ff + m - 1) / m)
	}
	return ff
}

// Config is what the conversion needs of an LFM3-Audio or LFM3-Omni config.json.
type Config struct {
	Omni       bool
	Enc        LFM2
	MelBins    int
	Window     int // the sliding window of the encoder's attention, in positions (keys)
	TextHidden int
	// ProjectorHidden is the projector's hidden width (adapter_hidden_size, else the text model's).
	ProjectorHidden int

	down, stack, norm, lin1, lin2 string // tensor name prefixes
}

// ParseConfig parses an LFM3-Audio or LFM3-Omni config.json and rejects encoder settings the
// conversion does not handle (those llama.cpp's graph does not handle either).
func ParseConfig(raw []byte) (*Config, error) {
	var top struct {
		ModelType string `json:"model_type"`
		Encoder   *struct {
			LFM     json.RawMessage `json:"lfm"`
			DimIn   int             `json:"dim_in"`
			Window  int             `json:"sliding_window"`
			Adapter int             `json:"adapter_hidden_size"`
		} `json:"encoder"`
		AudioEncoder *struct {
			Text    json.RawMessage `json:"text_config"`
			MelBins int             `json:"num_mel_bins"`
			Window  int             `json:"sliding_window"`
		} `json:"audio_encoder_config"`
		LFM struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"lfm"`
		TextConfig struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"text_config"`
		Adapter int `json:"adapter_hidden_size"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("lfm3audio: %w", err)
	}
	c := &Config{}
	var enc json.RawMessage
	switch {
	case top.ModelType == "lfm3_audio" && top.Encoder != nil:
		enc, c.MelBins, c.Window, c.TextHidden = top.Encoder.LFM, top.Encoder.DimIn, top.Encoder.Window, top.LFM.HiddenSize
		c.ProjectorHidden = cmp.Or(top.Encoder.Adapter, top.Adapter)
		c.down, c.stack, c.norm, c.lin1, c.lin2 = "encoder.downsample.", "encoder.lfm.lfm.", "mlp.model.0.", "mlp.model.1.", "mlp.model.3."
	case top.ModelType == "lfm3_omni" && top.AudioEncoder != nil:
		c.Omni = true
		enc, c.MelBins, c.Window, c.TextHidden = top.AudioEncoder.Text, top.AudioEncoder.MelBins, top.AudioEncoder.Window, top.TextConfig.HiddenSize
		c.ProjectorHidden = top.Adapter
		c.down, c.stack = "model.audio_encoder.downsample.", "model.audio_encoder.lfm."
		c.norm, c.lin1, c.lin2 = "model.audio_projector.layer_norm.", "model.audio_projector.linear_1.", "model.audio_projector.linear_2."
	default:
		return nil, fmt.Errorf("lfm3audio: model_type %q has no audio encoder (lfm3_audio with encoder, or lfm3_omni with audio_encoder_config)", top.ModelType)
	}
	if err := json.Unmarshal(enc, &c.Enc); err != nil {
		return nil, fmt.Errorf("lfm3audio: encoder: %w", err)
	}
	e := &c.Enc
	e.NumKVHeads = cmp.Or(e.NumKVHeads, e.NumHeads)
	if e.NumHeads == 0 || e.HiddenSize == 0 || c.TextHidden == 0 {
		return nil, fmt.Errorf("lfm3audio: the encoder needs hidden_size and num_attention_heads, the text model hidden_size")
	}
	e.HeadDim = cmp.Or(e.HeadDim, e.HiddenSize/e.NumHeads)
	e.SubsamplingChannels = cmp.Or(e.SubsamplingChannels, 256)
	c.ProjectorHidden = cmp.Or(c.ProjectorHidden, c.TextHidden)
	theta := cmp.Or(e.RopeParameters.Theta, e.RopeTheta)
	switch {
	case c.MelBins%8 != 0 || c.MelBins == 0:
		return nil, fmt.Errorf("lfm3audio: %d mel bins; the downsampler needs a multiple of 8", c.MelBins)
	case c.Window <= 0:
		return nil, fmt.Errorf("lfm3audio: the encoder's sliding_window is missing")
	case len(e.LayerTypes) != e.NumLayers:
		return nil, fmt.Errorf("lfm3audio: %d encoder layer_types for %d layers", len(e.LayerTypes), e.NumLayers)
	case theta != 1e6:
		// llama.cpp's graph hard codes it; a checkpoint with another would be read wrongly there
		return nil, fmt.Errorf("lfm3audio: encoder rope_theta %g is not supported; only 1e6", theta)
	case e.ConvBias:
		return nil, fmt.Errorf("lfm3audio: encoder conv_bias is not supported")
	case e.ConvKernel < 2:
		return nil, fmt.Errorf("lfm3audio: encoder conv_L_cache %d", e.ConvKernel)
	case e.NumHeads%e.NumKVHeads != 0:
		return nil, fmt.Errorf("lfm3audio: %d kv heads for %d heads", e.NumKVHeads, e.NumHeads)
	}
	e.RopeTheta = theta
	for _, t := range e.LayerTypes {
		if t != "conv" && t != "full_attention" && t != "sliding_attention" {
			return nil, fmt.Errorf("lfm3audio: encoder layer type %q is not supported", t)
		}
	}
	return c, nil
}

// Hidden is the encoder's width, of the hidden state between layers. The input of a program that
// starts at the first layer is the mel frames instead (MelBins wide).
func (c *Config) Hidden() int { return c.Enc.HiddenSize }

// Layers is the number of encoder layers.
func (c *Config) Layers() int { return c.Enc.NumLayers }

// Output is the text model's width: the clip's embeddings.
func (c *Config) Output() (int, bool) { return c.TextHidden, false }

// Embedding is empty: the input is mel frames.
func (c *Config) Embedding() string { return "" }

// Tokens is the number of embeddings a clip of frames mel frames becomes: each stride 2
// convolution keeps ceil(n/2).
func Tokens(frames int) int {
	for range 3 {
		frames = (frames + 1) / 2
	}
	return frames
}

// Tensors lists the tensors the conversion reads. The checkpoint's text model, vision tower and
// audio output (the depthformer, the codebook embeddings) are other programs' and unused here.
func (c *Config) Tensors() map[string][]int {
	e := c.Enc
	d, hd, ch := e.HiddenSize, e.HeadDim, e.SubsamplingChannels
	t := map[string][]int{
		c.down + "conv_in.weight":         {ch, 1, 3, 3},
		c.down + "conv_in.bias":           {ch},
		c.down + "conv_mid.weight":        {ch, 1, 3, 3},
		c.down + "conv_mid.bias":          {ch},
		c.down + "lin_mid.weight":         {ch, ch, 1, 1},
		c.down + "lin_mid.bias":           {ch},
		c.down + "conv_end.weight":        {ch, 1, 3, 3},
		c.down + "conv_end.bias":          {ch},
		c.down + "lin_end.weight":         {ch, ch, 1, 1},
		c.down + "lin_end.bias":           {ch},
		c.down + "lin_out.weight":         {d, ch * c.MelBins / 8},
		c.down + "lin_out.bias":           {d},
		c.stack + "embedding_norm.weight": {d},
		c.norm + "weight":                 {d},
		c.norm + "bias":                   {d},
		c.lin1 + "weight":                 {c.ProjectorHidden, d},
		c.lin1 + "bias":                   {c.ProjectorHidden},
		c.lin2 + "weight":                 {c.TextHidden, c.ProjectorHidden},
		c.lin2 + "bias":                   {c.TextHidden},
	}
	ff := e.FFDim()
	for i, typ := range e.LayerTypes {
		p := fmt.Sprintf("%slayers.%d.", c.stack, i)
		t[p+"operator_norm.weight"] = []int{d}
		t[p+"ffn_norm.weight"] = []int{d}
		t[p+"feed_forward.w1.weight"] = []int{ff, d}
		t[p+"feed_forward.w3.weight"] = []int{ff, d}
		t[p+"feed_forward.w2.weight"] = []int{d, ff}
		if typ == "conv" {
			t[p+"conv.in_proj.weight"] = []int{3 * d, d}
			t[p+"conv.conv.weight"] = []int{d, 1, e.ConvKernel}
			t[p+"conv.out_proj.weight"] = []int{d, d}
		} else {
			a := p + "self_attn."
			t[a+"q_proj.weight"] = []int{e.NumHeads * hd, d}
			t[a+"k_proj.weight"] = []int{e.NumKVHeads * hd, d}
			t[a+"v_proj.weight"] = []int{e.NumKVHeads * hd, d}
			t[a+"out_proj.weight"] = []int{d, e.NumHeads * hd}
			t[a+"q_layernorm.weight"] = []int{hd}
			t[a+"k_layernorm.weight"] = []int{hd}
		}
	}
	return t
}

// Frames parses the frames parameter.
func Frames(param string) (int, error) {
	n, err := strconv.Atoi(param)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("lfm3audio: frames %q is not a positive number", param)
	}
	return n, nil
}

type builder struct {
	*blocks.Graph
	cfg      *Config
	frames   int
	n        int // positions after the downsampler
	cos, sin *mlpackage.Var
	mask     *mlpackage.Var
}

// Build writes the program for clips of up to the frames of opts.Params; opts.Seq is not used.
// When the program starts at the first layer its inputs are x, the clip's log mel frames (1,
// frames, mel bins) as Features computes them, zero after the clip, and mask (1, frames), 1 for
// the clip's frames and 0 after them; otherwise x is the hidden state (1, Tokens(frames),
// hidden). The output y is the embeddings (1, Tokens(frames), text hidden), of which the first
// Tokens(the clip's frames) are the clip's, or for a truncated program the hidden state.
func (c *Config) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	frames, err := Frames(Arch.Param(opts, "frames"))
	if err != nil {
		return nil, out, err
	}
	if opts.Cache > 0 {
		return nil, out, fmt.Errorf("lfm3audio: the audio encoder has no KV cache")
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = "lut4"
	}
	nl := c.Enc.NumLayers
	if opts.Layers <= 0 || opts.Start+opts.Layers > nl {
		opts.Layers = nl - opts.Start
	}
	m := &builder{Graph: &blocks.Graph{Name: "lfm3audio", B: b, W: w, Opts: opts}, cfg: c, frames: frames, n: Tokens(frames)}
	d := c.Enc.HiddenSize
	err = blocks.Run(func() {
		var h *mlpackage.Var
		if opts.Start == 0 {
			x := b.Input("x", mlpackage.Float16, 1, frames, c.MelBins)
			mask := b.Input("mask", mlpackage.Float16, 1, frames)
			ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}, {Name: "mask", DType: mlpackage.Float16, Shape: mask.Shape}}
			h = m.downsample(x, mask)
		} else {
			h = b.Input("x", mlpackage.Float16, 1, m.n, d)
			ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: h.Shape}}
		}
		for i := opts.Start; i < opts.Start+opts.Layers; i++ {
			h = m.layer(i, h)
		}
		var y *mlpackage.Var
		if opts.Start+opts.Layers < nl || opts.NoHead {
			y = b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		} else {
			y = m.projector(h)
		}
		out = mlpackage.Feature{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}
	})
	return ins, out, err
}

// strided is a stride 2 convolution with a 3x3 kernel and padding 1 over (time, mel) of x
// (1, T, F, C) laid out channels last, as nine taps: kernel (out channels, 1, 3, 3) indexed
// [channel][mel][time] as ggml applies the checkpoint's weight (its last axis over time).
// depthwise takes channel c from channel c of x; otherwise x has one channel, broadcast.
func (m *builder) strided(x *mlpackage.Var, name string, ch int) *mlpackage.Var {
	t, f := x.Shape[1], x.Shape[2]
	t2, f2 := (t+1)/2, (f+1)/2
	padded := m.Op("pad", map[string]any{"x": x, "pad": m.B.Ints(1, 1, 1, 1, 0, 0), "mode": m.Str("constant"), "constant_val": m.B.ScalarF16(0)}, 1, t+2, f+2, x.Shape[3])
	w := m.Get(name+".weight", ch, 1, 3, 3)
	var y *mlpackage.Var
	for kt := 0; kt < 3; kt++ {
		for kf := 0; kf < 3; kf++ {
			tap := make([]float32, ch)
			for c := range tap {
				tap[c] = w[c*9+kf*3+kt]
			}
			wv := m.F16s(fmt.Sprintf("%s.weight.tap%d%d", name, kt, kf), []int{1, 1, 1, ch}, tap)
			seg := m.Op("slice_by_index", map[string]any{"x": padded, "begin": m.B.Ints(0, int32(kt), int32(kf), 0),
				"end": m.B.Ints(1, int32(kt+2*t2-1), int32(kf+2*f2-1), int32(x.Shape[3])), "stride": m.B.Ints(1, 2, 2, 1),
				"begin_mask": m.Bools(true, false, false, true), "end_mask": m.Bools(true, false, false, true)}, 1, t2, f2, x.Shape[3])
			term := m.Op("mul", map[string]any{"x": seg, "y": wv}, 1, t2, f2, ch)
			if y == nil {
				y = term
			} else {
				y = m.Op("add", map[string]any{"x": y, "y": term}, 1, t2, f2, ch)
			}
		}
	}
	bias := m.F16s(name+".bias", []int{1, 1, 1, ch}, m.Get(name+".bias", ch))
	return m.Op("add", map[string]any{"x": y, "y": bias}, 1, t2, f2, ch)
}

// pointwise is a 1x1 convolution over the channels (the last axis) with its bias.
func (m *builder) pointwise(x *mlpackage.Var, name string, ch int) *mlpackage.Var {
	w := m.F16s(name+".weight", []int{ch, ch}, m.Get(name+".weight", ch, ch, 1, 1))
	bias := m.F16s(name+".bias", []int{ch}, m.Get(name+".bias", ch))
	return m.Op("linear", map[string]any{"x": x, "weight": w, "bias": bias}, x.Shape...)
}

// halve keeps every other position of a validity mask (1, T, 1, 1): a stride 2 output t exists
// for the clip when the input at 2t does.
func (m *builder) halve(mask *mlpackage.Var) *mlpackage.Var {
	t := mask.Shape[1]
	return m.Op("slice_by_index", map[string]any{"x": mask, "begin": m.B.Ints(0, 0, 0, 0), "end": m.B.Ints(1, int32(t), 1, 1), "stride": m.B.Ints(1, 2, 1, 1),
		"begin_mask": m.Bools(true, true, true, true), "end_mask": m.Bools(true, true, true, true)}, 1, (t+1)/2, 1, 1)
}

// downsample is the encoder's downsampler. llama.cpp pads each convolution's input with zeros
// after the clip; here the frames after the clip are zero in x, and each stage's outputs past the
// clip are set to zero by the halved mask, so the clip's own outputs are the same.
func (m *builder) downsample(x, mask *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	ch, p := c.Enc.SubsamplingChannels, c.down
	t := x.Shape[1]
	h := m.Op("reshape", map[string]any{"x": x, "shape": m.B.Ints(1, int32(t), int32(c.MelBins), 1)}, 1, t, c.MelBins, 1)
	mk := m.Op("reshape", map[string]any{"x": mask, "shape": m.B.Ints(1, int32(t), 1, 1)}, 1, t, 1, 1)
	silu := func(v *mlpackage.Var) *mlpackage.Var { return m.Op("silu", map[string]any{"x": v}, v.Shape...) }
	gate := func(v *mlpackage.Var) *mlpackage.Var {
		mk = m.halve(mk)
		return m.Op("mul", map[string]any{"x": v, "y": mk}, v.Shape...)
	}
	h = gate(silu(m.strided(h, p+"conv_in", ch)))
	h = gate(silu(m.pointwise(m.strided(h, p+"conv_mid", ch), p+"lin_mid", ch)))
	h = gate(silu(m.pointwise(m.strided(h, p+"conv_end", ch), p+"lin_end", ch)))
	// (1, T/8, F/8, C) -> (1, T/8, C * F/8), channel major as ggml flattens it
	n, f := h.Shape[1], h.Shape[2]
	h = m.Op("transpose", map[string]any{"x": h, "perm": m.B.Ints(0, 1, 3, 2)}, 1, n, ch, f)
	h = m.Op("reshape", map[string]any{"x": h, "shape": m.B.Ints(1, int32(n), int32(ch*f))}, 1, n, ch*f)
	name := p + "lin_out.weight"
	raw := m.Get(name)
	d := c.Enc.HiddenSize
	if len(raw) != d*ch*f {
		m.Fail("%s: %d values, expected %d x %d", name, len(raw), d, ch*f)
	}
	wv := m.F16s(name, []int{d, ch * f}, raw)
	bias := m.F16s(p+"lin_out.bias", []int{d}, m.Get(p+"lin_out.bias", d))
	return m.Op("linear", map[string]any{"x": h, "weight": wv, "bias": bias}, 1, n, d)
}

func (m *builder) layer(i int, h *mlpackage.Var) *mlpackage.Var {
	e := m.cfg.Enc
	d := e.HiddenSize
	p := fmt.Sprintf("%slayers.%d.", m.cfg.stack, i)
	residual := h
	h = m.RMSNorm(h, p+"operator_norm.weight", d, e.NormEps, 1)
	if e.LayerTypes[i] == "conv" {
		h = m.conv(p+"conv.", h)
	} else {
		h = m.attention(p+"self_attn.", h)
	}
	h = m.Op("add", map[string]any{"x": h, "y": residual}, h.Shape...)
	residual = h
	f := m.RMSNorm(h, p+"ffn_norm.weight", d, e.NormEps, 1)
	f = m.SwiGLU(f, p+"feed_forward.w1.weight", p+"feed_forward.w3.weight", p+"feed_forward.w2.weight", e.FFDim(), d, 1)
	return m.Op("add", map[string]any{"x": f, "y": residual}, f.Shape...)
}

// conv is LFM2's gated short convolution, causal from zeros, as LFM3's in taps mode.
func (m *builder) conv(p string, h *mlpackage.Var) *mlpackage.Var {
	d, s, k := m.cfg.Enc.HiddenSize, m.n, m.cfg.Enc.ConvKernel
	bcx := m.Linear(h, p+"in_proj.weight", "", 3*d)
	part := func(j int) *mlpackage.Var {
		return m.Op("slice_by_index", map[string]any{"x": bcx, "begin": m.B.Ints(0, 0, int32(j*d)), "end": m.B.Ints(1, int32(s), int32((j+1)*d)),
			"begin_mask": m.Bools(true, true, false), "end_mask": m.Bools(true, true, false)}, 1, s, d)
	}
	bg, cg, xg := part(0), part(1), part(2)
	bx := m.Op("mul", map[string]any{"x": bg, "y": xg}, 1, s, d)
	padded := m.Op("pad", map[string]any{"x": bx, "pad": m.B.Ints(int32(k-1), 0, 0, 0), "mode": m.Str("constant"), "constant_val": m.B.ScalarF16(0)}, 1, s+k-1, d)
	cw := m.Get(p+"conv.weight", d, 1, k)
	var y *mlpackage.Var
	for j := 0; j < k; j++ {
		tap := make([]float32, d)
		for c := range tap {
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
	y = m.Op("mul", map[string]any{"x": cg, "y": y}, 1, s, d)
	return m.Linear(y, p+"out_proj.weight", "", d)
}

// attention is causal grouped query attention over the sliding window, with QK RMS norm and
// RoPE (rotate_half) over the whole head.
func (m *builder) attention(p string, h *mlpackage.Var) *mlpackage.Var {
	e := m.cfg.Enc
	s, nh, nkv, hd := m.n, e.NumHeads, e.NumKVHeads, e.HeadDim
	if m.cos == nil {
		inv := make([]float64, hd/2)
		for i := range inv {
			inv[i] = math.Pow(e.RopeTheta, -float64(2*i)/float64(hd))
		}
		m.cos, m.sin = m.RopeTables("audio", inv, s, hd)
	}
	heads := func(name string, k int, norm string) *mlpackage.Var {
		x := m.Linear(h, p+name+"_proj.weight", "", k*hd)
		x = m.Op("reshape", map[string]any{"x": x, "shape": m.B.Ints(1, int32(s), int32(k), int32(hd))}, 1, s, k, hd)
		if norm != "" {
			x = m.RMSNorm(x, p+norm+".weight", hd, e.NormEps, 1)
			x = m.ApplyRope(x, m.cos, m.sin)
		}
		return m.Op("transpose", map[string]any{"x": x, "perm": m.B.Ints(0, 2, 1, 3)}, 1, k, s, hd)
	}
	q, k, v := heads("q", nh, "q_layernorm"), heads("k", nkv, "k_layernorm"), heads("v", nkv, "")
	if m.mask == nil {
		m.mask = m.CausalMask("mask_audio", s, m.cfg.Window)
	}
	o := m.SDPA(q, k, v, func() *mlpackage.Var { return m.mask })
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, s, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(s), int32(nh*hd))}, 1, s, nh*hd)
	return m.Linear(o, p+"out_proj.weight", "", e.HiddenSize)
}

// projector is the final norm of the stack, then LayerNorm, linear, GELU, linear.
func (m *builder) projector(h *mlpackage.Var) *mlpackage.Var {
	c := m.cfg
	d := c.Enc.HiddenSize
	h = m.RMSNorm(h, c.stack+"embedding_norm.weight", d, c.Enc.NormEps, 1)
	h = m.LayerNorm(h, c.norm+"weight", c.norm+"bias", d, 1e-5)
	h = m.LinearBias(h, c.lin1+"weight", c.lin1+"bias", "", c.ProjectorHidden)
	h = m.Activation(h, "gelu")
	return m.LinearBias(h, c.lin2+"weight", c.lin2+"bias", "y", c.TextHidden)
}
