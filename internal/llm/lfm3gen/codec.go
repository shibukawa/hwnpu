package lfm3gen

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// CodecArch registers the decoder of the liquid codec (audio_tokenizer/ of an LFM3-Audio or
// LFM3-Omni checkpoint, model_type liquid_codec).
var CodecArch = &llm.Arch{
	Name:       "lfm3-codec",
	Title:      "LFM3 liquid codec decoder",
	ModelTypes: []string{"liquid_codec"},
	Params: []llm.Param{
		{Name: "codec-frames", Default: "64", Usage: "audio frames (80 ms each) of the codec decoder's program per call; internal/speech decodes longer audio in overlapping chunks"},
	},
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseCodec(raw) },
}

// Codec is the decoder of the liquid codec: the codes of a frame become STFT frames (FrameStack
// per audio frame) of log magnitude and phase.
type Codec struct {
	Levels     []int // FSQ levels of each codebook's code
	Codebooks  int
	QuantDim   int
	FrameStack int
	NFFT, Hop  int
	SampleRate int
	Window     int // the decoder's causal attention window, in positions
	st         stack
	resolved   bool
	quant      string // the FSQ projection's prefix
}

// The decoder's constants, as llama.cpp's clip.cpp has them for LiquidCodecConfig (its converter
// checks the codec's config.json against them).
const (
	codecLayers = 8
	codecWidth  = 512
	codecEps    = 1e-5
	codecTheta  = 1e6
)

// ParseCodec parses the codec's config.json and rejects what llama.cpp's graph does not handle.
// The decoder's layers (their kinds, heads and widths) are read from the weights when the
// program is built (Resolve).
func ParseCodec(raw []byte) (*Codec, error) {
	var f struct {
		ModelType  string `json:"model_type"`
		Levels     []int  `json:"fsq_levels"`
		NQ         int    `json:"n_q"`
		QuantDim   int    `json:"quantizer_dim"`
		EncoderDim int    `json:"encoder_dim"`
		NFFT       int    `json:"stft_n_fft"`
		Hop        int    `json:"stft_hop_length"`
		Win        int    `json:"stft_win_length"`
		FrameStack int    `json:"stft_frame_stack"`
		Window     int    `json:"decoder_window"`
		Rate       int    `json:"sampling_rate"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		return nil, fmt.Errorf("lfm3gen: %w", err)
	}
	if f.ModelType != "liquid_codec" {
		return nil, fmt.Errorf("lfm3gen: model_type %q is not liquid_codec", f.ModelType)
	}
	if f.NQ <= 0 || len(f.Levels) == 0 || f.QuantDim <= 0 || f.EncoderDim != codecWidth || f.NFFT <= 0 || f.Hop <= 0 || f.Win != f.NFFT || f.FrameStack <= 0 || f.Window <= 0 || f.Rate <= 0 {
		return nil, fmt.Errorf("lfm3gen: codec config %+v is not supported (llama.cpp's: 12 codebooks of FSQ levels, a 512 wide decoder, an STFT window of n_fft)", f)
	}
	return &Codec{Levels: f.Levels, Codebooks: f.NQ, QuantDim: f.QuantDim, FrameStack: f.FrameStack, NFFT: f.NFFT, Hop: f.Hop,
		SampleRate: f.Rate, Window: f.Window, quant: "quantizer.fsq.",
		st: stack{prefix: "decoder.lfm.", d: codecWidth, eps: codecEps, ropeTheta: codecTheta, sandwich: true, window: f.Window, qNorm: "q_norm", kNorm: "k_norm"}}, nil
}

// Resolve reads the decoder's layers from the weights: a layer with conv.in_proj is a short
// convolution, otherwise attention, with the widths of its tensors; the FSQ projection is under
// quantizer.fsq. or (older exports) quantizer.
func (c *Codec) Resolve(w llm.Weights) error {
	if c.resolved {
		return nil
	}
	if _, _, err := w.Get("quantizer.fsq.project_out.weight"); err != nil {
		c.quant = "quantizer."
	}
	c.st.layerTypes = nil
	for i := range codecLayers {
		p := fmt.Sprintf("%slayers.%d.", c.st.prefix, i)
		_, w1, err := w.Get(p + "feed_forward.w1.weight")
		if err != nil {
			return fmt.Errorf("lfm3gen: codec layer %d: %w", i, err)
		}
		c.st.ff = w1[0]
		if _, sh, err := w.Get(p + "conv.conv.weight"); err == nil {
			c.st.layerTypes = append(c.st.layerTypes, "conv")
			c.st.convKernel = sh[len(sh)-1]
			continue
		}
		_, qn, err1 := w.Get(p + "self_attn.q_norm.weight")
		_, q, err2 := w.Get(p + "self_attn.q_proj.weight")
		_, k, err3 := w.Get(p + "self_attn.k_proj.weight")
		if err1 != nil || err2 != nil || err3 != nil {
			return fmt.Errorf("lfm3gen: codec layer %d is neither a conv nor an attention block", i)
		}
		c.st.layerTypes = append(c.st.layerTypes, "full_attention")
		c.st.headDim, c.st.heads, c.st.kvHeads = qn[0], q[0]/qn[0], k[0]/qn[0]
	}
	c.resolved = true
	return nil
}

// SetLayers sets the decoder's layers instead of Resolve (for tests).
func (c *Codec) SetLayers(types []string, heads, kvHeads, headDim, ff, convKernel int) {
	c.st.layerTypes, c.st.heads, c.st.kvHeads, c.st.headDim, c.st.ff, c.st.convKernel = types, heads, kvHeads, headDim, ff, convKernel
	c.resolved = true
}

// Lookback is how many decoder positions before a position can change it: each attention layer
// sees Window-1 back, each convolution its kernel-1.
func (c *Codec) Lookback() int {
	n := 0
	for _, t := range c.st.layerTypes {
		if t == "conv" {
			n += c.st.convKernel - 1
		} else {
			n += c.Window - 1
		}
	}
	return n
}

// Bins is the number of frequency bins of an STFT frame.
func (c *Codec) Bins() int { return c.NFFT/2 + 1 }

// CodeDim is the width of a frame's code vectors (the program's input per frame).
func (c *Codec) CodeDim() int { return c.Codebooks * len(c.Levels) }

// Hidden is the decoder's width.
func (c *Codec) Hidden() int { return codecWidth }

// Layers is the number of decoder layers.
func (c *Codec) Layers() int { return codecLayers }

// Output is an STFT frame: the log magnitudes of the bins, then their phases.
func (c *Codec) Output() (int, bool) { return 2 * c.Bins(), false }

// Embedding is empty: the input is code vectors.
func (c *Codec) Embedding() string { return "" }

// Tensors lists the tensors the conversion reads (the decoder's layers once resolved). The
// codec's encoder is not used.
func (c *Codec) Tensors() map[string][]int {
	t := map[string][]int{
		c.quant + "project_out.weight":      {c.QuantDim, c.CodeDim()},
		c.quant + "project_out.bias":        {c.QuantDim},
		"from_quantizer.weight":             {codecWidth, c.QuantDim},
		"from_quantizer.bias":               {codecWidth},
		"decoder.lfm.embedding_norm.weight": {codecWidth},
		"decoder.lin.weight":                {2 * c.Bins(), codecWidth},
		"decoder.lin.bias":                  {2 * c.Bins()},
	}
	if c.resolved {
		c.st.tensors(t)
	}
	return t
}

// Codes is the FSQ table of a codebook's code: its level values in [-1, 1], the first level
// varying fastest (FiniteScalarQuantization._indices_to_codes).
func (c *Codec) Codes(code int) []float32 {
	out := make([]float32, len(c.Levels))
	basis := 1
	for i, l := range c.Levels {
		half := l / 2
		out[i] = float32((code/basis)%l-half) / float32(half)
		basis *= l
	}
	return out
}

// Build writes the decoder for a fixed number of frames (opts.Params "codec-frames"). x is (1, frames,
// CodeDim): each frame's code vectors (Codes) in codebook order, as the host looks them up; y is
// (1, frames*FrameStack, 2*Bins): the STFT frames. The decoder is causal from a zero state; the
// host decodes longer audio in chunks that overlap by Lookback positions.
func (c *Codec) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	frames, err := strconv.Atoi(CodecArch.Param(opts, "codec-frames"))
	if err != nil || frames <= 0 {
		return nil, out, fmt.Errorf("lfm3gen: codec-frames %q is not a positive number", CodecArch.Param(opts, "codec-frames"))
	}
	if opts.Cache > 0 || opts.Start > 0 || opts.Layers > 0 {
		return nil, out, fmt.Errorf("lfm3gen: the codec decoder is built whole")
	}
	if err := c.Resolve(w); err != nil {
		return nil, out, err
	}
	if opts.Format == "" {
		opts.Format = "lut4"
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	n := frames * c.FrameStack
	m := &builder{Graph: &blocks.Graph{Name: "lfm3gen", B: b, W: w, Opts: opts}, st: c.st, seq: n}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, frames, c.CodeDim())
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		h := m.LinearBias(x, c.quant+"project_out.weight", c.quant+"project_out.bias", "", c.QuantDim)
		h = m.LinearBias(h, "from_quantizer.weight", "from_quantizer.bias", "", codecWidth)
		// nearest upsampling: each frame becomes FrameStack positions
		h = m.Op("reshape", map[string]any{"x": h, "shape": m.B.Ints(1, int32(frames), 1, codecWidth)}, 1, frames, 1, codecWidth)
		h = m.Op("tile", map[string]any{"x": h, "reps": m.B.Ints(1, 1, int32(c.FrameStack), 1)}, 1, frames, c.FrameStack, codecWidth)
		h = m.Op("reshape", map[string]any{"x": h, "shape": m.B.Ints(1, int32(n), codecWidth)}, 1, n, codecWidth)
		for i := range c.st.layerTypes {
			h = m.layer(i, h)
		}
		h = m.RMSNorm(h, "decoder.lfm.embedding_norm.weight", codecWidth, codecEps, 1)
		y := m.LinearBias(h, "decoder.lin.weight", "decoder.lin.bias", "y", 2*c.Bins())
		out = mlpackage.Feature{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}
	})
	return ins, out, err
}
