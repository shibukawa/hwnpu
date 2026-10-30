// Package lfm3vl converts the vision encoder of Liquid AI's LFM3-VL (model_type lfm3_vl: the
// vision tower and the multimodal projector; the text model is internal/llm/lfm3) into a Core ML
// program for one fixed image size. The architecture is lfm3-vision, chosen with -arch on the
// checkpoint's config.json.
//
// The tower is LFM style, as llama.cpp's tools/mtmd/models/lfm3vl.cpp builds it: a patchifier (a
// linear over each patch's pixels), an embedding RMS norm, then pre-norm layers of two kinds,
// gated 3x3 depthwise convolutions over the patch grid and attention with QK norm and 2D RoPE,
// each followed by a SwiGLU MLP. The projector normalizes each patch (LayerNorm), concatenates
// the patches of each 2x2 block and maps them through linear, GELU, linear to the text model's
// width. The output rows are the image's embeddings in the text model's input, in raster order of
// the 2x2 blocks.
package lfm3vl

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch registers the LFM3-VL vision encoder with the whitelist. It has no model_type of its own
// (the checkpoint's is lfm3_vl, whose text model is lfm3), so it is chosen with -arch.
var Arch = &llm.Arch{
	Name:  "lfm3-vision",
	Title: "LFM3-VL vision encoder",
	Params: []llm.Param{
		{Name: "image", Default: "512x512", Usage: "image size WIDTHxHEIGHT in pixels of the vision encoder's program, multiples of twice the patch size"},
	},
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseConfig(raw) },
}

func init() { llm.Register(Arch) }

// Config is the subset of an LFM3-VL config.json the vision encoder needs.
type Config struct {
	Vision struct {
		HiddenSize        int      `json:"hidden_size"`
		IntermediateSize  int      `json:"intermediate_size"`
		NumLayers         int      `json:"num_hidden_layers"`
		NumHeads          int      `json:"num_attention_heads"`
		NumKVHeads        int      `json:"num_key_value_heads"`
		HeadDim           int      `json:"head_dim"`
		PatchSize         int      `json:"patch_size"`
		TemporalPatchSize int      `json:"temporal_patch_size"`
		InChannels        int      `json:"in_channels"`
		ConvKernel        int      `json:"conv_kernel_size"`
		NormEps           float64  `json:"norm_eps"`
		LayerTypes        []string `json:"layer_types"`
		RopeParameters    struct {
			Theta float64 `json:"rope_theta"`
		} `json:"rope_parameters"`
	} `json:"vision_config"`
	Text struct {
		HiddenSize int `json:"hidden_size"`
	} `json:"text_config"`
	ProjectorHiddenSize int      `json:"projector_hidden_size"`
	ProjectorHiddenAct  string   `json:"projector_hidden_act"`
	ProjectorBias       *bool    `json:"projector_bias"`
	ProjectorLNEps      *float64 `json:"projector_layer_norm_eps"`
	DownsampleFactor    int      `json:"downsample_factor"`
}

// ParseConfig parses an LFM3-VL config.json and rejects vision settings the conversion does not
// handle (those llama.cpp's graph does not handle either).
func ParseConfig(raw []byte) (*Config, error) {
	var top struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("lfm3vl: %w", err)
	}
	if top.ModelType != "lfm3_vl" {
		return nil, fmt.Errorf("lfm3vl: model_type %q is not lfm3_vl", top.ModelType)
	}
	var c Config
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("lfm3vl: %w", err)
	}
	v := &c.Vision
	if v.TemporalPatchSize == 0 {
		v.TemporalPatchSize = 1
	}
	if v.InChannels == 0 {
		v.InChannels = 3
	}
	if v.NumKVHeads == 0 {
		v.NumKVHeads = v.NumHeads
	}
	if v.NumHeads == 0 || v.HiddenSize == 0 || v.PatchSize == 0 || c.Text.HiddenSize == 0 {
		return nil, fmt.Errorf("lfm3vl: vision_config needs hidden_size, num_attention_heads and patch_size, text_config hidden_size")
	}
	if v.HeadDim == 0 {
		v.HeadDim = v.HiddenSize / v.NumHeads
	}
	if c.ProjectorHiddenSize == 0 {
		c.ProjectorHiddenSize = c.Text.HiddenSize
	}
	if c.DownsampleFactor == 0 {
		c.DownsampleFactor = 2
	}
	switch {
	case len(v.LayerTypes) != v.NumLayers:
		return nil, fmt.Errorf("lfm3vl: %d vision layer_types for %d layers", len(v.LayerTypes), v.NumLayers)
	case v.InChannels != 3 || v.ConvKernel != 3:
		return nil, fmt.Errorf("lfm3vl: in_channels %d and conv_kernel_size %d are not supported; expected 3 and 3", v.InChannels, v.ConvKernel)
	case v.TemporalPatchSize != 1 && v.TemporalPatchSize != 2:
		return nil, fmt.Errorf("lfm3vl: temporal_patch_size %d is not supported", v.TemporalPatchSize)
	case v.HeadDim%4 != 0 || v.NumHeads%v.NumKVHeads != 0:
		return nil, fmt.Errorf("lfm3vl: head_dim %d (2D RoPE needs a multiple of 4) or %d kv heads for %d heads", v.HeadDim, v.NumKVHeads, v.NumHeads)
	case v.RopeParameters.Theta <= 0:
		return nil, fmt.Errorf("lfm3vl: vision rope_theta missing")
	case c.DownsampleFactor != 2:
		return nil, fmt.Errorf("lfm3vl: downsample_factor %d is not supported; only 2", c.DownsampleFactor)
	case c.ProjectorHiddenAct != "gelu" || c.ProjectorBias == nil || !*c.ProjectorBias:
		return nil, fmt.Errorf("lfm3vl: projector %q with bias %v is not supported; only gelu with biases", c.ProjectorHiddenAct, c.ProjectorBias)
	}
	for _, t := range v.LayerTypes {
		if t != "conv" && t != "full_attention" {
			return nil, fmt.Errorf("lfm3vl: vision layer type %q is not supported", t)
		}
	}
	if c.ProjectorLNEps == nil {
		eps := 1e-6
		c.ProjectorLNEps = &eps
	}
	return &c, nil
}

// PatchDim is the width of a patch's pixels, the program's input per patch: channels then rows
// then columns, a still image's temporal slices summed into the weights.
func (c *Config) PatchDim() int { return c.Vision.InChannels * c.Vision.PatchSize * c.Vision.PatchSize }

// Hidden is the vision tower's width, of the hidden state between layers. The input of a program
// that starts at the first layer is the patches instead (PatchDim wide).
func (c *Config) Hidden() int { return c.Vision.HiddenSize }

// Layers is the number of vision layers.
func (c *Config) Layers() int { return c.Vision.NumLayers }

// Output is the text model's width: the image's embeddings.
func (c *Config) Output() (int, bool) { return c.Text.HiddenSize, false }

// Embedding is empty: the input is pixels, not tokens.
func (c *Config) Embedding() string { return "" }

const (
	tower = "model.vision_tower."
	proj  = "model.multi_modal_projector."
)

// Tensors lists the tensors the conversion reads. The checkpoint's text model and output layer
// belong to the lfm3 architecture and are unused here.
func (c *Config) Tensors() map[string][]int {
	v := c.Vision
	d, hd := v.HiddenSize, v.HeadDim
	t := map[string][]int{
		tower + "patchifier.weight":     {d, v.TemporalPatchSize * c.PatchDim()},
		tower + "embedding_norm.weight": {d},
		proj + "layer_norm.weight":      {d},
		proj + "layer_norm.bias":        {d},
		proj + "linear_1.weight":        {c.ProjectorHiddenSize, 4 * d},
		proj + "linear_1.bias":          {c.ProjectorHiddenSize},
		proj + "linear_2.weight":        {c.Text.HiddenSize, c.ProjectorHiddenSize},
		proj + "linear_2.bias":          {c.Text.HiddenSize},
	}
	for i, typ := range v.LayerTypes {
		p := fmt.Sprintf("%slayers.%d.", tower, i)
		t[p+"operator_norm.weight"] = []int{d}
		t[p+"ffn_norm.weight"] = []int{d}
		t[p+"feed_forward.w1.weight"] = []int{v.IntermediateSize, d}
		t[p+"feed_forward.w3.weight"] = []int{v.IntermediateSize, d}
		t[p+"feed_forward.w2.weight"] = []int{d, v.IntermediateSize}
		if typ == "conv" {
			t[p+"conv.in_proj.weight"] = []int{3 * d, d}
			t[p+"conv.spatial_conv.weight"] = []int{d, 1, 3, 3}
			t[p+"conv.out_proj.weight"] = []int{d, d}
		} else {
			a := p + "self_attn."
			t[a+"q_proj.weight"] = []int{v.NumHeads * hd, d}
			t[a+"k_proj.weight"] = []int{v.NumKVHeads * hd, d}
			t[a+"v_proj.weight"] = []int{v.NumKVHeads * hd, d}
			t[a+"out_proj.weight"] = []int{d, v.NumHeads * hd}
			t[a+"q_layernorm.weight"] = []int{hd}
			t[a+"k_layernorm.weight"] = []int{hd}
		}
	}
	return t
}

// ImageSize parses the image parameter: width and height in pixels, multiples of twice the
// patch size.
func (c *Config) ImageSize(param string) (w, h int, err error) {
	ws, hs, ok := strings.Cut(param, "x")
	if ok {
		w, err = strconv.Atoi(ws)
		if err == nil {
			h, err = strconv.Atoi(hs)
		}
	}
	f := 2 * c.Vision.PatchSize
	if !ok || err != nil || w <= 0 || h <= 0 || w%f != 0 || h%f != 0 {
		return 0, 0, fmt.Errorf("lfm3vl: image size %q is not WIDTHxHEIGHT in multiples of %d", param, f)
	}
	return w, h, nil
}

// Tokens is the number of embeddings an image of w x h pixels becomes.
func (c *Config) Tokens(w, h int) int {
	p := 2 * c.Vision.PatchSize
	return (w / p) * (h / p)
}

type builder struct {
	*blocks.Graph
	cfg    *Config
	gw, gh int // the patch grid
	n      int // patches
	cos    *mlpackage.Var
	sin    *mlpackage.Var
}

// Build writes the program for the image size of opts.Params ("image"); opts.Seq is not used.
// The input x is the image's patches (1, patches, PatchDim) in raster order, each normalized as
// the image processor does (Patches), when the program starts at the first layer, and the hidden
// state (1, patches, hidden) otherwise. The output y is the image's embeddings (1, tokens, text
// hidden), or for a truncated program the hidden state after its last layer.
func (c *Config) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	iw, ih, err := c.ImageSize(Arch.Param(opts, "image"))
	if err != nil {
		return nil, out, err
	}
	if opts.Cache > 0 {
		return nil, out, fmt.Errorf("lfm3vl: the vision encoder has no KV cache")
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = "lut4"
	}
	nl := c.Vision.NumLayers
	if opts.Layers <= 0 || opts.Start+opts.Layers > nl {
		opts.Layers = nl - opts.Start
	}
	m := &builder{Graph: &blocks.Graph{Name: "lfm3vl", B: b, W: w, Opts: opts}, cfg: c, gw: iw / c.Vision.PatchSize, gh: ih / c.Vision.PatchSize}
	m.n = m.gw * m.gh
	d := c.Vision.HiddenSize
	err = blocks.Run(func() {
		width := d
		if opts.Start == 0 {
			width = c.PatchDim()
		}
		x := b.Input("x", mlpackage.Float16, 1, m.n, width)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		h := x
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
			y = m.projector(h)
		}
		out = mlpackage.Feature{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}
	})
	return ins, out, err
}

// embed is the patchifier and the embedding norm. The patchifier's weight has a slice per
// temporal step; a still image fills them all with itself, so they are summed.
func (m *builder) embed(x *mlpackage.Var) *mlpackage.Var {
	v := m.cfg.Vision
	d, pd := v.HiddenSize, m.cfg.PatchDim()
	name := tower + "patchifier.weight"
	raw := m.Get(name)
	if len(raw) != d*v.TemporalPatchSize*pd {
		m.Fail("%s: %d values, expected %d x %d", name, len(raw), d, v.TemporalPatchSize*pd)
	}
	// (d, c, t, p, p) -> (d, c, p, p) summed over t
	pp := v.PatchSize * v.PatchSize
	wsum := make([]float32, d*pd)
	for o := 0; o < d; o++ {
		for ch := 0; ch < v.InChannels; ch++ {
			for t := 0; t < v.TemporalPatchSize; t++ {
				src := raw[((o*v.InChannels+ch)*v.TemporalPatchSize+t)*pp:][:pp]
				dst := wsum[o*pd+ch*pp:][:pp]
				for i, w := range src {
					dst[i] += w
				}
			}
		}
	}
	wv := m.F16s(name, []int{d, pd}, wsum)
	h := m.Op("linear", map[string]any{"x": x, "weight": wv}, 1, m.n, d)
	return m.RMSNorm(h, tower+"embedding_norm.weight", d, v.NormEps, 1)
}

func (m *builder) layer(i int, h *mlpackage.Var) *mlpackage.Var {
	v := m.cfg.Vision
	d := v.HiddenSize
	p := fmt.Sprintf("%slayers.%d.", tower, i)
	residual := h
	h = m.RMSNorm(h, p+"operator_norm.weight", d, v.NormEps, 1)
	if v.LayerTypes[i] == "conv" {
		h = m.conv(p+"conv.", h)
	} else {
		h = m.attention(p+"self_attn.", h)
	}
	h = m.Op("add", map[string]any{"x": h, "y": residual}, h.Shape...)
	residual = h
	f := m.RMSNorm(h, p+"ffn_norm.weight", d, v.NormEps, 1)
	f = m.SwiGLU(f, p+"feed_forward.w1.weight", p+"feed_forward.w3.weight", p+"feed_forward.w2.weight", v.IntermediateSize, d, 1)
	return m.Op("add", map[string]any{"x": f, "y": residual}, f.Shape...)
}

// conv is the gated spatial convolution: in_proj splits into B, C and x; the 3x3 depthwise
// convolution (zero padded) runs over B*x on the patch grid, and C gates the result. Like LFM3's
// short convolution in taps mode, it is nine shifted multiply adds in the (grid rows, grid
// columns, channels) layout.
func (m *builder) conv(p string, h *mlpackage.Var) *mlpackage.Var {
	d, n, gh, gw := m.cfg.Vision.HiddenSize, m.n, m.gh, m.gw
	bcx := m.Linear(h, p+"in_proj.weight", "", 3*d)
	part := func(j int) *mlpackage.Var {
		return m.Op("slice_by_index", map[string]any{"x": bcx, "begin": m.B.Ints(0, 0, int32(j*d)), "end": m.B.Ints(1, int32(n), int32((j+1)*d)),
			"begin_mask": m.Bools(true, true, false), "end_mask": m.Bools(true, true, false)}, 1, n, d)
	}
	bg, cg, xg := part(0), part(1), part(2)
	bx := m.Op("mul", map[string]any{"x": bg, "y": xg}, 1, n, d)
	grid := m.Op("reshape", map[string]any{"x": bx, "shape": m.B.Ints(1, int32(gh), int32(gw), int32(d))}, 1, gh, gw, d)
	padded := m.Op("pad", map[string]any{"x": grid, "pad": m.B.Ints(1, 1, 1, 1, 0, 0), "mode": m.Str("constant"), "constant_val": m.B.ScalarF16(0)}, 1, gh+2, gw+2, d)
	cw := m.Get(p+"spatial_conv.weight", d, 1, 3, 3)
	var y *mlpackage.Var
	for dy := 0; dy < 3; dy++ {
		for dx := 0; dx < 3; dx++ {
			tap := make([]float32, d)
			for ch := range tap {
				tap[ch] = cw[ch*9+dy*3+dx]
			}
			wv := m.F16s(fmt.Sprintf("%sspatial_conv.weight.tap%d%d", p, dy, dx), []int{1, 1, 1, d}, tap)
			seg := m.Op("slice_by_index", map[string]any{"x": padded, "begin": m.B.Ints(0, int32(dy), int32(dx), 0), "end": m.B.Ints(1, int32(dy+gh), int32(dx+gw), int32(d)),
				"begin_mask": m.Bools(true, false, false, true), "end_mask": m.Bools(true, false, false, true)}, 1, gh, gw, d)
			term := m.Op("mul", map[string]any{"x": seg, "y": wv}, 1, gh, gw, d)
			if y == nil {
				y = term
			} else {
				y = m.Op("add", map[string]any{"x": y, "y": term}, 1, gh, gw, d)
			}
		}
	}
	y = m.Op("reshape", map[string]any{"x": y, "shape": m.B.Ints(1, int32(n), int32(d))}, 1, n, d)
	y = m.Op("mul", map[string]any{"x": cg, "y": y}, 1, n, d)
	return m.Linear(y, p+"out_proj.weight", "", d)
}

// RopeValues returns the 2D rotary tables (patches, head_dim) of a gw x gh grid in raster order,
// laid out for blocks.ApplyRope: pair (k, k+head_dim/2) turns by the patch's row for k below
// head_dim/4 and by its column above, each axis with the frequencies of a head_dim/2 wide RoPE
// (ggml's GGML_ROPE_TYPE_VISION, as llama.cpp runs LFM3-VL).
func RopeValues(gw, gh, headDim int, theta float64) (cos, sin []float32) {
	half, quarter := headDim/2, headDim/4
	n := gw * gh
	cos, sin = make([]float32, n*headDim), make([]float32, n*headDim)
	for py := 0; py < gh; py++ {
		for px := 0; px < gw; px++ {
			i := py*gw + px
			for k := 0; k < half; k++ {
				pos, f := py, k
				if k >= quarter {
					pos, f = px, k-quarter
				}
				a := float64(pos) * math.Pow(theta, -float64(2*f)/float64(half))
				c, s := float32(math.Cos(a)), float32(math.Sin(a))
				cos[i*headDim+k], cos[i*headDim+k+half] = c, c
				sin[i*headDim+k], sin[i*headDim+k+half] = s, s
			}
		}
	}
	return cos, sin
}

// attention is bidirectional grouped query attention with QK RMS norm and 2D RoPE.
func (m *builder) attention(p string, h *mlpackage.Var) *mlpackage.Var {
	v := m.cfg.Vision
	n, nh, nkv, hd := m.n, v.NumHeads, v.NumKVHeads, v.HeadDim
	if m.cos == nil {
		c, s := RopeValues(m.gw, m.gh, hd, v.RopeParameters.Theta)
		m.cos, m.sin = m.F16s("rope_cos_vision", []int{1, n, 1, hd}, c), m.F16s("rope_sin_vision", []int{1, n, 1, hd}, s)
	}
	heads := func(name string, k int, norm string) *mlpackage.Var {
		x := m.Linear(h, p+name+"_proj.weight", "", k*hd)
		x = m.Op("reshape", map[string]any{"x": x, "shape": m.B.Ints(1, int32(n), int32(k), int32(hd))}, 1, n, k, hd)
		if norm != "" {
			x = m.RMSNorm(x, p+norm+".weight", hd, v.NormEps, 1)
			x = m.ApplyRope(x, m.cos, m.sin)
		}
		return m.Op("transpose", map[string]any{"x": x, "perm": m.B.Ints(0, 2, 1, 3)}, 1, k, n, hd)
	}
	q, k, vv := heads("q", nh, "q_layernorm"), heads("k", nkv, "k_layernorm"), heads("v", nkv, "")
	o := m.SDPA(q, k, vv, nil)
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, n, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(n), int32(nh*hd))}, 1, n, nh*hd)
	return m.Linear(o, p+"out_proj.weight", "", v.HiddenSize)
}

// projector normalizes each patch, merges each 2x2 block of patches into one row (the block's
// top row then its bottom row, left to right) and maps it to the text model's width.
func (m *builder) projector(h *mlpackage.Var) *mlpackage.Var {
	d, gh, gw := m.cfg.Vision.HiddenSize, m.gh, m.gw
	h = m.LayerNorm(h, proj+"layer_norm.weight", proj+"layer_norm.bias", d, *m.cfg.ProjectorLNEps)
	h = m.Op("reshape", map[string]any{"x": h, "shape": m.B.Ints(int32(gh/2), 2, int32(gw/2), 2, int32(d))}, gh/2, 2, gw/2, 2, d)
	h = m.Op("transpose", map[string]any{"x": h, "perm": m.B.Ints(0, 2, 1, 3, 4)}, gh/2, gw/2, 2, 2, d)
	t := gh * gw / 4
	h = m.Op("reshape", map[string]any{"x": h, "shape": m.B.Ints(1, int32(t), int32(4*d))}, 1, t, 4*d)
	h = m.LinearBias(h, proj+"linear_1.weight", proj+"linear_1.bias", "", m.cfg.ProjectorHiddenSize)
	h = m.Activation(h, "gelu")
	return m.LinearBias(h, proj+"linear_2.weight", proj+"linear_2.bias", "y", m.cfg.Text.HiddenSize)
}
