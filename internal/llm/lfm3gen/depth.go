package lfm3gen

import (
	"cmp"
	"encoding/json"
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// DepthArch registers the depthformer; it is chosen with -arch on LFM3-Audio's or LFM3-Omni's
// config.json.
var DepthArch = &llm.Arch{
	Name:       "lfm3-depthformer",
	Title:      "LFM3 depthformer (speech codes)",
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseDepth(raw) },
}

func init() {
	llm.Register(DepthArch)
	llm.Register(CodecArch)
}

// Depth is the depthformer of an LFM3-Audio or LFM3-Omni checkpoint.
type Depth struct {
	Omni       bool
	Codebooks  int // codes per audio frame
	Vocab      int // codes per codebook, the last of them the end of audio
	TextHidden int
	st         stack
	// tensor names
	linear, embeddings, mask, feedback string
	// sampling of the model card, generation_config.json overrides it (internal/speech)
	Temperature float64
	TopK        int
	Repetition  float64
	RepWindow   int
	// Schedule is how many codebooks each MaskGIT pass reveals.
	Schedule []int
}

type depthConfig struct {
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"head_dim"`
	NormEps          float64  `json:"norm_eps"`
	LayerTypes       []string `json:"layer_types"`
	Sandwich         bool     `json:"use_sandwich_norm"`
	Causal           bool     `json:"is_causal"`
	RopeParameters   map[string]struct {
		Theta float64 `json:"rope_theta"`
	} `json:"rope_parameters"`
}

// ParseDepth parses the depthformer's settings of an LFM3-Audio or LFM3-Omni config.json, and
// rejects what llama.cpp's graph does not handle.
func ParseDepth(raw []byte) (*Depth, error) {
	var top struct {
		ModelType       string          `json:"model_type"`
		Depthformer     json.RawMessage `json:"depthformer"`
		DepthConfig     json.RawMessage `json:"depth_config"`
		Codebooks       int             `json:"codebooks"`
		NumCodebooks    int             `json:"num_codebooks"`
		AudioVocab      int             `json:"audio_vocab_size"`
		CodebookSize    int             `json:"codebook_size"`
		MaskgitSteps    int             `json:"maskgit_steps"`
		MaskgitSchedule []int           `json:"maskgit_schedule"`
		LFM             struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"lfm"`
		TextConfig struct {
			HiddenSize int `json:"hidden_size"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("lfm3gen: %w", err)
	}
	c := &Depth{}
	var dc json.RawMessage
	switch top.ModelType {
	case "lfm3_audio":
		dc, c.Codebooks, c.Vocab, c.TextHidden = top.Depthformer, top.Codebooks, top.AudioVocab, top.LFM.HiddenSize
		if top.MaskgitSteps > 0 && c.Codebooks%top.MaskgitSteps == 0 {
			for range top.MaskgitSteps {
				c.Schedule = append(c.Schedule, c.Codebooks/top.MaskgitSteps)
			}
		}
		c.st.prefix, c.linear, c.embeddings, c.mask, c.feedback = "depthformer.model.", "depth_linear.", "depth_embeddings.", "mask_embedding", "codebook_embedding.weight"
		c.Temperature, c.TopK, c.Repetition, c.RepWindow = 1.0, 128, 1.1, 4096
	case "lfm3_omni":
		dc, c.Codebooks, c.Vocab, c.TextHidden, c.Schedule = top.DepthConfig, top.NumCodebooks, top.CodebookSize, top.TextConfig.HiddenSize, top.MaskgitSchedule
		c.Omni = true
		c.st.prefix, c.linear, c.embeddings, c.mask, c.feedback = "model.depth_decoder.model.", "model.depth_decoder.depth_linear.", "model.depth_decoder.depth_embeddings.", "model.depth_decoder.mask_embedding", "model.codebook_embedding.weight"
		c.Temperature, c.TopK, c.Repetition, c.RepWindow = 0.4, 0, 1.2, 64
	default:
		return nil, fmt.Errorf("lfm3gen: model_type %q has no depthformer (lfm3_audio or lfm3_omni)", top.ModelType)
	}
	if len(dc) == 0 {
		return nil, fmt.Errorf("lfm3gen: no depthformer config")
	}
	var d depthConfig
	if err := json.Unmarshal(dc, &d); err != nil {
		return nil, fmt.Errorf("lfm3gen: depthformer: %w", err)
	}
	theta := d.RopeParameters["full_attention"].Theta
	full := len(d.LayerTypes) == d.NumLayers
	for _, t := range d.LayerTypes {
		full = full && t == "full_attention"
	}
	sum := 0
	for _, n := range c.Schedule {
		sum += n
	}
	switch {
	case d.Causal || d.Sandwich || !full:
		return nil, fmt.Errorf("lfm3gen: the depthformer must be a bidirectional stack of attention layers without sandwich norms")
	case theta != 1e6:
		return nil, fmt.Errorf("lfm3gen: depthformer rope_theta %g is not supported; only 1e6", theta)
	case c.Codebooks <= 0 || c.Vocab <= 1 || c.TextHidden == 0 || d.HiddenSize == 0 || d.NumHeads == 0:
		return nil, fmt.Errorf("lfm3gen: codebooks %d, vocabulary %d, text hidden %d, depth hidden %d", c.Codebooks, c.Vocab, c.TextHidden, d.HiddenSize)
	case sum != c.Codebooks:
		return nil, fmt.Errorf("lfm3gen: MaskGIT schedule %v does not reveal the %d codebooks", c.Schedule, c.Codebooks)
	}
	d.NumKVHeads = cmp.Or(d.NumKVHeads, d.NumHeads)
	c.st.d, c.st.layerTypes, c.st.heads, c.st.kvHeads = d.HiddenSize, d.LayerTypes, d.NumHeads, d.NumKVHeads
	c.st.headDim, c.st.ff, c.st.eps, c.st.ropeTheta = cmp.Or(d.HeadDim, d.HiddenSize/d.NumHeads), d.IntermediateSize, d.NormEps, theta
	c.st.qNorm, c.st.kNorm = "q_norm", "k_norm"
	return c, nil
}

// Hidden is the depthformer's width.
func (c *Depth) Hidden() int { return c.st.d }

// Layers is the number of depthformer layers.
func (c *Depth) Layers() int { return len(c.st.layerTypes) }

// Output is the logits of every codebook.
func (c *Depth) Output() (int, bool) { return c.Vocab, true }

// Embedding is the first codebook's table; the host reads every codebook's (Table).
func (c *Depth) Embedding() string { return c.Table(0) }

// Table names codebook cb's code embeddings (vocab, hidden), which are also its output layer.
func (c *Depth) Table(cb int) string { return fmt.Sprintf("%s%d.embedding.weight", c.embeddings, cb) }

// MaskEmbedding names the embedding of a masked codebook (hidden).
func (c *Depth) MaskEmbedding() string { return c.mask }

// Feedback names the table (codebooks * vocab, text hidden) whose rows of a frame's codes (code
// + codebook * vocab) are summed into the text model's next input.
func (c *Depth) Feedback() string { return c.feedback }

// Tensors lists the tensors the conversion reads. The codebooks' to_logits weights are the
// embeddings (tied) and unused; the feedback table is the host's.
func (c *Depth) Tensors() map[string][]int {
	d := c.st.d
	t := map[string][]int{
		c.st.prefix + "embedding_norm.weight": {d},
		c.linear + "weight":                   {c.Codebooks * d, c.TextHidden},
		c.linear + "bias":                     {c.Codebooks * d},
		c.mask:                                {d},
	}
	for cb := range c.Codebooks {
		t[c.Table(cb)] = []int{c.Vocab, d}
		t[fmt.Sprintf("%s%d.embedding_norm.weight", c.embeddings, cb)] = []int{d}
	}
	c.st.tensors(t)
	return t
}

// Build writes one MaskGIT pass. Its inputs are x (1, codebooks, hidden), each codebook's
// embedding of its code or the mask embedding (the host looks them up), and h (1, 1, text
// hidden), the text model's final hidden state; y is the logits (1, codebooks, vocab): the
// stack's output for each codebook, normed by its own weight, times its code table.
func (c *Depth) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Cache > 0 || opts.Start > 0 || opts.Layers > 0 {
		return nil, out, fmt.Errorf("lfm3gen: the depthformer is built whole")
	}
	if opts.Format == "" {
		opts.Format = "lut4"
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	n, d := c.Codebooks, c.st.d
	m := &builder{Graph: &blocks.Graph{Name: "lfm3gen", B: b, W: w, Opts: opts}, st: c.st, seq: n}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, n, d)
		h := b.Input("h", mlpackage.Float16, 1, 1, c.TextHidden)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}, {Name: "h", DType: mlpackage.Float16, Shape: h.Shape}}
		p := m.LinearBias(h, c.linear+"weight", c.linear+"bias", "", n*d)
		p = m.Op("reshape", map[string]any{"x": p, "shape": m.B.Ints(1, int32(n), int32(d))}, 1, n, d)
		cur := m.Op("add", map[string]any{"x": p, "y": x}, 1, n, d)
		for i := range c.st.layerTypes {
			cur = m.layer(i, cur)
		}
		cur = m.RMSNorm(cur, c.st.prefix+"embedding_norm.weight", d, c.st.eps, 1)
		// each codebook's own norm
		gains := make([]float32, n*d)
		for cb := range n {
			copy(gains[cb*d:], m.Get(fmt.Sprintf("%s%d.embedding_norm.weight", c.embeddings, cb), d))
		}
		sq := m.Op("mul", map[string]any{"x": cur, "y": cur}, 1, n, d)
		ms := m.Op("reduce_mean", map[string]any{"x": sq, "axes": m.B.Ints(-1), "keep_dims": m.B.ScalarBool(true)}, 1, n, 1)
		r := m.Op("rsqrt", map[string]any{"x": ms, "epsilon": m.B.ScalarF16(float32(c.st.eps))}, 1, n, 1)
		cur = m.Op("mul", map[string]any{"x": cur, "y": r}, 1, n, d)
		cur = m.Op("mul", map[string]any{"x": cur, "y": m.F16s("codebook_norm.weight", []int{1, n, d}, gains)}, 1, n, d)
		// logits: per codebook, (1, d) x (d, vocab)
		tables := make([]float32, n*d*c.Vocab)
		for cb := range n {
			e := m.Get(c.Table(cb), c.Vocab, d)
			for v := range c.Vocab {
				for j := range d {
					tables[(cb*d+j)*c.Vocab+v] = e[v*d+j]
				}
			}
		}
		cur = m.Op("reshape", map[string]any{"x": cur, "shape": m.B.Ints(int32(n), 1, int32(d))}, n, 1, d)
		logits := m.Op("matmul", map[string]any{"x": cur, "y": m.F16s("codebook_heads", []int{n, d, c.Vocab}, tables),
			"transpose_x": m.B.ScalarBool(false), "transpose_y": m.B.ScalarBool(false)}, n, 1, c.Vocab)
		y := b.NamedOp("reshape", "y", map[string]any{"x": logits, "shape": m.B.Ints(1, int32(n), int32(c.Vocab))}, mlpackage.Float16, 1, n, c.Vocab)
		out = mlpackage.Feature{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}
	})
	return ins, out, err
}
