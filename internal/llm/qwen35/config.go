// Package qwen35 converts the text model of Qwen's Qwen3.5 (Hugging Face `qwen3_5`, the dense
// 0.8B to 27B) into Core ML programs for the Neural Engine.
//
// Qwen3.5 interleaves two kinds of layers, three Gated DeltaNet layers (linear attention with a
// recurrent state of a fixed size, a short convolution in front) to one full attention layer
// (grouped query attention with QK norm, partial RoPE and a sigmoid gate per channel). Each is
// followed by a SwiGLU MLP, with pre-norms around both. The RMS norms are zero centered (the
// weight w scales by 1 + w), except the gated norm at the end of the Gated DeltaNet.
//
// The checkpoints are image and text models (Qwen3_5ForConditionalGeneration, the text model under
// text_config and its tensors under model.language_model.); only the text model is converted.
// A text only checkpoint (Qwen3_5ForCausalLM, model_type qwen3_5_text) has them under model.
package qwen35

import (
	"cmp"
	"encoding/json"
	"fmt"
	"math"
)

// Config is the subset of the text model's config.json the conversion needs.
type Config struct {
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"head_dim"`
	VocabSize        int      `json:"vocab_size"`
	NormEps          float64  `json:"rms_norm_eps"`
	LayerTypes       []string `json:"layer_types"`
	TieWordEmbed     bool     `json:"tie_word_embeddings"`
	MaxPositions     int      `json:"max_position_embeddings"`
	HiddenAct        string   `json:"hidden_act"`
	AttentionBias    bool     `json:"attention_bias"`
	AttnOutputGate   *bool    `json:"attn_output_gate"`

	// The Gated DeltaNet layers: key heads and value heads (value heads a multiple of the key
	// heads, each key head shared by consecutive value heads), their sizes and the width of the
	// short convolution over the projected queries, keys and values.
	LinearKeyHeads   int `json:"linear_num_key_heads"`
	LinearValueHeads int `json:"linear_num_value_heads"`
	LinearKeyDim     int `json:"linear_key_head_dim"`
	LinearValueDim   int `json:"linear_value_head_dim"`
	ConvKernel       int `json:"linear_conv_kernel_dim"`

	RopeParameters struct {
		Theta         float64  `json:"rope_theta"`
		PartialFactor *float64 `json:"partial_rotary_factor"`
		Type          string   `json:"rope_type"`
		MropeSection  []int    `json:"mrope_section"`
	} `json:"rope_parameters"`

	// Mixture of experts settings of the MoE models (Qwen3.5-35B-A3B and up), rejected.
	NumExperts int `json:"num_experts"`

	// The multi-token prediction layers (mtp.), which the MTP model (-arch qwen35-mtp) reads:
	// how many, and whether they have an embedding table of their own.
	MTPLayers        int  `json:"mtp_num_hidden_layers"`
	MTPOwnEmbeddings bool `json:"mtp_use_dedicated_embeddings"`

	// Prefix begins the names of the text model's tensors: "model.language_model." in an image
	// and text checkpoint, "model." in a text only one.
	Prefix string `json:"-"`
}

// ParseConfig parses config.json and rejects settings the conversion does not handle.
func ParseConfig(raw []byte) (*Config, error) {
	var top struct {
		ModelType  string          `json:"model_type"`
		TextConfig json.RawMessage `json:"text_config"`
		Tie        *bool           `json:"tie_word_embeddings"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("qwen35: %w", err)
	}
	var c Config
	text := raw
	switch top.ModelType {
	case "qwen3_5":
		if len(top.TextConfig) == 0 {
			return nil, fmt.Errorf("qwen35: qwen3_5 config has no text_config")
		}
		var t struct {
			ModelType string `json:"model_type"`
		}
		if err := json.Unmarshal(top.TextConfig, &t); err != nil {
			return nil, fmt.Errorf("qwen35: text_config: %w", err)
		}
		if t.ModelType != "" && t.ModelType != "qwen3_5_text" {
			return nil, fmt.Errorf("qwen35: text_config model_type %q is not qwen3_5_text", t.ModelType)
		}
		text, c.Prefix = top.TextConfig, "model.language_model."
	case "qwen3_5_text":
		c.Prefix = "model."
	default:
		return nil, fmt.Errorf("qwen35: model_type %q is not supported", top.ModelType)
	}
	if err := json.Unmarshal(text, &c); err != nil {
		return nil, fmt.Errorf("qwen35: %w", err)
	}
	if top.Tie != nil && c.Prefix == "model.language_model." {
		c.TieWordEmbed = *top.Tie
	}
	if err := c.check(); err != nil {
		return nil, err
	}
	return &c, nil
}

// check fills the defaults and rejects what Build does not implement.
func (c *Config) check() error {
	if c.HeadDim == 0 && c.NumHeads > 0 {
		c.HeadDim = c.HiddenSize / c.NumHeads
	}
	c.NormEps = cmp.Or(c.NormEps, 1e-6)
	c.HiddenAct = cmp.Or(c.HiddenAct, "silu")
	if c.HiddenAct != "silu" {
		return fmt.Errorf("qwen35: hidden_act %q is not supported", c.HiddenAct)
	}
	if c.NumExperts > 0 {
		return fmt.Errorf("qwen35: mixture of experts (num_experts %d) is not supported", c.NumExperts)
	}
	if c.AttentionBias {
		return fmt.Errorf("qwen35: attention_bias is not supported")
	}
	if c.AttnOutputGate != nil && !*c.AttnOutputGate {
		return fmt.Errorf("qwen35: attention without its output gate (attn_output_gate false) is not supported")
	}
	if c.NumLayers <= 0 || len(c.LayerTypes) != c.NumLayers {
		return fmt.Errorf("qwen35: %d layer_types for %d layers", len(c.LayerTypes), c.NumLayers)
	}
	for _, t := range c.LayerTypes {
		if t != "linear_attention" && t != "full_attention" {
			return fmt.Errorf("qwen35: unknown layer type %q", t)
		}
	}
	if c.NumKVHeads <= 0 || c.NumHeads%c.NumKVHeads != 0 {
		return fmt.Errorf("qwen35: %d attention heads over %d kv heads", c.NumHeads, c.NumKVHeads)
	}
	if c.LinearKeyHeads <= 0 || c.LinearValueHeads%c.LinearKeyHeads != 0 {
		return fmt.Errorf("qwen35: %d linear value heads over %d key heads", c.LinearValueHeads, c.LinearKeyHeads)
	}
	if c.ConvKernel < 1 || c.LinearKeyDim <= 0 || c.LinearValueDim <= 0 {
		return fmt.Errorf("qwen35: linear attention sizes key %d, value %d, kernel %d", c.LinearKeyDim, c.LinearValueDim, c.ConvKernel)
	}
	r := c.RopeParameters
	if r.Type != "" && r.Type != "default" {
		return fmt.Errorf("qwen35: rope type %q is not supported", r.Type)
	}
	if r.Theta <= 0 {
		return fmt.Errorf("qwen35: rope_theta %g", r.Theta)
	}
	rot := c.RotaryDim()
	if rot <= 0 || rot%2 != 0 || rot > c.HeadDim {
		return fmt.Errorf("qwen35: rotary dimension %d of a head of %d", rot, c.HeadDim)
	}
	// The multimodal RoPE gives text tokens the same position on its three axes, which is the
	// one dimensional RoPE: its sections only need to cover the rotary pairs.
	if s := r.MropeSection; len(s) > 0 {
		sum := 0
		for _, v := range s {
			sum += v
		}
		if sum != rot/2 {
			return fmt.Errorf("qwen35: mrope_section %v does not cover the %d rotary pairs", s, rot/2)
		}
	}
	return nil
}

// RotaryDim is the leading dimensions of each head that RoPE rotates (partial_rotary_factor).
func (c *Config) RotaryDim() int {
	f := 1.0
	if p := c.RopeParameters.PartialFactor; p != nil {
		f = *p
	}
	return int(float64(c.HeadDim) * f)
}

// RopeInvFreq is the inverse frequencies of the rotary dimensions, RotaryDim / 2 of them, as
// Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters.
func (c *Config) RopeInvFreq() []float64 {
	dim := c.RotaryDim()
	inv := make([]float64, dim/2)
	for i := range inv {
		inv[i] = 1 / math.Pow(c.RopeParameters.Theta, float64(2*i)/float64(dim))
	}
	return inv
}

// IsLinear reports whether layer i is a Gated DeltaNet layer.
func (c *Config) IsLinear(i int) bool { return c.LayerTypes[i] == "linear_attention" }

// FullLayers are the full attention layers.
func (c *Config) FullLayers() []int {
	var out []int
	for i := range c.LayerTypes {
		if !c.IsLinear(i) {
			out = append(out, i)
		}
	}
	return out
}

// keyDim, valueDim and convDim are the Gated DeltaNet's widths: the keys (and queries) and values
// of all heads, and the convolution over the three.
func (c *Config) keyDim() int   { return c.LinearKeyHeads * c.LinearKeyDim }
func (c *Config) valueDim() int { return c.LinearValueHeads * c.LinearValueDim }
func (c *Config) convDim() int  { return 2*c.keyDim() + c.valueDim() }

// prefix is Prefix with its default.
func (c *Config) prefix() string { return cmp.Or(c.Prefix, "model.") }
