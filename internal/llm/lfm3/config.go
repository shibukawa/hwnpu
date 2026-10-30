// Package lfm3 converts Liquid AI's LFM3 language models (Hugging Face `lfm3` architecture) into
// Core ML programs for the Neural Engine, from either the bf16 safetensors checkpoint or a GGUF.
//
// LFM3 interleaves three kinds of layers: gated short convolutions, full attention and sliding
// window attention. Every layer is wrapped in sandwich RMS norms and followed by a SwiGLU MLP.
//
// The text model of the multimodal checkpoints (LFM3-VL and LFM3-Omni, model_type lfm3_vl and
// lfm3_omni, with the LFM3 config under text_config; LFM3-Audio, lfm3_audio, under lfm) is the same, and so are Liquid AI's d1 decision models built on them
// (architectures D1Model): those read the whole prompt at once, the full attention layers both
// ways, and answer from the logits of its last token (Config.D1).
package lfm3

import (
	"cmp"
	"encoding/json"
	"fmt"
	"os"
	"slices"
)

// Config is the subset of config.json the conversion needs.
type Config struct {
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"head_dim"`
	VocabSize        int      `json:"vocab_size"`
	NormEps          float64  `json:"norm_eps"`
	ConvKernel       int      `json:"conv_L_cache"`
	ConvBias         bool     `json:"conv_bias"`
	SlidingWindow    int      `json:"sliding_window"`
	LayerTypes       []string `json:"layer_types"`
	UseSandwichNorm  bool     `json:"use_sandwich_norm"`
	TieWordEmbed     bool     `json:"tie_word_embeddings"`
	RopeParameters   map[string]struct {
		Theta         float64  `json:"rope_theta"`
		PartialFactor *float64 `json:"partial_rotary_factor"`
		Type          string   `json:"rope_type"`
	} `json:"rope_parameters"`

	// Prefix begins the names of the text model's tensors: "model." in Lfm3ForCausalLM (also
	// when empty), "model.language_model." in an LFM3-VL or LFM3-Omni checkpoint, "lfm.model."
	// in an LFM3-Audio one. HeadName is the output layer's, lm_head.weight when empty.
	Prefix   string `json:"-"`
	HeadName string `json:"-"`
	// D1 is a d1 decision model: one call reads the whole prompt with the full attention layers
	// bidirectional (a padding mask is the program's second input) and the sliding attention
	// layers and short convolutions causal, as llama.cpp runs it. The output is the final hidden
	// state; the host reads the logits of the answer labels at the last token from it.
	D1 bool `json:"-"`
	// Multimodal is the text model of an LFM3-VL, LFM3-Omni or LFM3-Audio checkpoint (d1 models
	// included), converted with lut8 when no format is given (Format).
	Multimodal bool `json:"-"`
}

// Format is lut8 for the text model of a multimodal checkpoint: with lut4, d1-2B's answers move
// away from the fp16 ones (the README's ticket goes to the wrong team), LFM3-VL describes
// pictures wrongly and LFM3-Audio's speech is no longer understood, while lut8 answers as fp16
// does (docs/design.md 16.31). LFM3 itself keeps the architecture's lut4.
func (c *Config) Format() string {
	if c.Multimodal {
		return "lut8"
	}
	return ""
}

// LoadConfig reads config.json.
func LoadConfig(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	c, err := ParseConfig(raw)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return c, nil
}

// The d1 serving settings this package implements; other values are rejected.
const (
	d1Codes     = 2
	d1Attention = "set"
)

// ParseConfig parses config.json and rejects settings the conversion does not handle.
func ParseConfig(raw []byte) (*Config, error) {
	var top struct {
		ModelType     string          `json:"model_type"`
		Architectures []string        `json:"architectures"`
		TextConfig    json.RawMessage `json:"text_config"`
		LFM           json.RawMessage `json:"lfm"`
		Tie           *bool           `json:"tie_word_embeddings"`
		D1Codes       *int            `json:"d1_codes"`
		D1Attention   *string         `json:"d1_attention"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("lfm3: %w", err)
	}
	var c Config
	text, nested := raw, false
	switch top.ModelType {
	case "lfm3_vl", "lfm3_omni", "lfm3_audio":
		// LFM3-Audio's release config has the text model under lfm (and its tensors under lfm.,
		// as llama.cpp reads them); the others, under text_config in the HF composite layout
		sub, key, prefix := top.TextConfig, "text_config", "model.language_model."
		if top.ModelType == "lfm3_audio" {
			sub, key, prefix = top.LFM, "lfm", "lfm.model."
			c.HeadName = "lfm.lm_head.weight"
		}
		if len(sub) == 0 {
			return nil, fmt.Errorf("lfm3: %s config has no %s", top.ModelType, key)
		}
		var t struct {
			ModelType string `json:"model_type"`
		}
		if err := json.Unmarshal(sub, &t); err != nil {
			return nil, fmt.Errorf("lfm3: %s: %w", key, err)
		}
		if t.ModelType != "" && t.ModelType != "lfm3" {
			return nil, fmt.Errorf("lfm3: %s model_type %q is not lfm3", key, t.ModelType)
		}
		text, nested, c.Prefix, c.Multimodal = sub, true, prefix, true
	case "", "lfm3":
	default:
		return nil, fmt.Errorf("lfm3: model_type %q is not supported", top.ModelType)
	}
	if err := json.Unmarshal(text, &c); err != nil {
		return nil, fmt.Errorf("lfm3: %w", err)
	}
	if nested && top.Tie != nil {
		c.TieWordEmbed = *top.Tie
	}
	if slices.Contains(top.Architectures, "D1Model") {
		// llama.cpp reads these from the merged config, so they may be in either
		var t struct {
			D1Codes     *int    `json:"d1_codes"`
			D1Attention *string `json:"d1_attention"`
		}
		json.Unmarshal(text, &t)
		codes, attn := cmp.Or(top.D1Codes, t.D1Codes), cmp.Or(top.D1Attention, t.D1Attention)
		if codes == nil || *codes != d1Codes || attn == nil || *attn != d1Attention {
			return nil, fmt.Errorf("lfm3: d1 model with d1_codes %v and d1_attention %v is not supported; expected %d and %q", deref(codes), deref(attn), d1Codes, d1Attention)
		}
		c.D1 = true
	}
	if c.HeadDim == 0 {
		c.HeadDim = c.HiddenSize / c.NumHeads
	}
	if len(c.LayerTypes) != c.NumLayers {
		return nil, fmt.Errorf("lfm3: %d layer_types for %d layers", len(c.LayerTypes), c.NumLayers)
	}
	for _, t := range c.LayerTypes {
		switch t {
		case "conv", "full_attention", "sliding_attention":
		default:
			return nil, fmt.Errorf("lfm3: unknown layer type %q", t)
		}
	}
	for _, t := range []string{"full_attention", "sliding_attention"} {
		if p, ok := c.RopeParameters[t]; ok && p.Type != "default" && p.Type != "" {
			return nil, fmt.Errorf("lfm3: rope type %q for %s is not supported", p.Type, t)
		}
	}
	return &c, nil
}

// prefix is Prefix with its default.
func (c *Config) prefix() string { return cmp.Or(c.Prefix, "model.") }

func deref[T any](p *T) any {
	if p == nil {
		return nil
	}
	return *p
}

// IsAttention reports whether layer i is an attention layer.
func (c *Config) IsAttention(i int) bool { return c.LayerTypes[i] != "conv" }

// RopeInvFreq returns the inverse frequencies of a layer type in LFM3's zero tail layout: the
// first partial_rotary_factor * head_dim / 2 entries follow the full head dimension's
// progression and the rest are zero, so those dimensions are not rotated.
func (c *Config) RopeInvFreq(layerType string) []float64 {
	p := c.RopeParameters[layerType]
	factor := 1.0
	if p.PartialFactor != nil {
		factor = *p.PartialFactor
	}
	half := c.HeadDim / 2
	angles := int(factor*float64(c.HeadDim)) / 2
	inv := make([]float64, half)
	for i := 0; i < angles; i++ {
		inv[i] = 1 / pow(p.Theta, float64(2*i)/float64(c.HeadDim))
	}
	return inv
}
