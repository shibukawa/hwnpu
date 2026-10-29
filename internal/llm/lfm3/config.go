// Package lfm3 converts Liquid AI's LFM3 language models (Hugging Face `lfm3` architecture) into
// Core ML programs for the Neural Engine, from either the bf16 safetensors checkpoint or a GGUF.
//
// LFM3 interleaves three kinds of layers: gated short convolutions, full attention and sliding
// window attention. Every layer is wrapped in sandwich RMS norms and followed by a SwiGLU MLP.
package lfm3

import (
	"encoding/json"
	"fmt"
	"os"
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

// ParseConfig parses config.json and rejects settings the conversion does not handle.
func ParseConfig(raw []byte) (*Config, error) {
	var c Config
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("lfm3: %w", err)
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
