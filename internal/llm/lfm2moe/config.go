// Package lfm2moe converts Liquid AI's LFM2 mixture of experts models (Hugging Face
// `lfm2_moe`, Lfm2MoeForCausalLM: LFM2-8B-A1B) into Core AI programs for the Neural Engine.
//
// LFM2-MoE interleaves gated short convolutions and full attention as LFM2 does (QK RMS norms,
// RoPE, no gate on the heads, no sandwich norms). The first num_dense_layers layers end with a
// SwiGLU; the others with a mixture of experts: a router of sigmoid scores, the top k experts
// chosen with a bias of the layer (expert_bias) and weighed by their scores normalized, each a
// SwiGLU. The host picks the experts (llm.Routed): the programs are cut at each such layer, and
// the experts run in a program of their own between the parts, every layer's experts held in
// int8 (internal/llm/experts, docs/handoff-ane-gather.md 1b).
package lfm2moe

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm/experts"
)

// Config is the subset of config.json the conversion needs.
type Config struct {
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	MoEInter         int      `json:"moe_intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumDense         int      `json:"num_dense_layers"`
	NumExperts       int      `json:"num_experts"`
	TopK             int      `json:"num_experts_per_tok"`
	NormTopK         bool     `json:"norm_topk_prob"`
	RoutedScale      float64  `json:"routed_scaling_factor"`
	ExpertBias       bool     `json:"use_expert_bias"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"head_dim"`
	VocabSize        int      `json:"vocab_size"`
	NormEps          float64  `json:"norm_eps"`
	ConvKernel       int      `json:"conv_L_cache"`
	ConvBias         bool     `json:"conv_bias"`
	LayerTypes       []string `json:"layer_types"`
	TieWordEmbed     bool     `json:"tie_word_embeddings"`
	MaxPositions     int      `json:"max_position_embeddings"`
	RopeTheta        float64  `json:"rope_theta"`
}

// ParseConfig reads config.json and rejects settings the conversion does not handle.
func ParseConfig(raw []byte) (*Config, error) {
	var top struct {
		ModelType string           `json:"model_type"`
		Tie       *bool            `json:"tie_word_embeddings"`
		Rope      *json.RawMessage `json:"rope_scaling"`
		RopeP     *struct {
			Type  string  `json:"rope_type"`
			Theta float64 `json:"rope_theta"`
		} `json:"rope_parameters"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("lfm2moe: %w", err)
	}
	if top.ModelType != "lfm2_moe" {
		return nil, fmt.Errorf("lfm2moe: model_type %q is not lfm2_moe", top.ModelType)
	}
	c := Config{TieWordEmbed: true, RoutedScale: 1}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("lfm2moe: %w", err)
	}
	if top.Tie != nil {
		c.TieWordEmbed = *top.Tie
	}
	if top.Rope != nil && string(*top.Rope) != "null" {
		return nil, fmt.Errorf("lfm2moe: rope_scaling %s is not supported", *top.Rope)
	}
	if top.RopeP != nil {
		if top.RopeP.Type != "" && top.RopeP.Type != "default" {
			return nil, fmt.Errorf("lfm2moe: rope type %q is not supported", top.RopeP.Type)
		}
		if c.RopeTheta == 0 {
			c.RopeTheta = top.RopeP.Theta
		}
	}
	if c.RopeTheta == 0 {
		c.RopeTheta = 1e6
	}
	if c.HeadDim == 0 && c.NumHeads > 0 {
		c.HeadDim = c.HiddenSize / c.NumHeads
	}
	switch {
	case len(c.LayerTypes) != c.NumLayers:
		return nil, fmt.Errorf("lfm2moe: %d layer_types for %d layers", len(c.LayerTypes), c.NumLayers)
	case c.NumDense < 1 || c.NumDense >= c.NumLayers:
		return nil, fmt.Errorf("lfm2moe: %d dense layers of %d: layer 0 must be dense and one at least of experts", c.NumDense, c.NumLayers)
	case c.NumExperts < c.TopK || c.TopK < 1 || c.MoEInter < 1:
		return nil, fmt.Errorf("lfm2moe: top %d of %d experts of %d", c.TopK, c.NumExperts, c.MoEInter)
	case c.ConvBias:
		return nil, fmt.Errorf("lfm2moe: conv_bias is not supported")
	}
	for _, t := range c.LayerTypes {
		if t != "conv" && t != "full_attention" {
			return nil, fmt.Errorf("lfm2moe: unknown layer type %q", t)
		}
	}
	return &c, nil
}

// IsAttention reports whether layer i is an attention layer.
func (c *Config) IsAttention(i int) bool { return c.LayerTypes[i] != "conv" }

// IsRouted reports whether layer i ends with a mixture of experts.
func (c *Config) IsRouted(i int) bool { return i >= c.NumDense }

// RopeInvFreq is the inverse frequencies over the whole head dimension.
func (c *Config) RopeInvFreq() []float64 {
	half := c.HeadDim / 2
	inv := make([]float64, half)
	for i := range inv {
		inv[i] = 1 / math.Pow(c.RopeTheta, float64(2*i)/float64(c.HeadDim))
	}
	return inv
}

// RoutedLayers are the layers past the dense ones (llm.Routed).
func (c *Config) RoutedLayers() []int {
	var out []int
	for i := c.NumDense; i < c.NumLayers; i++ {
		out = append(out, i)
	}
	return out
}

// ExpertShape is the experts' sizes.
func (c *Config) ExpertShape() experts.Shape {
	return experts.Shape{Hidden: c.HiddenSize, Inter: c.MoEInter, Experts: c.NumExperts, TopK: c.TopK}
}

// Routing is sigmoid scores, the choice biased by expert_bias, the weights normalized.
func (c *Config) Routing() experts.Routing {
	return experts.Routing{Score: "sigmoid", Bias: c.ExpertBias, Norm: c.NormTopK, Scale: c.RoutedScale}
}

// ExpertNames are layer's experts' w1 (gate), w3 (up) and w2 (down), and its expert_bias.
func (c *Config) ExpertNames(layer int) experts.Names {
	p := fmt.Sprintf("model.layers.%d.feed_forward.", layer)
	n := experts.Names{
		Gate: func(e int) string { return fmt.Sprintf("%sexperts.%d.w1.weight", p, e) },
		Up:   func(e int) string { return fmt.Sprintf("%sexperts.%d.w3.weight", p, e) },
		Down: func(e int) string { return fmt.Sprintf("%sexperts.%d.w2.weight", p, e) },
	}
	if c.ExpertBias {
		n.Bias = p + "expert_bias"
	}
	return n
}

// UnpagedContext is the cache of the whole programs (their caches inside): the model is not
// split at its attention, being cut at its experts already.
func (c *Config) UnpagedContext() int { return 4096 }

// Format is lut8: LFM3's family loses its answers in lut4 on the Neural Engine
// (docs/design.md 16.31). The experts are int8 whatever it is.
func (c *Config) Format() string { return "lut8" }
