// Package qwen3 converts Qwen's dense Qwen3 models (Hugging Face `qwen3`, Qwen3ForCausalLM in
// transformers' modeling_qwen3.py, e.g. Qwen/Qwen3-0.6B) into Core ML programs for the Neural
// Engine and OpenVINO IR for Intel's NPU, from the bf16 safetensors checkpoint.
//
// Qwen3 is the plain decoder: RMS norms before the attention and the MLP, grouped query attention
// with an RMS norm over each head of the queries and the keys (q_norm, k_norm) before RoPE, and a
// SwiGLU MLP. Every layer attends to the whole context. The text encoder of Anima
// (internal/imagegen/anima) is the same structure without a cache or an output layer.
package qwen3

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Config is the subset of config.json the conversion needs. Missing fields take the defaults of
// transformers' Qwen3Config.
type Config struct {
	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	NumLayers        int     `json:"num_hidden_layers"`
	NumHeads         int     `json:"num_attention_heads"`
	NumKVHeads       int     `json:"num_key_value_heads"`
	HeadDim          int     `json:"head_dim"`
	VocabSize        int     `json:"vocab_size"`
	NormEps          float64 `json:"rms_norm_eps"`
	RopeTheta        float64 `json:"rope_theta"`
	MaxPositions     int     `json:"max_position_embeddings"`
	TieWordEmbed     bool    `json:"tie_word_embeddings"`

	// Settings the conversion rejects unless they are off.
	ModelType        string          `json:"model_type"`
	HiddenAct        string          `json:"hidden_act"`
	AttentionBias    bool            `json:"attention_bias"`
	UseSlidingWindow bool            `json:"use_sliding_window"`
	LayerTypes       []string        `json:"layer_types"`
	RopeScaling      json.RawMessage `json:"rope_scaling"`
	// RopeParameters is where transformers 5 writes rope_theta and the scaling.
	RopeParameters *struct {
		RopeType  string   `json:"rope_type"`
		RopeTheta *float64 `json:"rope_theta"`
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
	c := Config{
		HiddenSize: 4096, IntermediateSize: 22016, NumLayers: 32, NumHeads: 32, NumKVHeads: 32, HeadDim: 128,
		VocabSize: 151936, NormEps: 1e-6, RopeTheta: 10000, MaxPositions: 32768, HiddenAct: "silu",
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("qwen3: %w", err)
	}
	if p := c.RopeParameters; p != nil && p.RopeTheta != nil {
		c.RopeTheta = *p.RopeTheta
	}
	if err := c.check(); err != nil {
		return nil, fmt.Errorf("qwen3: %w", err)
	}
	return &c, nil
}

func (c *Config) check() error {
	switch {
	case c.ModelType != "qwen3":
		return fmt.Errorf("model_type %q is not qwen3", c.ModelType)
	case c.HiddenAct != "silu":
		return fmt.Errorf("hidden_act %q is not supported (Qwen3's MLP is SwiGLU)", c.HiddenAct)
	case c.AttentionBias:
		return fmt.Errorf("attention_bias is not supported")
	case c.UseSlidingWindow:
		return fmt.Errorf("use_sliding_window is not supported")
	case len(c.RopeScaling) > 0 && string(c.RopeScaling) != "null":
		return fmt.Errorf("rope_scaling %s is not supported", c.RopeScaling)
	case c.RopeParameters != nil && c.RopeParameters.RopeType != "" && c.RopeParameters.RopeType != "default":
		return fmt.Errorf("rope_type %q is not supported", c.RopeParameters.RopeType)
	case c.NumKVHeads <= 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("%d heads do not share %d key and value heads evenly", c.NumHeads, c.NumKVHeads)
	case c.HeadDim%2 != 0:
		return fmt.Errorf("head_dim %d is odd", c.HeadDim)
	case c.RopeTheta <= 0:
		return fmt.Errorf("rope_theta %g", c.RopeTheta)
	}
	for i, t := range c.LayerTypes {
		if t != "full_attention" {
			return fmt.Errorf("layer %d is %q: only full_attention is supported", i, t)
		}
	}
	return nil
}

// head names the output layer: lm_head, or the embeddings when they are tied.
func (c *Config) head() string {
	if c.TieWordEmbed {
		return c.Embedding()
	}
	return "lm_head.weight"
}

// RopeInvFreq returns the head_dim/2 inverse frequencies of the RoPE (Qwen3RotaryEmbedding's
// inv_freq, the default rope type).
func (c *Config) RopeInvFreq() []float64 {
	hd := c.HeadDim
	inv := make([]float64, hd/2)
	for j := range inv {
		inv[j] = 1 / math.Pow(c.RopeTheta, float64(2*j)/float64(hd))
	}
	return inv
}

// scoreScale multiplies the attention scores: 1/sqrt(head_dim).
func (c *Config) scoreScale() float64 { return 1 / math.Sqrt(float64(c.HeadDim)) }
