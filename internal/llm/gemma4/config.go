// Package gemma4 converts the text model of Google's Gemma 4 (Hugging Face `gemma4`, the E2B
// and E4B sizes) into Core ML programs for the Neural Engine, from the bf16 safetensors
// checkpoint.
//
// Gemma 4 interleaves sliding window and full attention layers wrapped in sandwich RMS norms,
// each followed by a GeGLU MLP. Three things set it apart from the other architectures here:
// per-layer embeddings (a second embedding table with a slice for every layer, mixed into each
// layer's output), layers that reuse the keys and values of an earlier layer instead of
// computing their own (KV sharing), and proportional RoPE on the full attention layers, which
// rotates only the first quarter of the frequencies. The vision and audio towers of the
// multimodal checkpoint are not converted.
package gemma4

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"slices"
)

// Config is the subset of the text model's config the conversion needs. Fields missing from
// config.json take the defaults of transformers' Gemma4TextConfig.
type Config struct {
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"head_dim"`        // sliding attention layers
	GlobalHeadDim    int      `json:"global_head_dim"` // full attention layers
	VocabSize        int      `json:"vocab_size"`
	NormEps          float64  `json:"rms_norm_eps"`
	SlidingWindow    int      `json:"sliding_window"`
	LayerTypes       []string `json:"layer_types"`
	HiddenAct        string   `json:"hidden_activation"`
	Softcap          *float64 `json:"final_logit_softcapping"`
	TieWordEmbed     bool     `json:"tie_word_embeddings"`
	// PerLayerDim is the width of each layer's slice of the per-layer embeddings (0: none).
	PerLayerDim   int  `json:"hidden_size_per_layer_input"`
	PerLayerVocab int  `json:"vocab_size_per_layer_input"`
	KVShared      int  `json:"num_kv_shared_layers"`
	DoubleWideMLP bool `json:"use_double_wide_mlp"`

	// Settings the conversion rejects unless they are off.
	AttentionBias bool    `json:"attention_bias"`
	KEqV          bool    `json:"attention_k_eq_v"`
	MoE           bool    `json:"enable_moe_block"`
	Bidirectional *string `json:"use_bidirectional_attention"`

	RopeParameters map[string]RopeParams `json:"rope_parameters"`

	// Prefix is where the text model's tensors are: "model.language_model." in the multimodal
	// checkpoint (Gemma4ForConditionalGeneration), "model." in a text only one.
	Prefix string `json:"-"`
}

// RopeParams are the rotary embedding settings of a layer type.
type RopeParams struct {
	Theta         float64  `json:"rope_theta"`
	PartialFactor *float64 `json:"partial_rotary_factor"`
	Factor        *float64 `json:"factor"`
	Type          string   `json:"rope_type"`
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

// ParseConfig parses config.json, the multimodal one (model_type gemma4, whose text_config is
// the text model's) or a text only one (gemma4_text), and rejects settings the conversion does
// not handle.
func ParseConfig(raw []byte) (*Config, error) {
	var top struct {
		ModelType  string          `json:"model_type"`
		TextConfig json.RawMessage `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("gemma4: %w", err)
	}
	c := Config{
		HiddenSize: 2304, IntermediateSize: 9216, NumLayers: 30, NumHeads: 8, NumKVHeads: 4, HeadDim: 256,
		GlobalHeadDim: 512, VocabSize: 262144, NormEps: 1e-6, SlidingWindow: 512, HiddenAct: "gelu_pytorch_tanh",
		TieWordEmbed: true, PerLayerDim: 256, PerLayerVocab: 262144, Prefix: "model.",
	}
	switch top.ModelType {
	case "gemma4":
		if len(top.TextConfig) == 0 {
			return nil, fmt.Errorf("gemma4: config.json has no text_config")
		}
		raw, c.Prefix = top.TextConfig, "model.language_model."
	case "gemma4_text":
	default:
		return nil, fmt.Errorf("gemma4: model_type %q is not gemma4 or gemma4_text", top.ModelType)
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("gemma4: %w", err)
	}
	// An explicit null turns the per-layer embeddings off; a missing key takes the default.
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return nil, fmt.Errorf("gemma4: %w", err)
	}
	if v, ok := fields["hidden_size_per_layer_input"]; ok && bytes.Equal(bytes.TrimSpace(v), []byte("null")) {
		c.PerLayerDim = 0
	}
	if err := c.check(); err != nil {
		return nil, fmt.Errorf("gemma4: %w", err)
	}
	return &c, nil
}

func (c *Config) check() error {
	switch {
	case c.MoE:
		return fmt.Errorf("enable_moe_block (the mixture of experts sizes) is not supported")
	case c.KEqV:
		return fmt.Errorf("attention_k_eq_v is not supported")
	case c.AttentionBias:
		return fmt.Errorf("attention_bias is not supported")
	case c.Bidirectional != nil:
		return fmt.Errorf("use_bidirectional_attention %q is not supported", *c.Bidirectional)
	case !c.TieWordEmbed:
		return fmt.Errorf("an output layer apart from the embeddings (tie_word_embeddings false) is not supported")
	case c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("%d heads do not share %d key and value heads evenly", c.NumHeads, c.NumKVHeads)
	case c.PerLayerDim > 0 && c.PerLayerVocab != c.VocabSize:
		return fmt.Errorf("vocab_size_per_layer_input %d differs from vocab_size %d", c.PerLayerVocab, c.VocabSize)
	case c.KVShared < 0 || c.KVShared >= c.NumLayers:
		return fmt.Errorf("num_kv_shared_layers %d of %d layers", c.KVShared, c.NumLayers)
	}
	switch c.HiddenAct {
	case "gelu", "gelu_new", "gelu_pytorch_tanh", "relu", "silu":
	default:
		return fmt.Errorf("hidden_activation %q is not supported", c.HiddenAct)
	}
	if c.LayerTypes == nil {
		// Gemma4TextConfig's default: five sliding layers, then a full one.
		for i := 0; i < c.NumLayers; i++ {
			t := "sliding_attention"
			if (i+1)%6 == 0 {
				t = "full_attention"
			}
			c.LayerTypes = append(c.LayerTypes, t)
		}
	}
	if len(c.LayerTypes) != c.NumLayers {
		return fmt.Errorf("%d layer_types for %d layers", len(c.LayerTypes), c.NumLayers)
	}
	for _, t := range c.LayerTypes {
		if t != "sliding_attention" && t != "full_attention" {
			return fmt.Errorf("unknown layer type %q", t)
		}
	}
	// Gemma4TextConfig forces the last layer to full attention (with a warning).
	c.LayerTypes[c.NumLayers-1] = "full_attention"
	if c.RopeParameters == nil {
		quarter := 0.25
		c.RopeParameters = map[string]RopeParams{
			"sliding_attention": {Theta: 10000, Type: "default"},
			"full_attention":    {Theta: 1e6, PartialFactor: &quarter, Type: "proportional"},
		}
	}
	for _, t := range []string{"sliding_attention", "full_attention"} {
		if !slices.Contains(c.LayerTypes, t) {
			continue
		}
		p, ok := c.RopeParameters[t]
		switch {
		case !ok:
			return fmt.Errorf("no rope_parameters for %s", t)
		case p.Type == "default" && (p.PartialFactor != nil && *p.PartialFactor != 1 || p.Factor != nil && *p.Factor != 1):
			return fmt.Errorf("rope type default for %s with partial_rotary_factor or factor is not supported", t)
		case p.Type != "default" && p.Type != "proportional":
			return fmt.Errorf("rope type %q for %s is not supported", p.Type, t)
		}
	}
	for i := c.firstShared(); i < c.NumLayers && c.KVShared > 0; i++ {
		if c.source(c.LayerTypes[i]) < 0 {
			return fmt.Errorf("layer %d shares the keys and values of an earlier %s layer, and there is none", i, c.LayerTypes[i])
		}
	}
	return nil
}

// headDim is the head dimension of a layer type: full attention layers have wider heads.
func (c *Config) headDim(layerType string) int {
	if layerType == "full_attention" {
		return c.GlobalHeadDim
	}
	return c.HeadDim
}

// firstShared is the first layer that reuses an earlier layer's keys and values.
func (c *Config) firstShared() int { return c.NumLayers - c.KVShared }

// shared reports whether layer i reuses an earlier layer's keys and values (is_kv_shared_layer).
func (c *Config) shared(i int) bool { return c.KVShared > 0 && i >= c.firstShared() }

// source is the layer whose keys and values the shared layers of a type reuse: the last layer of
// that type before them (store_full_length_kv), or -1.
func (c *Config) source(layerType string) int {
	for i := c.firstShared() - 1; i >= 0; i-- {
		if c.LayerTypes[i] == layerType {
			return i
		}
	}
	return -1
}

// mlpWidth is the intermediate size of layer i: twice as wide in the shared layers with
// use_double_wide_mlp.
func (c *Config) mlpWidth(i int) int {
	if c.DoubleWideMLP && c.KVShared > 0 && i >= c.firstShared() && c.firstShared() > 0 {
		return 2 * c.IntermediateSize
	}
	return c.IntermediateSize
}

// RopeInvFreq returns the head_dim/2 inverse frequencies of a layer type. Proportional RoPE
// (_compute_proportional_rope_parameters) takes the first partial_rotary_factor * head_dim / 2
// frequencies of the full head dimension's progression and leaves the rest zero, so those
// dimensions are not rotated, and divides them all by factor.
func (c *Config) RopeInvFreq(layerType string) []float64 {
	p := c.RopeParameters[layerType]
	hd := c.headDim(layerType)
	angles := hd / 2
	factor := 1.0
	if p.Type == "proportional" {
		if p.PartialFactor != nil {
			angles = int(math.Floor(*p.PartialFactor * float64(hd) / 2))
		}
		if p.Factor != nil {
			factor = *p.Factor
		}
	}
	inv := make([]float64, hd/2)
	for i := 0; i < angles; i++ {
		inv[i] = 1 / math.Pow(p.Theta, float64(2*i)/float64(hd)) / factor
	}
	return inv
}

// cacheLen is the length of the KV cache of a layer of the given type in a program built with
// Options.Cache: the whole cache, or for a sliding layer the window when that is smaller (a ring
// the host writes at position mod length, internal/llm/decode).
func (c *Config) cacheLen(layerType string, cache int) int {
	if layerType == "sliding_attention" && c.SlidingWindow > 0 && c.SlidingWindow < cache {
		return c.SlidingWindow
	}
	return cache
}
