// Package plamo3 converts Preferred Networks' PLaMo 3 (Hugging Face `plamo3`, Plamo3ForCausalLM
// in the modeling_plamo.py the checkpoints ship, e.g. pfnet/plamo-3-translate-2b) into Core ML
// programs for the Neural Engine and OpenVINO IR for Intel's NPU, from the bf16 safetensors
// checkpoint.
//
// PLaMo 3 interleaves sliding window and full attention layers (every sliding_window_pattern-th
// layer is full), each wrapped in sandwich RMS norms and followed by a SwiGLU MLP. What sets it
// apart: its RMSNorm scales by offset + weight with an offset of its own per norm (1 before the
// blocks, 1/5 after the attention, 1/5^1.5 after the MLP), the projections are fused (qkv_proj,
// gate_up_proj), and the full layers' RoPE is YaRN, whose cos and sin carry a factor (the
// "concentration") that scales the attention scores.
package plamo3

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Config is the subset of config.json the conversion needs. Missing fields take the defaults of
// modeling_plamo.py's Plamo3Config.
type Config struct {
	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	NumLayers        int     `json:"num_hidden_layers"`
	NumHeads         int     `json:"num_attention_heads"`
	NumKVHeads       int     `json:"num_key_value_heads"`
	HeadDim          int     `json:"head_dim"`
	VocabSize        int     `json:"vocab_size"`
	NormEps          float64 `json:"rms_norm_eps"`
	Window           int     `json:"window_size"`
	Pattern          int     `json:"sliding_window_pattern"`
	RopeTheta        float64 `json:"rope_theta"`       // the full attention layers'
	RopeLocalTheta   float64 `json:"rope_local_theta"` // the sliding window layers'
	// RopeScalingFactor is YaRN's factor on the full layers (1: plain RoPE), over the first
	// InitialContext positions. modeling_plamo.py builds the RoPE settings from these two and
	// does not read config.json's rope_parameters.
	RopeScalingFactor float64 `json:"rope_scaling_factor"`
	InitialContext    *int    `json:"initial_context_length"`
	MaxPositions      int     `json:"max_position_embeddings"`
	TieWordEmbed      bool    `json:"tie_word_embeddings"`

	// Settings the conversion rejects unless they are off.
	ScaleEmbedding   bool    `json:"scale_embedding"`
	ImageFeatureSize *int    `json:"image_feature_size"`
	LinearType       string  `json:"linear_type"`
	ModelType        string  `json:"model_type"`
	HiddenAct        *string `json:"hidden_act"`
}

// The offsets of PLaMo's RMSNorms (Plamo3RMSNorm's offset): the norm scales by offset + weight.
const (
	offsetPre      = 1.0                 // pre_mixer_norm, pre_mlp_norm, q_norm, k_norm and the final norm
	offsetPostAttn = 1.0 / 5             // post_mixer_norm
	offsetPostMLP  = 0.08944271909999159 // post_mlp_norm: 1 / 5^1.5
)

// YaRN's beta_fast and beta_slow, fixed in modeling_plamo.py's rope_parameters.
const betaFast, betaSlow = 32.0, 1.0

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
		HiddenSize: 4096, IntermediateSize: 13312, NumLayers: 32, NumHeads: 32, NumKVHeads: 4, HeadDim: 128,
		VocabSize: 32000, NormEps: 1e-6, Window: 2048, Pattern: 8, RopeTheta: 150000, RopeLocalTheta: 10000,
		RopeScalingFactor: 1, MaxPositions: 2048, TieWordEmbed: true, LinearType: "normal",
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("plamo3: %w", err)
	}
	if err := c.check(); err != nil {
		return nil, fmt.Errorf("plamo3: %w", err)
	}
	return &c, nil
}

func (c *Config) check() error {
	switch {
	case c.ModelType != "plamo3":
		return fmt.Errorf("model_type %q is not plamo3", c.ModelType)
	case c.ScaleEmbedding:
		return fmt.Errorf("scale_embedding is not supported")
	case c.ImageFeatureSize != nil:
		return fmt.Errorf("the image projection (image_feature_size) is not supported")
	case c.LinearType != "normal":
		return fmt.Errorf("linear_type %q is not supported", c.LinearType)
	case c.HiddenAct != nil && *c.HiddenAct != "silu":
		return fmt.Errorf("hidden_act %q is not supported (PLaMo 3's MLP is SwiGLU)", *c.HiddenAct)
	case c.NumKVHeads <= 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("%d heads do not share %d key and value heads evenly", c.NumHeads, c.NumKVHeads)
	case c.Pattern <= 0:
		return fmt.Errorf("sliding_window_pattern %d", c.Pattern)
	case c.Window <= 0:
		return fmt.Errorf("window_size %d", c.Window)
	case c.HeadDim%2 != 0:
		return fmt.Errorf("head_dim %d is odd", c.HeadDim)
	case c.RopeScalingFactor < 1:
		return fmt.Errorf("rope_scaling_factor %g is below 1", c.RopeScalingFactor)
	case c.RopeScalingFactor != 1 && c.InitialContext == nil:
		return fmt.Errorf("rope_scaling_factor %g without initial_context_length", c.RopeScalingFactor)
	case c.TieWordEmbed && c.headRows() != c.VocabSize:
		return fmt.Errorf("a tied output layer of %d rows for a vocabulary of %d", c.headRows(), c.VocabSize)
	}
	if c.RopeScalingFactor != 1 {
		low, high := c.yarnRamp()
		if !(0 < low && low < high && high < float64(c.HeadDim/2-1)) {
			return fmt.Errorf("YaRN's ramp %g..%g is outside 0..%d", low, high, c.HeadDim/2-1)
		}
	}
	return nil
}

// Format is the weight format of the checkpoint for the Neural Engine (llm.Formatter): 4 bit
// tables for the 8B (hidden 4096) and larger, the architecture's lut8 for the smaller ones. The
// 2B breaks in lut4 (KL 1.91 nats on the CPU), but the 8B in lut4 translated 13 of 20 sentences
// word for word as in lut8 and the other 7 as correct rewordings, as did 4 paragraphs, in half the
// memory (3.77 GiB against 7.58) and twice as fast (docs/design.md 16.155).
func (c *Config) Format() string {
	if c.HiddenSize >= 4096 {
		return "lut4"
	}
	return ""
}

// IntelFormat is the weight format of the checkpoint for Intel's NPU (llm.IntelFormatter): int4
// by rows for the 8B and larger, the architecture's int8 for the smaller ones. The 8B's int8
// (7.6 GiB) is over the about 4.6 GB the NPU of a Core Ultra 5 135U held when it ran out of memory
// (docs/design.md 16.143); its int4 (3.8 GiB, one program with models.yaml's prefill off) is
// under it. Its precision is not measured: the 2B's int4 lost KL 0.72 nats, but the 8B kept its
// translations in lut4, which loses the 2B more (16.155).
func (c *Config) IntelFormat() string {
	if c.HiddenSize >= 4096 {
		return "int4"
	}
	return ""
}

// full reports whether layer i attends to the whole context (is_full_attn).
func (c *Config) full(i int) bool { return (i+1)%c.Pattern == 0 }

// headRows is the rows of lm_head: the vocabulary padded to a multiple of 16.
func (c *Config) headRows() int { return (c.VocabSize + 15) / 16 * 16 }

// head names the output layer: lm_head, or the embeddings when they are tied.
func (c *Config) head() string {
	if c.TieWordEmbed {
		return c.Embedding()
	}
	return "lm_head.weight"
}

// yarnRamp is where YaRN's ramp between interpolated and extrapolated frequencies starts and
// ends, in pairs of dimensions.
func (c *Config) yarnRamp() (low, high float64) {
	dh := float64(c.HeadDim / 2)
	orig := float64(*c.InitialContext)
	low = dh * math.Log(orig/(betaFast*2*math.Pi)) / math.Log(c.RopeTheta)
	high = dh * math.Log(orig/(betaSlow*2*math.Pi)) / math.Log(c.RopeTheta)
	return low, high
}

// RopeInvFreq returns the head_dim/2 inverse frequencies of layer i's RoPE (RotaryEmbedding's
// inv_freq): rope_local_theta's for a sliding layer; rope_theta's for a full one, under YaRN
// interpolated by the factor below the ramp's start, kept above its end, and mixed between.
func (c *Config) RopeInvFreq(full bool) []float64 {
	hd := c.HeadDim
	theta := c.RopeLocalTheta
	if full {
		theta = c.RopeTheta
	}
	inv := make([]float64, hd/2)
	for j := range inv {
		inv[j] = 1 / math.Pow(theta, float64(2*j)/float64(hd))
	}
	if !full || c.RopeScalingFactor == 1 {
		return inv
	}
	low, high := c.yarnRamp()
	for j := range inv {
		mask := 1 - min(max((float64(j)-low)/(high-low), 0), 1)
		inv[j] = inv[j]/c.RopeScalingFactor*(1-mask) + inv[j]*mask
	}
	return inv
}

// Concentration is the factor YaRN multiplies a full layer's cos and sin by (1 without YaRN):
// the queries and the keys both carry it, so the scores carry its square.
func (c *Config) Concentration() float64 {
	if c.RopeScalingFactor == 1 {
		return 1
	}
	return 0.1*math.Log(c.RopeScalingFactor) + 1
}

// scoreScale multiplies layer i's attention scores: 1/sqrt(head_dim), times the concentration
// squared on a full layer.
func (c *Config) scoreScale(full bool) float64 {
	s := 1 / math.Sqrt(float64(c.HeadDim))
	if full {
		k := c.Concentration()
		s *= k * k
	}
	return s
}

// cacheLen is the length of the KV cache of layer i in a program built with Options.Cache: the
// whole cache, or for a sliding layer the window when that is shorter (a ring the host writes at
// position mod length, internal/llm/decode).
func (c *Config) cacheLen(full bool, cache int) int {
	if !full && c.Window < cache {
		return c.Window
	}
	return cache
}
