// Package gptoss converts OpenAI's gpt-oss models (Hugging Face `gpt_oss`, GptOssForCausalLM in
// transformers' modeling_gpt_oss.py, e.g. openai/gpt-oss-20b) into Core AI programs for Apple's
// Neural Engine, from the safetensors checkpoint whose experts are MXFP4.
//
// gpt-oss is a mixture of experts: 32 experts a layer, 4 routed to a token by a linear router
// and a softmax over their logits. The Neural Engine holds about 128 loaded programs for the
// whole machine, each function counting (docs/design.md 16.229), so the 768 experts of
// gpt-oss-20b cannot be a program each and routed on the host: here a layer computes all its
// experts for every token and weighs them by the routing, which the program computes itself
// (zero for an expert not among the top 4). The Neural Engine then reads every expert's weights
// each token, in 4 bit tables: about 10 GB, over the 2.8 GiB an M1's Neural Engine holds before
// it loads programs again at every call (also through Core AI: 24 layers' experts 2.37 s a round
// against 0.23 s), so on an M1 it is not practical; the M3 and M4 are not measured at that size.
//
// The attention is grouped, with biases on q, k, v and o, YaRN's RoPE on every layer and a
// learned logit per head (the sink) that takes part in the softmax and is dropped from its
// output. The layers alternate between a window of 128 positions (sliding_attention), whose
// cache is a ring in the program, and the whole context (full_attention), whose attention the
// host runs over its cache in chunks (llm.Paged), adding the sink when it merges them.
package gptoss

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Config is the subset of config.json the conversion needs. Missing fields take the defaults of
// transformers' GptOssConfig.
type Config struct {
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"head_dim"`
	VocabSize        int      `json:"vocab_size"`
	NormEps          float64  `json:"rms_norm_eps"`
	RopeTheta        float64  `json:"rope_theta"`
	MaxPositions     int      `json:"max_position_embeddings"`
	Window           int      `json:"sliding_window"`
	LayerTypes       []string `json:"layer_types"`
	Experts          int      `json:"num_local_experts"`
	TopK             int      `json:"num_experts_per_tok"`
	SwigluLimit      float64  `json:"swiglu_limit"`
	TieWordEmbed     bool     `json:"tie_word_embeddings"`
	AttentionBias    bool     `json:"attention_bias"`

	RopeScaling *ropeScaling `json:"rope_scaling"`
	// RopeParameters is where transformers 5 writes rope_theta and the scaling.
	RopeParameters *ropeScaling `json:"rope_parameters"`
	// Quantization is the checkpoint's quantization_config: mxfp4 experts, or none (bf16).
	Quantization *struct {
		QuantMethod string `json:"quant_method"`
	} `json:"quantization_config"`

	// Settings the conversion rejects unless they are what it handles.
	ModelType string `json:"model_type"`
}

// ropeScaling is rope_scaling (or transformers 5's rope_parameters).
type ropeScaling struct {
	RopeType     string   `json:"rope_type"`
	RopeTheta    *float64 `json:"rope_theta"`
	Factor       float64  `json:"factor"`
	BetaFast     *float64 `json:"beta_fast"`
	BetaSlow     *float64 `json:"beta_slow"`
	OriginalMax  int      `json:"original_max_position_embeddings"`
	Truncate     *bool    `json:"truncate"`
	Mscale       *float64 `json:"mscale"`
	MscaleAllDim *float64 `json:"mscale_all_dim"`
	AttnFactor   *float64 `json:"attention_factor"`
}

// alpha is the gate's sigmoid scale in GptOssExperts (_apply_gate's alpha, a constant).
const alpha = 1.702

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
		HiddenSize: 2880, IntermediateSize: 2880, NumLayers: 36, NumHeads: 64, NumKVHeads: 8, HeadDim: 64,
		VocabSize: 201088, NormEps: 1e-5, RopeTheta: 150000, MaxPositions: 131072, Window: 128,
		Experts: 128, TopK: 4, SwigluLimit: 7, AttentionBias: true,
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("gptoss: %w", err)
	}
	if c.RopeScaling == nil && c.RopeParameters != nil {
		c.RopeScaling = c.RopeParameters
	}
	if p := c.RopeScaling; p != nil && p.RopeTheta != nil {
		c.RopeTheta = *p.RopeTheta
	}
	if len(c.LayerTypes) == 0 {
		// GptOssConfig's default: sliding and full in turn, from a sliding layer
		c.LayerTypes = make([]string, c.NumLayers)
		for i := range c.LayerTypes {
			c.LayerTypes[i] = "sliding_attention"
			if i%2 == 1 {
				c.LayerTypes[i] = "full_attention"
			}
		}
	}
	if err := c.check(); err != nil {
		return nil, fmt.Errorf("gptoss: %w", err)
	}
	return &c, nil
}

func (c *Config) check() error {
	switch {
	case c.ModelType != "gpt_oss":
		return fmt.Errorf("model_type %q is not gpt_oss", c.ModelType)
	case !c.AttentionBias:
		return fmt.Errorf("attention_bias false is not supported (gpt-oss's projections have biases)")
	case c.TieWordEmbed:
		return fmt.Errorf("tie_word_embeddings is not supported (gpt-oss's lm_head is its own)")
	case c.NumKVHeads <= 0 || c.NumHeads%c.NumKVHeads != 0:
		return fmt.Errorf("%d heads do not share %d key and value heads evenly", c.NumHeads, c.NumKVHeads)
	case c.HeadDim%2 != 0:
		return fmt.Errorf("head_dim %d is odd", c.HeadDim)
	case c.RopeTheta <= 0:
		return fmt.Errorf("rope_theta %g", c.RopeTheta)
	case c.Experts <= 0 || c.TopK <= 0 || c.TopK > c.Experts:
		return fmt.Errorf("%d experts per token of %d", c.TopK, c.Experts)
	case c.Window <= 0:
		return fmt.Errorf("sliding_window %d", c.Window)
	case len(c.LayerTypes) != c.NumLayers:
		return fmt.Errorf("%d layer_types for %d layers", len(c.LayerTypes), c.NumLayers)
	case c.SwigluLimit <= 0:
		return fmt.Errorf("swiglu_limit %g", c.SwigluLimit)
	}
	if q := c.Quantization; q != nil && q.QuantMethod != "mxfp4" {
		return fmt.Errorf("quantization_config %q is not supported (mxfp4 or none)", q.QuantMethod)
	}
	for i, t := range c.LayerTypes {
		if t != "full_attention" && t != "sliding_attention" {
			return fmt.Errorf("layer %d is %q: only full_attention and sliding_attention are supported", i, t)
		}
	}
	if s := c.RopeScaling; s != nil {
		switch s.RopeType {
		case "yarn":
			if s.Factor < 1 {
				return fmt.Errorf("YaRN factor %g", s.Factor)
			}
			if s.Mscale != nil || s.MscaleAllDim != nil {
				return fmt.Errorf("YaRN's mscale and mscale_all_dim are not supported")
			}
		case "", "default":
		default:
			return fmt.Errorf("rope_type %q is not supported (yarn or default)", s.RopeType)
		}
	}
	return nil
}

// full reports whether layer i attends to the whole context (full_attention) rather than a
// window (sliding_attention).
func (c *Config) full(i int) bool { return c.LayerTypes[i] == "full_attention" }

// mxfp4 reports whether the checkpoint's experts are MXFP4 (blocks and scales) rather than dense.
func (c *Config) mxfp4() bool { return c.Quantization != nil && c.Quantization.QuantMethod == "mxfp4" }

// yarn reports whether the RoPE is YaRN's.
func (c *Config) yarn() bool { return c.RopeScaling != nil && c.RopeScaling.RopeType == "yarn" }

// RopeInvFreq returns the head_dim/2 inverse frequencies of the RoPE, YaRN's as transformers'
// _compute_yarn_parameters makes them: interpolated by the factor below the ramp, extrapolated
// above it, mixed between.
func (c *Config) RopeInvFreq() []float64 {
	hd := c.HeadDim
	inv := make([]float64, hd/2)
	for j := range inv {
		inv[j] = 1 / math.Pow(c.RopeTheta, float64(2*j)/float64(hd))
	}
	if !c.yarn() {
		return inv
	}
	s := c.RopeScaling
	factor, orig := s.Factor, c.MaxPositions
	if s.OriginalMax > 0 {
		// as transformers does: the factor is the ratio of the contexts when both are given
		orig = s.OriginalMax
		factor = float64(c.MaxPositions) / float64(orig)
	}
	betaFast, betaSlow := 32.0, 1.0
	if s.BetaFast != nil {
		betaFast = *s.BetaFast
	}
	if s.BetaSlow != nil {
		betaSlow = *s.BetaSlow
	}
	dim := float64(hd)
	correction := func(rotations float64) float64 {
		return dim * math.Log(float64(orig)/(rotations*2*math.Pi)) / (2 * math.Log(c.RopeTheta))
	}
	low, high := correction(betaFast), correction(betaSlow)
	if s.Truncate == nil || *s.Truncate {
		low, high = math.Floor(low), math.Ceil(high)
	}
	low, high = max(low, 0), min(high, dim-1)
	if low == high {
		high += 0.001
	}
	for j := range inv {
		ramp := min(max((float64(j)-low)/(high-low), 0), 1)
		extrapolate := 1 - ramp
		inv[j] = inv[j]/factor*(1-extrapolate) + inv[j]*extrapolate
	}
	return inv
}

// Concentration is YaRN's attention_factor, which multiplies cos and sin (1 without YaRN): the
// queries and the keys both carry it, so the scores carry its square.
func (c *Config) Concentration() float64 {
	if !c.yarn() {
		return 1
	}
	if a := c.RopeScaling.AttnFactor; a != nil {
		return *a
	}
	factor := c.RopeScaling.Factor
	if c.RopeScaling.OriginalMax > 0 {
		factor = float64(c.MaxPositions) / float64(c.RopeScaling.OriginalMax)
	}
	if factor <= 1 {
		return 1
	}
	return 0.1*math.Log(factor) + 1
}

// scoreScale multiplies the attention scores: 1/sqrt(head_dim) times the concentration squared.
// The rotary tables leave the concentration out, so that the cached keys do not carry it.
func (c *Config) scoreScale() float64 {
	k := c.Concentration()
	return k * k / math.Sqrt(float64(c.HeadDim))
}

// cacheLen is the length of layer i's KV cache in a program built with Options.Cache: the whole
// cache, or for a sliding layer the window when that is shorter (a ring the host writes at
// position mod length, internal/llm/decode).
func (c *Config) cacheLen(full bool, cache int) int {
	if !full && c.Window < cache {
		return c.Window
	}
	return cache
}
