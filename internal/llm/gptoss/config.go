// Package gptoss converts OpenAI's gpt-oss (Hugging Face `gpt_oss`, GptOssForCausalLM:
// gpt-oss-20b) into Core AI programs for the Neural Engine, its experts run by the host.
//
// gpt-oss alternates sliding window (128) and full attention layers, every one with biases on
// its projections and a learned logit per head that the softmax includes (a sink, which takes
// weight and gives no value), RoPE with YaRN over the two halves of a head. Every layer ends with
// a mixture of experts: a router with a bias, the top 4 of 32 experts weighed by the softmax of
// their logits, each a gated linear unit with biases whose gate and up interleave (gate clamped
// to 7, up to ±7, gate·sigmoid(1.702·gate)·(up+1)). The experts are in MXFP4 as released; the
// host picks them (llm.Routed) and runs them on its CPU from the checkpoint as it is
// (internal/mxfp4), the programs cut at every layer's router.
package gptoss

import (
	"encoding/json"
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/llm/experts"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
)

// Config is the subset of config.json the conversion needs.
type Config struct {
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumExperts       int      `json:"num_local_experts"`
	TopK             int      `json:"num_experts_per_tok"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"head_dim"`
	VocabSize        int      `json:"vocab_size"`
	NormEps          float64  `json:"rms_norm_eps"`
	LayerTypes       []string `json:"layer_types"`
	SlidingWindow    int      `json:"sliding_window"`
	SwigluLimit      float64  `json:"swiglu_limit"`
	SwigluAlpha      float64  `json:"swiglu_alpha"`
	RopeTheta        float64  `json:"rope_theta"`
	MaxPositions     int      `json:"max_position_embeddings"`
	TieWordEmbed     bool     `json:"tie_word_embeddings"`
	AttentionBias    bool     `json:"attention_bias"`
	Yarn             Yarn     `json:"-"`
}

// Yarn is rope_scaling's YaRN.
type Yarn struct {
	Factor, BetaFast, BetaSlow float64
	Original                   int
	Truncate                   bool
}

// ParseConfig reads config.json and rejects settings the conversion does not handle.
func ParseConfig(raw []byte) (*Config, error) {
	var top struct {
		ModelType string `json:"model_type"`
		Rope      *struct {
			Type     string   `json:"rope_type"`
			Factor   float64  `json:"factor"`
			BetaFast float64  `json:"beta_fast"`
			BetaSlow float64  `json:"beta_slow"`
			Original int      `json:"original_max_position_embeddings"`
			Truncate *bool    `json:"truncate"`
			Theta    float64  `json:"rope_theta"`
			Attn     *float64 `json:"attention_factor"`
		} `json:"rope_scaling"`
		Quant *struct {
			Method string `json:"quant_method"`
		} `json:"quantization_config"`
		TopK2 int `json:"experts_per_token"`
	}
	if err := json.Unmarshal(raw, &top); err != nil {
		return nil, fmt.Errorf("gptoss: %w", err)
	}
	if top.ModelType != "gpt_oss" {
		return nil, fmt.Errorf("gptoss: model_type %q is not gpt_oss", top.ModelType)
	}
	c := Config{SwigluAlpha: 1.702, SwigluLimit: 7, RopeTheta: 150000, NormEps: 1e-5, AttentionBias: true}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("gptoss: %w", err)
	}
	if c.TopK == 0 {
		c.TopK = top.TopK2
	}
	if c.HeadDim == 0 && c.NumHeads > 0 {
		c.HeadDim = c.HiddenSize / c.NumHeads
	}
	if top.Quant != nil && top.Quant.Method != "mxfp4" {
		return nil, fmt.Errorf("gptoss: quant_method %q is not mxfp4", top.Quant.Method)
	}
	if r := top.Rope; r != nil {
		if r.Type != "yarn" {
			return nil, fmt.Errorf("gptoss: rope type %q is not supported", r.Type)
		}
		if r.Attn != nil {
			return nil, fmt.Errorf("gptoss: rope_scaling's attention_factor is not supported")
		}
		c.Yarn = Yarn{Factor: r.Factor, BetaFast: r.BetaFast, BetaSlow: r.BetaSlow, Original: r.Original, Truncate: true}
		if r.Truncate != nil {
			c.Yarn.Truncate = *r.Truncate
		}
		if c.Yarn.BetaFast == 0 {
			c.Yarn.BetaFast = 32
		}
		if c.Yarn.BetaSlow == 0 {
			c.Yarn.BetaSlow = 1
		}
	}
	switch {
	case len(c.LayerTypes) != c.NumLayers:
		return nil, fmt.Errorf("gptoss: %d layer_types for %d layers", len(c.LayerTypes), c.NumLayers)
	case c.NumExperts < c.TopK || c.TopK < 1:
		return nil, fmt.Errorf("gptoss: top %d of %d experts", c.TopK, c.NumExperts)
	case !c.AttentionBias:
		return nil, fmt.Errorf("gptoss: attention without biases is not supported")
	case c.HiddenSize%32 != 0 || c.IntermediateSize%32 != 0:
		return nil, fmt.Errorf("gptoss: the hidden size %d and intermediate size %d are not of MXFP4's blocks of 32", c.HiddenSize, c.IntermediateSize)
	}
	for _, t := range c.LayerTypes {
		if t != "sliding_attention" && t != "full_attention" {
			return nil, fmt.Errorf("gptoss: unknown layer type %q", t)
		}
	}
	if len(c.PagedLayers()) < c.NumLayers && c.SlidingWindow <= 0 {
		return nil, fmt.Errorf("gptoss: no sliding_window")
	}
	return &c, nil
}

// IsSliding reports whether layer i attends over the sliding window.
func (c *Config) IsSliding(i int) bool { return c.LayerTypes[i] == "sliding_attention" }

// RopeInvFreq is YaRN's inverse frequencies over the head (transformers' _compute_yarn_parameters)
// and the factor the cos and sin tables are scaled by.
func (c *Config) RopeInvFreq() ([]float64, float64) {
	dim := c.HeadDim
	half := dim / 2
	inv := make([]float64, half)
	y := c.Yarn
	if y.Factor <= 1 {
		for i := range inv {
			inv[i] = 1 / math.Pow(c.RopeTheta, float64(2*i)/float64(dim))
		}
		return inv, 1
	}
	corr := func(rot float64) float64 {
		return float64(dim) * math.Log(float64(y.Original)/(rot*2*math.Pi)) / (2 * math.Log(c.RopeTheta))
	}
	low, high := corr(y.BetaFast), corr(y.BetaSlow)
	if y.Truncate {
		low, high = math.Floor(low), math.Ceil(high)
	}
	low, high = max(low, 0), min(high, float64(dim-1))
	if low == high {
		high += 0.001
	}
	for i := range inv {
		pos := math.Pow(c.RopeTheta, float64(2*i)/float64(dim))
		extra, inter := 1/pos, 1/(y.Factor*pos)
		ramp := min(max((float64(i)-low)/(high-low), 0), 1)
		ef := 1 - ramp // the share of extrapolation
		inv[i] = inter*(1-ef) + extra*ef
	}
	return inv, 0.1*math.Log(y.Factor) + 1
}

// RoutedLayers are every layer (llm.Routed).
func (c *Config) RoutedLayers() []int {
	out := make([]int, c.NumLayers)
	for i := range out {
		out[i] = i
	}
	return out
}

// ExpertShape is the experts' sizes.
func (c *Config) ExpertShape() experts.Shape {
	return experts.Shape{Hidden: c.HiddenSize, Inter: c.IntermediateSize, Experts: c.NumExperts, TopK: c.TopK}
}

// Routing is the softmax of the chosen experts' logits: the softmax over all, renormalized over
// the top k.
func (c *Config) Routing() experts.Routing { return experts.Routing{Score: "softmax", Norm: true} }

// ExpertNames has no experts of their own names: gpt-oss's are in MXFP4 tensors of all the layer's
// experts, which the host runs as they are (HostExperts).
func (c *Config) ExpertNames(layer int) experts.Names { return experts.Names{} }

// HostExperts are layer's experts in the checkpoint, MXFP4, run by the host (llm.HostRouted).
func (c *Config) HostExperts(layer int) experts.MXFP4 {
	p := fmt.Sprintf("model.layers.%d.mlp.experts.", layer)
	return experts.MXFP4{
		GateUpBlocks: p + "gate_up_proj_blocks", GateUpScales: p + "gate_up_proj_scales", GateUpBias: p + "gate_up_proj_bias",
		DownBlocks: p + "down_proj_blocks", DownScales: p + "down_proj_scales", DownBias: p + "down_proj_bias",
	}
}

// ExpertAct is gpt-oss's gated unit.
func (c *Config) ExpertAct() experts.Act {
	return experts.Act{Kind: experts.ActGptOss, Alpha: c.SwigluAlpha, Limit: c.SwigluLimit}
}

// PagedLayers are the full attention layers: split at them, the host attends over its cache in
// chunks; the sliding layers keep rings of the window in the program.
func (c *Config) PagedLayers() []int {
	var out []int
	for i := range c.LayerTypes {
		if !c.IsSliding(i) {
			out = append(out, i)
		}
	}
	return out
}

// PagedShape is the attention's heads, kv heads and head size, its scores scaled by 1/sqrt(head_dim).
func (c *Config) PagedShape() paged.Shape {
	return paged.Shape{Heads: c.NumHeads, KVHeads: c.NumKVHeads, HeadDim: c.HeadDim}
}

// PagedKV is the layer itself.
func (c *Config) PagedKV(layer int) int { return layer }

// MaxContext is max_position_embeddings: 131072 for gpt-oss-20b.
func (c *Config) MaxContext() int { return c.MaxPositions }

// RingWindow is the sliding layers' window, the length of their rings (llm.Windowed).
func (c *Config) RingWindow() int { return c.SlidingWindow }

// Format is lut8: the attention, the router and the head; the experts are the checkpoint's.
func (c *Config) Format() string { return "lut8" }

// cacheLen is a layer's cache in a program of Options.Cache: the window for a sliding layer.
func (c *Config) cacheLen(i, cache int) int {
	if c.IsSliding(i) && c.SlidingWindow < cache {
		return c.SlidingWindow
	}
	return cache
}
