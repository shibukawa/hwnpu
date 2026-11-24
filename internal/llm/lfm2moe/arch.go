package lfm2moe

import (
	"fmt"

	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
)

// Arch registers LFM2-MoE with the whitelist in internal/llm.
var Arch = &llm.Arch{
	Name:          "lfm2moe",
	Title:         "LFM2-MoE",
	ModelTypes:    []string{"lfm2_moe"},
	Architectures: []string{"Lfm2MoeForCausalLM"},
	LoadConfig:    func(raw []byte) (llm.Model, error) { return ParseConfig(raw) },
	LastRow:       true,
}

func init() { llm.Register(Arch) }

var (
	_ llm.Routed    = (*Config)(nil)
	_ llm.Paged     = (*Config)(nil)
	_ llm.Formatter = (*Config)(nil)
)

// Hidden is the hidden size.
func (c *Config) Hidden() int { return c.HiddenSize }

// Layers is the number of layers.
func (c *Config) Layers() int { return c.NumLayers }

// Output is the logits.
func (c *Config) Output() (int, bool) { return c.VocabSize, true }

// Embedding names the token embedding table.
func (c *Config) Embedding() string { return "model.embed_tokens.weight" }

// Head names the output layer's weight: the embedding table when they are tied.
func (c *Config) Head() string {
	if c.TieWordEmbed {
		return c.Embedding()
	}
	return "lm_head.weight"
}

// Tensors lists the tensors the programs read: not the experts', which the conversion writes into
// a file of their own (llm.Routed).
func (c *Config) Tensors() map[string][]int {
	d, hd := c.HiddenSize, c.HeadDim
	t := map[string][]int{
		c.Embedding():                 {c.VocabSize, d},
		"model.embedding_norm.weight": {d},
	}
	if !c.TieWordEmbed {
		t[c.Head()] = []int{c.VocabSize, d}
	}
	for i := 0; i < c.NumLayers; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		t[p+"operator_norm.weight"] = []int{d}
		t[p+"ffn_norm.weight"] = []int{d}
		if c.IsRouted(i) {
			t[p+"feed_forward.gate.weight"] = []int{c.NumExperts, d}
		} else {
			t[p+"feed_forward.w1.weight"] = []int{c.IntermediateSize, d}
			t[p+"feed_forward.w3.weight"] = []int{c.IntermediateSize, d}
			t[p+"feed_forward.w2.weight"] = []int{d, c.IntermediateSize}
		}
		if c.IsAttention(i) {
			a := p + "self_attn."
			t[a+"q_proj.weight"] = []int{c.NumHeads * hd, d}
			t[a+"k_proj.weight"] = []int{c.NumKVHeads * hd, d}
			t[a+"v_proj.weight"] = []int{c.NumKVHeads * hd, d}
			t[a+"out_proj.weight"] = []int{d, c.NumHeads * hd}
			t[a+"q_layernorm.weight"] = []int{hd}
			t[a+"k_layernorm.weight"] = []int{hd}
		} else {
			t[p+"conv.in_proj.weight"] = []int{3 * d, d}
			t[p+"conv.out_proj.weight"] = []int{d, d}
			t[p+"conv.conv.weight"] = []int{d, 1, c.ConvKernel}
		}
	}
	return t
}

// Build writes a program without a KV cache (a part, as llm.Routed cuts it).
func (c *Config) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Cache > 0 {
		return nil, out, fmt.Errorf("lfm2moe: a program with a KV cache has more inputs and outputs; use BuildCached")
	}
	i, o, err := c.BuildCached(b, w, opts)
	if err != nil {
		return nil, out, err
	}
	return i, o[0], nil
}

// BuildCached writes a part of the model (llm.Cached, llm.Routed).
func (c *Config) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	return build(b, c, w, opts)
}
