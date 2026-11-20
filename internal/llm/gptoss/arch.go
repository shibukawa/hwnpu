package gptoss

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch registers gpt-oss with the whitelist in internal/llm.
var Arch = &llm.Arch{
	Name:          "gptoss",
	Title:         "gpt-oss",
	ModelTypes:    []string{"gpt_oss"},
	Architectures: []string{"GptOssForCausalLM"},
	LoadConfig:    func(raw []byte) (llm.Model, error) { return ParseConfig(raw) },
	LastRow:       true,
	// 1: the attention's projections fp16 (build), 2026-11-19
	// 2: fp16 for the first 8 layers, int8 for the rest, 2026-11-19
	// 3: the attention lut8r (defaultFormats), 2026-11-19
	// 4: the head lut6r, 2026-11-19
	Revision: 4,
}

func init() { llm.Register(Arch) }

var (
	_ llm.HostRouted = (*Config)(nil)
	_ llm.Paged      = (*Config)(nil)
	_ llm.Windowed   = (*Config)(nil)
	_ llm.Formatter  = (*Config)(nil)
)

// Hidden is the hidden size.
func (c *Config) Hidden() int { return c.HiddenSize }

// Layers is the number of layers.
func (c *Config) Layers() int { return c.NumLayers }

// Output is the logits.
func (c *Config) Output() (int, bool) { return c.VocabSize, true }

// Embedding names the token embedding table.
func (c *Config) Embedding() string { return "model.embed_tokens.weight" }

// Head names the output layer's weight.
func (c *Config) Head() string {
	if c.TieWordEmbed {
		return c.Embedding()
	}
	return "lm_head.weight"
}

// Tensors lists the tensors the programs read: not the experts', which the host runs from the
// checkpoint (llm.HostRouted).
func (c *Config) Tensors() map[string][]int {
	d, hd := c.HiddenSize, c.HeadDim
	t := map[string][]int{
		c.Embedding():       {c.VocabSize, d},
		"model.norm.weight": {d},
	}
	if !c.TieWordEmbed {
		t[c.Head()] = []int{c.VocabSize, d}
	}
	for i := 0; i < c.NumLayers; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		a := p + "self_attn."
		t[p+"input_layernorm.weight"] = []int{d}
		t[p+"post_attention_layernorm.weight"] = []int{d}
		t[a+"q_proj.weight"], t[a+"q_proj.bias"] = []int{c.NumHeads * hd, d}, []int{c.NumHeads * hd}
		t[a+"k_proj.weight"], t[a+"k_proj.bias"] = []int{c.NumKVHeads * hd, d}, []int{c.NumKVHeads * hd}
		t[a+"v_proj.weight"], t[a+"v_proj.bias"] = []int{c.NumKVHeads * hd, d}, []int{c.NumKVHeads * hd}
		t[a+"o_proj.weight"], t[a+"o_proj.bias"] = []int{d, c.NumHeads * hd}, []int{d}
		t[a+"sinks"] = []int{c.NumHeads}
		t[p+"mlp.router.weight"], t[p+"mlp.router.bias"] = []int{c.NumExperts, d}, []int{c.NumExperts}
	}
	return t
}

// Build writes a program without a KV cache (a part, as llm.Routed cuts it).
func (c *Config) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Cache > 0 {
		return nil, out, fmt.Errorf("gptoss: a program with a KV cache has more inputs and outputs; use BuildCached")
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
