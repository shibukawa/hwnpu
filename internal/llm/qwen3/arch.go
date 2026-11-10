package qwen3

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch registers Qwen3 with the whitelist in internal/llm.
var Arch = &llm.Arch{
	Name:          "qwen3",
	Title:         "Qwen3",
	ModelTypes:    []string{"qwen3"},
	Architectures: []string{"Qwen3ForCausalLM"},
	// 8 bit tables: Qwen3-0.6B's logits on the CPU are fp16's with KL 0.014 nats in lut8 and
	// 0.24 in lut6 (31 and 23 of 32 next tokens the same as fp32's), as for the other small
	// models (Qwen3.5-0.8B, PLaMo 3 Translate 0.8B).
	Format:     "lut8",
	LastRow:    true,
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseConfig(raw) },
}

func init() { llm.Register(Arch) }

// Hidden is the hidden size.
func (c *Config) Hidden() int { return c.HiddenSize }

// Layers is the number of decoder layers.
func (c *Config) Layers() int { return c.NumLayers }

// Output is the logits.
func (c *Config) Output() (int, bool) { return c.VocabSize, true }

// Embedding names the token embedding table.
func (c *Config) Embedding() string { return "model.embed_tokens.weight" }

func layerPrefix(i int) string { return fmt.Sprintf("model.layers.%d.", i) }

// Tensors lists the tensors the conversion reads. A tied checkpoint (Qwen3-0.6B) also has
// lm_head.weight, which transformers replaces with the embeddings when it loads it: it is not read.
func (c *Config) Tensors() map[string][]int {
	d, f, hd := c.HiddenSize, c.IntermediateSize, c.HeadDim
	t := map[string][]int{
		c.Embedding():       {c.VocabSize, d},
		"model.norm.weight": {d},
	}
	if !c.TieWordEmbed {
		t[c.head()] = []int{c.VocabSize, d}
	}
	for i := 0; i < c.NumLayers; i++ {
		l := layerPrefix(i)
		t[l+"input_layernorm.weight"] = []int{d}
		t[l+"post_attention_layernorm.weight"] = []int{d}
		t[l+"self_attn.q_proj.weight"] = []int{c.NumHeads * hd, d}
		t[l+"self_attn.k_proj.weight"] = []int{c.NumKVHeads * hd, d}
		t[l+"self_attn.v_proj.weight"] = []int{c.NumKVHeads * hd, d}
		t[l+"self_attn.o_proj.weight"] = []int{d, c.NumHeads * hd}
		t[l+"self_attn.q_norm.weight"] = []int{hd}
		t[l+"self_attn.k_norm.weight"] = []int{hd}
		t[l+"mlp.gate_proj.weight"] = []int{f, d}
		t[l+"mlp.up_proj.weight"] = []int{f, d}
		t[l+"mlp.down_proj.weight"] = []int{d, f}
	}
	return t
}

// Build writes the program. It returns x and y only; a part of a split model has more inputs and
// outputs, which BuildCached returns.
func (c *Config) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	ins, outs, err := Build(b, c, w, opts)
	if err != nil {
		return nil, out, err
	}
	return ins, outs[0], nil
}

// BuildCached writes the program with every input and output (llm.Cached), with or without a
// KV cache.
func (c *Config) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	return Build(b, c, w, opts)
}

// StepInputs gives the mask and the rotary tables of a step of valid tokens from pos
// (llm.Cached): query t (position pos+t) sees cached position j < pos and step token u <= t.
func (c *Config) StepInputs(pos, valid, q, cache int) map[string][]float32 {
	out := c.PagedStepInputs(pos, valid, q, cache)
	mask := make([]float32, q*(cache+q))
	for t := 0; t < q; t++ {
		row := mask[t*(cache+q):]
		for j := 0; j < cache+q; j++ {
			if (j < cache && j >= pos) || (j >= cache && j-cache > t) {
				row[j] = -1e4
			}
		}
	}
	out["mask"] = mask
	return out
}

// PagedStepInputs is StepInputs for a program split at its attention (llm.Paged): the rotary
// tables without the mask, which the host's attention makes.
func (c *Config) PagedStepInputs(pos, valid, q, cache int) map[string][]float32 {
	cos, sin := blocks.RopeValues(c.RopeInvFreq(), pos, q, c.HeadDim)
	return map[string][]float32{"rope_cos": cos, "rope_sin": sin}
}

// PagedLayers are all the layers: split at its attention (llm.Options.Paged), every layer's
// attention runs on the host's cache in chunks. hwnpu-chat does it only when told (llm.Unpaged).
func (c *Config) PagedLayers() []int {
	out := make([]int, c.NumLayers)
	for i := range out {
		out[i] = i
	}
	return out
}

// PagedShape is the attention's heads, kv heads and head size (llm.Paged), the scores scaled by
// 1/sqrt(head_dim).
func (c *Config) PagedShape() paged.Shape {
	return paged.Shape{Heads: c.NumHeads, KVHeads: c.NumKVHeads, HeadDim: c.HeadDim, Scale: c.scoreScale()}
}

// PagedKV is layer itself: every layer has its own keys and values.
func (c *Config) PagedKV(layer int) int { return layer }

// MaxContext is max_position_embeddings: 40960 for Qwen3-0.6B.
func (c *Config) MaxContext() int { return c.MaxPositions }

// UnpagedContext is the cache of a whole program (llm.Unpaged): with 28 layers of keys and
// values as inputs, Qwen3-0.6B's programs of 8192 and 16320 compile and load on an M1, then fail
// to predict; 4096 runs.
func (c *Config) UnpagedContext() int { return min(4096, c.MaxPositions) }

var (
	_ llm.Paged   = (*Config)(nil)
	_ llm.Unpaged = (*Config)(nil)
)
