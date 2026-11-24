package ouro

import (
	"fmt"

	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/llm/blocks"
	"github.com/shibukawa/bonsaiai/internal/llm/paged"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
)

// Arch registers Ouro with the whitelist in internal/llm. 8 bit tables by default, as for the
// other models of 2B and less (Qwen3, Qwen3.5-0.8B): the format is not measured on Ouro's logits.
var Arch = &llm.Arch{
	Name:          "ouro",
	Title:         "Ouro",
	ModelTypes:    []string{"ouro"},
	Architectures: []string{"OuroForCausalLM"},
	Format:        "lut8",
	LastRow:       true,
	LoadConfig:    func(raw []byte) (llm.Model, error) { return ParseConfig(raw) },
}

func init() { llm.Register(Arch) }

// Hidden is the hidden size.
func (c *Config) Hidden() int { return c.HiddenSize }

// Layers is the number of layers the model runs on a token: the decoder layers once for each
// pass.
func (c *Config) Layers() int { return c.NumLayers * c.Passes }

// WeightLayer is the decoder layer whose weights layer i runs (llm.Looped): i within its pass.
func (c *Config) WeightLayer(i int) int { return i % c.NumLayers }

// pass is the pass layer i is in, from 0.
func (c *Config) pass(i int) int { return i / c.NumLayers }

// endsPass reports whether layer i is the last of its pass.
func (c *Config) endsPass(i int) bool { return c.WeightLayer(i) == c.NumLayers-1 }

// Output is the logits.
func (c *Config) Output() (int, bool) { return c.VocabSize, true }

// Embedding names the token embedding table.
func (c *Config) Embedding() string { return "model.embed_tokens.weight" }

// layerPrefix names the weights of layer i (of Layers): those of its decoder layer.
func (c *Config) layerPrefix(i int) string { return fmt.Sprintf("model.layers.%d.", c.WeightLayer(i)) }

// Tensors lists the tensors the conversion reads. model.early_exit_gate (weight and bias) is not
// read: it scores each pass as the last, which matters only for an exit before the last pass
// (early_exit_threshold under 1, refused by ParseConfig). A tied checkpoint's lm_head.weight is
// the embeddings, as for Qwen3.
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
		l := fmt.Sprintf("model.layers.%d.", i)
		for _, n := range []string{"input_layernorm", "input_layernorm_2", "post_attention_layernorm", "post_attention_layernorm_2"} {
			t[l+n+".weight"] = []int{d}
		}
		t[l+"self_attn.q_proj.weight"] = []int{c.NumHeads * hd, d}
		t[l+"self_attn.k_proj.weight"] = []int{c.NumKVHeads * hd, d}
		t[l+"self_attn.v_proj.weight"] = []int{c.NumKVHeads * hd, d}
		t[l+"self_attn.o_proj.weight"] = []int{d, c.NumHeads * hd}
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
// (llm.Cached): query t (position pos+t) sees cached position j < pos and step token u <= t. Every
// pass sees the same positions.
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

// PagedLayers are all the layers of every pass: split at its attention (llm.Options.Paged), every
// layer's attention runs on the host's cache in chunks.
func (c *Config) PagedLayers() []int {
	out := make([]int, c.Layers())
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

// PagedKV is layer itself: each pass of each layer has its own keys and values.
func (c *Config) PagedKV(layer int) int { return layer }

// MaxContext is max_position_embeddings: 65536 for Ouro-1.4B and 2.6B.
func (c *Config) MaxContext() int { return c.MaxPositions }

// UnpagedContext is the cache of a whole program (llm.Unpaged): Qwen3-0.6B's 28 layers of keys
// and values ran at 4096 and not at 8192 on an M1; Ouro has 96 or 192, so a quarter of that. Not
// measured on Ouro.
func (c *Config) UnpagedContext() int { return min(1024, c.MaxPositions) }

var (
	_ llm.Paged   = (*Config)(nil)
	_ llm.Unpaged = (*Config)(nil)
	_ llm.Looped  = (*Config)(nil)
)
