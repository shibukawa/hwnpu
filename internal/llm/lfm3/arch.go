package lfm3

import (
	"fmt"
	"strconv"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch registers LFM3 with the whitelist in internal/llm.
var Arch = &llm.Arch{
	Name:          "lfm3",
	Title:         "LFM3",
	ModelTypes:    []string{"lfm3", "lfm3_vl"},
	Architectures: []string{"Lfm3ForCausalLM", "Lfm3VlForConditionalGeneration"},
	Params: []llm.Param{
		{Name: "post-scale", Default: "64", Usage: "power of two multiplied into the weights of out_proj, o_proj and w2 and absorbed by the sandwich norms (Neural Engine accuracy); 1 disables"},
		{Name: "conv", Default: "taps", Usage: "short convolution as shifted multiply adds (taps) or one conv op (conv)"},
	},
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseConfig(raw) },
	GGUFName:   ggufName,
	GGUFShape:  ggufShape,
}

func init() { llm.Register(Arch) }

// Hidden is the hidden size.
func (c *Config) Hidden() int { return c.HiddenSize }

// Layers is the number of layers.
func (c *Config) Layers() int { return c.NumLayers }

// Output is the logits, or for a d1 model the final hidden state (Config.D1).
func (c *Config) Output() (int, bool) {
	if c.D1 {
		return c.HiddenSize, false
	}
	return c.VocabSize, true
}

// Embedding names the token embedding table.
func (c *Config) Embedding() string { return c.prefix() + "embed_tokens.weight" }

// Head names the output layer's weight: lm_head, or the embedding table when they are tied.
func (c *Config) Head() string {
	if c.TieWordEmbed {
		return c.Embedding()
	}
	return "lm_head.weight"
}

// Tensors lists the tensors the conversion reads. A d1 model's program ends with the final norm,
// and its lm_head (when not tied) is left to the host, which reads the rows of the answer
// labels: it is unused by the conversion.
func (c *Config) Tensors() map[string][]int {
	d, hd := c.HiddenSize, c.HeadDim
	t := map[string][]int{
		c.Embedding():                        {c.VocabSize, d},
		c.prefix() + "embedding_norm.weight": {d},
	}
	if !c.TieWordEmbed && !c.D1 {
		t["lm_head.weight"] = []int{c.VocabSize, d}
	}
	for i := 0; i < c.NumLayers; i++ {
		p := fmt.Sprintf("%slayers.%d.", c.prefix(), i)
		norms := []string{"operator_norm", "ffn_norm"}
		if c.UseSandwichNorm {
			norms = append(norms, "post_operator_norm", "post_ffn_norm")
		}
		for _, n := range norms {
			t[p+n+".weight"] = []int{d}
		}
		t[p+"feed_forward.w1.weight"] = []int{c.IntermediateSize, d}
		t[p+"feed_forward.w3.weight"] = []int{c.IntermediateSize, d}
		t[p+"feed_forward.w2.weight"] = []int{d, c.IntermediateSize}
		if c.IsAttention(i) {
			a := p + "self_attn."
			t[a+"q_proj.weight"] = []int{c.NumHeads * hd, d}
			t[a+"k_proj.weight"] = []int{c.NumKVHeads * hd, d}
			t[a+"v_proj.weight"] = []int{c.NumKVHeads * hd, d}
			t[a+"o_proj.weight"] = []int{d, c.NumHeads * hd}
			t[a+"gate_proj.weight"] = []int{c.NumHeads, d}
			t[a+"q_norm.weight"] = []int{hd}
			t[a+"k_norm.weight"] = []int{hd}
		} else {
			t[p+"conv.in_proj.weight"] = []int{3 * d, d}
			t[p+"conv.out_proj.weight"] = []int{d, d}
			t[p+"conv.conv.weight"] = []int{d, 1, c.ConvKernel}
			if c.ConvBias {
				t[p+"conv.conv.bias"] = []int{d}
			}
		}
	}
	return t
}

// Build writes the program with the LFM3 settings taken from opts.Params.
func (c *Config) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	ps := Arch.Param(opts, "post-scale")
	postScale, err := strconv.ParseFloat(ps, 64)
	if err != nil {
		return nil, out, fmt.Errorf("lfm3: post-scale %q: %w", ps, err)
	}
	in, out, err := Build(b, c, w, Options{Options: opts, PostScale: float32(postScale), ConvMode: Arch.Param(opts, "conv")})
	return []mlpackage.Feature{in}, out, err
}

// BuildCached writes the program with a KV cache (llm.Cached), with the settings of Build.
func (c *Config) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	ps := Arch.Param(opts, "post-scale")
	postScale, err := strconv.ParseFloat(ps, 64)
	if err != nil {
		return nil, nil, fmt.Errorf("lfm3: post-scale %q: %w", ps, err)
	}
	return BuildCached(b, c, w, Options{Options: opts, PostScale: float32(postScale), ConvMode: Arch.Param(opts, "conv")})
}
