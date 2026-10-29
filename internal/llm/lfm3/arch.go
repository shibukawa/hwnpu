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
	ModelTypes:    []string{"lfm3"},
	Architectures: []string{"Lfm3ForCausalLM"},
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

// Build writes the program with the LFM3 settings taken from opts.Params.
func (c *Config) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (in, out mlpackage.Feature, err error) {
	ps := Arch.Param(opts, "post-scale")
	postScale, err := strconv.ParseFloat(ps, 64)
	if err != nil {
		return in, out, fmt.Errorf("lfm3: post-scale %q: %w", ps, err)
	}
	return Build(b, c, w, Options{Options: opts, PostScale: float32(postScale), ConvMode: Arch.Param(opts, "conv")})
}
