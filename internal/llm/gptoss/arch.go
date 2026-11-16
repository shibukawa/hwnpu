package gptoss

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch registers gpt-oss with the whitelist in internal/llm. Its name is the chat format's
// (internal/chat's GptOss, as AMD's export names it).
var Arch = &llm.Arch{
	Name:          "gptoss",
	Title:         "gpt-oss",
	ModelTypes:    []string{"gpt_oss"},
	Architectures: []string{"GptOssForCausalLM"},
	// 4 bit tables for the experts and the output layer: every expert is read every token
	// (Config's doc), 9.6 GB of them for gpt-oss-20b in lut4, which a 16 GB machine holds; the
	// attention's projections in attn-format.
	Format: "lut4",
	Params: []llm.Param{
		{Name: "attn-format", Default: "lut8", Usage: "the weight format of the attention's q, k, v and o projections"},
		{Name: "expert-tables", Default: "exact", Usage: "how the experts' tables are fitted: exact-rows (a power of two per row taken out first, then exact), exact (the best table for each group of rows over the MXFP4 values) or palette (the format's own fitting)"},
	},
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

// Tensors lists the tensors the conversion reads: all the checkpoint has. The experts are MXFP4
// blocks and scales (uint8) in a quantized checkpoint, dense parameters (in by out) otherwise.
func (c *Config) Tensors() map[string][]int {
	d, f, hd, nE := c.HiddenSize, c.IntermediateSize, c.HeadDim, c.Experts
	t := map[string][]int{
		c.Embedding():       {c.VocabSize, d},
		"model.norm.weight": {d},
		"lm_head.weight":    {c.VocabSize, d},
	}
	for i := 0; i < c.NumLayers; i++ {
		l := layerPrefix(i)
		t[l+"input_layernorm.weight"] = []int{d}
		t[l+"post_attention_layernorm.weight"] = []int{d}
		for _, p := range []struct {
			name string
			out  int
		}{{"q", c.NumHeads * hd}, {"k", c.NumKVHeads * hd}, {"v", c.NumKVHeads * hd}} {
			t[l+"self_attn."+p.name+"_proj.weight"] = []int{p.out, d}
			t[l+"self_attn."+p.name+"_proj.bias"] = []int{p.out}
		}
		t[l+"self_attn.o_proj.weight"] = []int{d, c.NumHeads * hd}
		t[l+"self_attn.o_proj.bias"] = []int{d}
		t[l+"self_attn.sinks"] = []int{c.NumHeads}
		t[l+"mlp.router.weight"] = []int{nE, d}
		t[l+"mlp.router.bias"] = []int{nE}
		t[l+"mlp.experts.gate_up_proj_bias"] = []int{nE, 2 * f}
		t[l+"mlp.experts.down_proj_bias"] = []int{nE, d}
		if c.mxfp4() {
			t[l+"mlp.experts.gate_up_proj_blocks"] = []int{nE, 2 * f, d / 32, 16}
			t[l+"mlp.experts.gate_up_proj_scales"] = []int{nE, 2 * f, d / 32}
			t[l+"mlp.experts.down_proj_blocks"] = []int{nE, d, f / 32, 16}
			t[l+"mlp.experts.down_proj_scales"] = []int{nE, d, f / 32}
		} else {
			t[l+"mlp.experts.gate_up_proj"] = []int{nE, d, 2 * f}
			t[l+"mlp.experts.down_proj"] = []int{nE, f, d}
		}
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

// StepInputs gives the masks and the rotary tables of a step of valid tokens from pos
// (llm.Cached): one mask for the sliding layers and one for the full ones, the tables shared.
// Query t (position pos+t) sees cached position j < pos and step token u <= t, and in a sliding
// layer only those less than the window behind it. A sliding layer's cache is a ring when the
// window is shorter than the cache: slot j holds the latest position before pos congruent to j.
func (c *Config) StepInputs(pos, valid, q, cache int) map[string][]float32 {
	return c.stepInputs(pos, q, cache, false)
}

// PagedStepInputs is StepInputs for a program split at its attention (llm.Paged): without the
// full layers' mask, which the host's attention makes.
func (c *Config) PagedStepInputs(pos, valid, q, cache int) map[string][]float32 {
	return c.stepInputs(pos, q, cache, true)
}

func (c *Config) stepInputs(pos, q, cache int, paged bool) map[string][]float32 {
	out := map[string][]float32{}
	out["rope_cos"], out["rope_sin"] = blocks.RopeValues(c.RopeInvFreq(), pos, q, c.HeadDim)
	for _, full := range []bool{false, true} {
		if !c.has(full) || (paged && full) {
			continue
		}
		window := 0
		if !full {
			window = c.Window
		}
		keys := c.cacheLen(full, cache)
		ring := keys < cache
		mask := make([]float32, q*(keys+q))
		for t := 0; t < q; t++ {
			p := pos + t
			row := mask[t*(keys+q):]
			for j := 0; j < keys+q; j++ {
				var key int
				var visible bool
				switch {
				case j >= keys: // the step's own tokens
					key = pos + j - keys
					visible = j-keys <= t
				case ring: // the latest position written to slot j, if any
					key = pos - 1 - ((pos-1-j)%keys+keys)%keys
					visible = key >= 0
				default:
					key = j
					visible = j < pos
				}
				if window > 0 && p-key >= window {
					visible = false
				}
				if !visible {
					row[j] = -1e4
				}
			}
		}
		out["mask_"+layerType(full)] = mask
	}
	return out
}

// has reports whether the model has layers of a kind.
func (c *Config) has(full bool) bool {
	for i := 0; i < c.NumLayers; i++ {
		if c.full(i) == full {
			return true
		}
	}
	return false
}

// PagedLayers are the full attention layers: split at them (llm.Options.Paged), their attention
// runs on the host's cache in chunks, and the sliding ones keep their rings in the program.
func (c *Config) PagedLayers() []int {
	var out []int
	for i := 0; i < c.NumLayers; i++ {
		if c.full(i) {
			out = append(out, i)
		}
	}
	return out
}

// RingWindow is the window of the sliding layers, the length of their rings (llm.Windowed).
func (c *Config) RingWindow() int {
	if !c.has(false) {
		return 0
	}
	return c.Window
}

// PagedShape is the full attention layers' heads, kv heads and head size (llm.Paged), with their
// scores scaled by 1/sqrt(head_dim) times YaRN's concentration squared.
func (c *Config) PagedShape() paged.Shape {
	return paged.Shape{Heads: c.NumHeads, KVHeads: c.NumKVHeads, HeadDim: c.HeadDim, Scale: c.scoreScale()}
}

// PagedKV is layer itself: every layer has its own keys and values.
func (c *Config) PagedKV(layer int) int { return layer }

// MaxContext is max_position_embeddings: 131072 for gpt-oss-20b.
func (c *Config) MaxContext() int { return c.MaxPositions }

// HostSinks gives the sinks of the layers whose attention the host runs (llm.Sinks): each full
// layer's self_attn.sinks, a logit per head.
func (c *Config) HostSinks(w llm.Weights) (map[int][]float32, error) {
	out := map[int][]float32{}
	for _, i := range c.PagedLayers() {
		name := layerPrefix(i) + "self_attn.sinks"
		v, shape, err := w.Get(name)
		if err != nil {
			return nil, err
		}
		if len(shape) != 1 || shape[0] != c.NumHeads {
			return nil, fmt.Errorf("gptoss: %s: shape %v, expected [%d]", name, shape, c.NumHeads)
		}
		out[i] = v
	}
	return out, nil
}

var (
	_ llm.Paged    = (*Config)(nil)
	_ llm.Windowed = (*Config)(nil)
	_ llm.Sinks    = (*Config)(nil)
)
