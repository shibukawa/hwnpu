package gemma4

import (
	"fmt"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch registers Gemma 4 with the whitelist in internal/llm.
var Arch = &llm.Arch{
	Name:          "gemma4",
	Title:         "Gemma 4",
	ModelTypes:    []string{"gemma4", "gemma4_text"},
	Architectures: []string{"Gemma4ForConditionalGeneration", "Gemma4ForCausalLM"},
	// 4 bit tables cost E2B too much (KL 2.54 nats against the reference, against 0.25 with
	// 6 bits and 0.05 with 8); 6 bits keeps 2 parts on the M3 and runs at 31 ms a token, against
	// 24 with 4 bits and 41 with 8 (docs/design.md 16.22).
	Format:     "lut6",
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseConfig(raw) },
}

func init() { llm.Register(Arch) }

// Hidden is the hidden size.
func (c *Config) Hidden() int { return c.HiddenSize }

// Layers is the number of decoder layers.
func (c *Config) Layers() int { return c.NumLayers }

// Output is the logits.
func (c *Config) Output() (int, bool) { return c.VocabSize, true }

// Embedding names the token embedding table, also the output layer.
func (c *Config) Embedding() string { return c.Prefix + "embed_tokens.weight" }

// perLayerTable names the per-layer embeddings: for each token, a slice of PerLayerDim values
// for every layer.
func (c *Config) perLayerTable() string { return c.Prefix + "embed_tokens_per_layer.weight" }

// TokenInputs is the per-layer embeddings, which the host looks up like the token embeddings
// (llm.TokenInputs): the program's input ple, the rows as they are in the checkpoint.
func (c *Config) TokenInputs() []llm.TokenInput {
	if c.PerLayerDim == 0 {
		return nil
	}
	return []llm.TokenInput{{Input: "ple", Table: c.perLayerTable(), Width: c.NumLayers * c.PerLayerDim}}
}

// BeforeLayers reports the tensors of the per-layer embeddings' projection, which the program
// runs before layer 0 (llm.Prologue).
func (c *Config) BeforeLayers(name string) bool {
	return name == c.Prefix+"per_layer_model_projection.weight" || name == c.Prefix+"per_layer_projection_norm.weight"
}

// Tensors lists the tensors the conversion reads. The checkpoint has more that it leaves alone:
// the vision and audio towers and their embedders (audio_tower, vision_tower, embed_audio,
// embed_vision), and the k_proj, v_proj and k_norm of the layers that reuse an earlier layer's
// keys and values, which transformers does not load either (_keys_to_ignore_on_load_unexpected).
func (c *Config) Tensors() map[string][]int {
	d, p := c.HiddenSize, c.Prefix
	t := map[string][]int{
		c.Embedding():     {c.VocabSize, d},
		p + "norm.weight": {d},
	}
	if c.PerLayerDim > 0 {
		t[c.perLayerTable()] = []int{c.PerLayerVocab, c.NumLayers * c.PerLayerDim}
		t[p+"per_layer_model_projection.weight"] = []int{c.NumLayers * c.PerLayerDim, d}
		t[p+"per_layer_projection_norm.weight"] = []int{c.PerLayerDim}
	}
	for i := 0; i < c.NumLayers; i++ {
		l := fmt.Sprintf("%slayers.%d.", p, i)
		for _, n := range []string{"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"} {
			t[l+n+".weight"] = []int{d}
		}
		t[l+"layer_scalar"] = []int{1}
		f := c.mlpWidth(i)
		t[l+"mlp.gate_proj.weight"] = []int{f, d}
		t[l+"mlp.up_proj.weight"] = []int{f, d}
		t[l+"mlp.down_proj.weight"] = []int{d, f}
		hd := c.headDim(c.LayerTypes[i])
		a := l + "self_attn."
		t[a+"q_proj.weight"] = []int{c.NumHeads * hd, d}
		t[a+"q_norm.weight"] = []int{hd}
		t[a+"o_proj.weight"] = []int{d, c.NumHeads * hd}
		if !c.shared(i) {
			t[a+"k_proj.weight"] = []int{c.NumKVHeads * hd, d}
			t[a+"v_proj.weight"] = []int{c.NumKVHeads * hd, d}
			t[a+"k_norm.weight"] = []int{hd}
		}
		if c.PerLayerDim > 0 {
			t[l+"per_layer_input_gate.weight"] = []int{c.PerLayerDim, d}
			t[l+"per_layer_projection.weight"] = []int{d, c.PerLayerDim}
			t[l+"post_per_layer_input_norm.weight"] = []int{d}
		}
	}
	return t
}

// Build writes the program. It returns x and y only; a part of a split model has more inputs and
// outputs, which BuildCached returns.
func (c *Config) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	ins, outs, err := Build(b, c, w, Options{Options: opts})
	if err != nil {
		return nil, out, err
	}
	return ins, outs[0], nil
}

// BuildCached writes the program with every input and output (llm.Cached), with or without a
// KV cache.
func (c *Config) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	return Build(b, c, w, Options{Options: opts})
}

// StepInputs gives the masks and rotary tables of a step of valid tokens from pos (llm.Cached),
// one of each per layer type. Query t (position pos+t) sees cached position j < pos and step
// token u <= t, and in a sliding layer only those less than the window behind it. A sliding
// layer's cache is a ring when the window is shorter than the cache: slot j holds the latest
// position before pos congruent to j.
func (c *Config) StepInputs(pos, valid, q, cache int) map[string][]float32 {
	out := map[string][]float32{}
	for _, lt := range c.LayerTypes {
		if _, done := out["mask_"+lt]; done {
			continue
		}
		window := 0
		if lt == "sliding_attention" {
			window = c.SlidingWindow
		}
		keys := c.cacheLen(lt, cache)
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
		out["mask_"+lt] = mask
		out["rope_cos_"+lt], out["rope_sin_"+lt] = blocks.RopeValues(c.RopeInvFreq(lt), pos, q, c.headDim(lt))
	}
	return out
}

// sharedName is the name of the keys ("k") or values ("v") of the layers of a type that the
// shared layers reuse, as an output of the program that computes them and an input of a later
// part that reuses them (internal/llm/chain binds the two).
func sharedName(kv, layerType string) string {
	return "shared_" + kv + "_" + strings.TrimSuffix(layerType, "_attention")
}
