package plamo3

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch registers PLaMo 3 with the whitelist in internal/llm.
var Arch = &llm.Arch{
	Name:          "plamo3",
	Title:         "PLaMo 3",
	ModelTypes:    []string{"plamo3"},
	Architectures: []string{"Plamo3ForCausalLM"},
	// 8 bit tables: on an M3's Neural Engine the next token is fp16's at 98% of a conversation's
	// positions with lut8, 89% with lut6, for 8% more time per call (57.8 ms against 53.7); lut4
	// is unusable (KL 1.91 nats on the CPU). The two programs of lut8 (4.9 GiB) are over the Neural
	// Engine's compile cache, so hwnpu-chat keeps the one that generates and reads the prompts
	// with it, which short texts to translate hardly feel (docs/design.md 16.69).
	Format:     "lut8",
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

// layerPrefix is where layer i's tensors are: Plamo3Model.layers is a Plamo3Decoder whose
// layers are a ModuleList called layers.
func layerPrefix(i int) string { return fmt.Sprintf("model.layers.layers.%d.", i) }

// Tensors lists the tensors the conversion reads: all the checkpoint has.
func (c *Config) Tensors() map[string][]int {
	d, f, hd := c.HiddenSize, c.IntermediateSize, c.HeadDim
	t := map[string][]int{
		c.Embedding():       {c.VocabSize, d},
		"model.norm.weight": {d},
	}
	if !c.TieWordEmbed {
		t[c.head()] = []int{c.headRows(), d}
	}
	for i := 0; i < c.NumLayers; i++ {
		l := layerPrefix(i)
		for _, n := range []string{"pre_mixer_norm", "post_mixer_norm", "pre_mlp_norm", "post_mlp_norm"} {
			t[l+n+".weight"] = []int{d}
		}
		t[l+"mixer.qkv_proj.weight"] = []int{(c.NumHeads + 2*c.NumKVHeads) * hd, d}
		t[l+"mixer.o_proj.weight"] = []int{d, c.NumHeads * hd}
		t[l+"mixer.q_norm.weight"] = []int{hd}
		t[l+"mixer.k_norm.weight"] = []int{hd}
		t[l+"mlp.gate_up_proj.weight"] = []int{2 * f, d}
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

// layerType names the inputs of a kind of layer: "full" or "sliding".
func layerType(full bool) string {
	if full {
		return "full"
	}
	return "sliding"
}

// StepInputs gives the masks and rotary tables of a step of valid tokens from pos (llm.Cached),
// one of each for the sliding and the full layers. Query t (position pos+t) sees cached position
// j < pos and step token u <= t, and in a sliding layer only those less than the window behind
// it. A sliding layer's cache is a ring when the window is shorter than the cache: slot j holds
// the latest position before pos congruent to j.
func (c *Config) StepInputs(pos, valid, q, cache int) map[string][]float32 {
	return c.stepInputs(pos, q, cache, false)
}

func (c *Config) stepInputs(pos, q, cache int, paged bool) map[string][]float32 {
	out := map[string][]float32{}
	for _, full := range []bool{false, true} {
		lt := layerType(full)
		if !c.has(full) {
			continue
		}
		out["rope_cos_"+lt], out["rope_sin_"+lt] = blocks.RopeValues(c.RopeInvFreq(full), pos, q, c.HeadDim)
		if paged && full {
			continue // the host's attention masks
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
		out["mask_"+lt] = mask
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

// PagedStepInputs is StepInputs for a program split at its attention (llm.Paged): the full
// attention layers' rotary tables without their mask, which the host's attention makes.
func (c *Config) PagedStepInputs(pos, valid, q, cache int) map[string][]float32 {
	return c.stepInputs(pos, q, cache, true)
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

// PagedShape is the full attention layers' heads, kv heads and head size (llm.Paged), with their
// scores scaled by 1/sqrt(head_dim) times YaRN's concentration squared.
func (c *Config) PagedShape() paged.Shape {
	return paged.Shape{Heads: c.NumHeads, KVHeads: c.NumKVHeads, HeadDim: c.HeadDim, Scale: c.scoreScale(true)}
}

// PagedKV is layer itself: every layer has its own keys and values.
func (c *Config) PagedKV(layer int) int { return layer }

// MaxContext is max_position_embeddings: 262144 for PLaMo 3 Translate.
func (c *Config) MaxContext() int { return c.MaxPositions }

var _ llm.Paged = (*Config)(nil)
