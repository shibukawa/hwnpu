package qwen35

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Arch registers Qwen3.5 with the whitelist in internal/llm.
var Arch = &llm.Arch{
	Name:          "qwen35",
	Title:         "Qwen3.5",
	ModelTypes:    []string{"qwen3_5", "qwen3_5_text"},
	Architectures: []string{"Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"},
	Format:        "lut8",
	LastRow:       true,
	// 1: the gated delta rule over chunks of rows (deltaRuleChunks)
	// 2: a chunk of one row without the matmul of an inner dimension of 1, which the Neural
	// Engine gets wrong (deltaRuleChunks), 2026-11-20
	Revision:   2,
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseConfig(raw) },
	Params: []llm.Param{
		{Name: "delta", Default: "chunk", Usage: "the Gated DeltaNet's recurrence: chunk (over chunks of rows, the state read and written once a chunk) or rows (one row after the other, the state read and written five times a row)"},
		{Name: "hidden-out", Default: "0", Usage: "1: a program with a KV cache also outputs h, the final hidden state of every row, which the MTP layer (-arch qwen35-mtp) reads for speculative decoding"},
		{Name: llm.RotateParam, Default: "0", Usage: "1: the residual stream rotated in the weights (llm.Rotate, a Walsh-Hadamard matrix with random signs): the same program, a lookup table's error about halved (lut6 on the Neural Engine, docs/design/apple-neural-engine.md); the host's embedding table is then the rotated one (tables-rot.safetensors). Not for a mixture of experts"},
		{Name: "state-rows", Default: "0", Usage: "1: a program with a KV cache also outputs st_rec_N_rows, each row's k, v, beta and decay of the Gated DeltaNet layer N, so that the host keeps a beginning of a step's tokens without running them again (speculative decoding)"},
	},
}

// MTPArch registers the multi-token prediction layer of a Qwen3.5 checkpoint (mtp.) as a model of
// its own, which guesses the token after next for speculative decoding; it is chosen with -arch.
var MTPArch = &llm.Arch{
	Name:       "qwen35-mtp",
	Title:      "Qwen3.5 multi-token prediction",
	Format:     "lut8",
	LastRow:    true,
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseMTP(raw) },
}

func init() {
	llm.Register(Arch)
	llm.Register(MTPArch)
}

// Hidden is the hidden size.
func (c *Config) Hidden() int { return c.HiddenSize }

// Layers is the number of layers.
func (c *Config) Layers() int { return c.NumLayers }

// Output is the logits.
func (c *Config) Output() (int, bool) { return c.VocabSize, true }

// Embedding names the token embedding table.
func (c *Config) Embedding() string { return c.prefix() + "embed_tokens.weight" }

// Head names the output layer's weight: lm_head, or the embedding table when they are tied.
func (c *Config) Head() string {
	if c.TieWordEmbed {
		return c.Embedding()
	}
	return "lm_head.weight"
}

// Tensors lists the tensors the conversion reads. A checkpoint has two more groups the text model
// does not use: the vision encoder (model.visual.), and the multi-token prediction layer (mtp.),
// which predicts the token after next for speculative decoding.
func (c *Config) Tensors() map[string][]int {
	d, hd := c.HiddenSize, c.HeadDim
	t := map[string][]int{
		c.Embedding():              {c.VocabSize, d},
		c.prefix() + "norm.weight": {d},
	}
	if !c.TieWordEmbed {
		t[c.Head()] = []int{c.VocabSize, d}
	}
	for i := 0; i < c.NumLayers; i++ {
		p := fmt.Sprintf("%slayers.%d.", c.prefix(), i)
		t[p+"input_layernorm.weight"] = []int{d}
		t[p+"post_attention_layernorm.weight"] = []int{d}
		t[p+"mlp.gate_proj.weight"] = []int{c.IntermediateSize, d}
		t[p+"mlp.up_proj.weight"] = []int{c.IntermediateSize, d}
		t[p+"mlp.down_proj.weight"] = []int{d, c.IntermediateSize}
		if c.IsLinear(i) {
			a := p + "linear_attn."
			hv := c.LinearValueHeads
			t[a+"in_proj_qkv.weight"] = []int{c.convDim(), d}
			t[a+"in_proj_z.weight"] = []int{c.valueDim(), d}
			t[a+"in_proj_b.weight"] = []int{hv, d}
			t[a+"in_proj_a.weight"] = []int{hv, d}
			t[a+"conv1d.weight"] = []int{c.convDim(), 1, c.ConvKernel}
			t[a+"dt_bias"] = []int{hv}
			t[a+"A_log"] = []int{hv}
			t[a+"norm.weight"] = []int{c.LinearValueDim}
			t[a+"out_proj.weight"] = []int{d, c.valueDim()}
		} else {
			a := p + "self_attn."
			t[a+"q_proj.weight"] = []int{2 * c.NumHeads * hd, d}
			t[a+"k_proj.weight"] = []int{c.NumKVHeads * hd, d}
			t[a+"v_proj.weight"] = []int{c.NumKVHeads * hd, d}
			t[a+"o_proj.weight"] = []int{d, c.NumHeads * hd}
			t[a+"q_norm.weight"] = []int{hd}
			t[a+"k_norm.weight"] = []int{hd}
		}
	}
	return t
}

// Build writes the program (Build in build.go with the options as they are).
func (c *Config) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	in, out, err := Build(b, c, w, opts)
	return []mlpackage.Feature{in}, out, err
}

// BuildCached writes the program with its state held by the host (llm.Cached).
func (c *Config) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	return BuildCached(b, c, w, opts)
}

// MaxContext is max_position_embeddings: 262144 for Qwen3.5.
func (c *Config) MaxContext() int { return c.MaxPositions }

var _ llm.Paged = (*Config)(nil)

// Rotation is how an orthogonal matrix passes through the text model's residual stream
// (llm.Rotate): every reader is a linear after a zero centered norm, every writer a linear added
// as it is. The MTP layer's fc reads two normed halves and is not rotated.
func (c *Config) Rotation() (llm.Rotation, error) {
	if c.NumExperts > 0 {
		// the experts run on the CPU from the checkpoint's MXFP4 (llm.HostRouted): rotated
		// they would be quantized again
		return llm.Rotation{}, fmt.Errorf("qwen35: a mixture of experts is not rotated")
	}
	r := llm.Rotation{Dim: c.HiddenSize, Embed: c.Embedding(), Head: c.Head(), HeadNorm: c.prefix() + "norm.weight",
		Readers: map[string]string{}, Writers: map[string]bool{}, Norms: map[string]bool{c.prefix() + "norm.weight": true}, ZeroCentered: true}
	for i := 0; i < c.NumLayers; i++ {
		p := fmt.Sprintf("%slayers.%d.", c.prefix(), i)
		in, post := p+"input_layernorm.weight", p+"post_attention_layernorm.weight"
		r.Norms[in], r.Norms[post] = true, true
		r.Readers[p+"mlp.gate_proj.weight"], r.Readers[p+"mlp.up_proj.weight"] = post, post
		r.Writers[p+"mlp.down_proj.weight"] = true
		if c.IsLinear(i) {
			a := p + "linear_attn."
			for _, n := range []string{"in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"} {
				r.Readers[a+n+".weight"] = in
			}
			r.Writers[a+"out_proj.weight"] = true
		} else {
			a := p + "self_attn."
			for _, n := range []string{"q_proj", "k_proj", "v_proj"} {
				r.Readers[a+n+".weight"] = in
			}
			r.Writers[a+"o_proj.weight"] = true
		}
	}
	return r, nil
}
