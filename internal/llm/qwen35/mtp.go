package qwen35

import (
	"fmt"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// MTP is the multi-token prediction layer of a Qwen3.5 checkpoint (its tensors under mtp.) as a
// causal model of its own (Qwen3_5MultiTokenPredictor in vLLM's qwen3_5_mtp.py, which
// transformers does not implement). At position t it reads the model's final hidden state h_t
// and the embedding of token t+1, and its logits guess token t+2: the drafts of speculative
// decoding, which the model then checks in one call. Its one decoder layer is a full attention
// layer over its own KV cache, which holds an entry for every position it has read.
//
// As an llm.Model it has two layers: layer 0 is the front (the two inputs' norms, their
// concatenation and fc) and layer 1 the decoder layer mtp.layers.0, so that the attention of
// layer 1 can run on the host (llm.Paged, which cannot split at layer 0). The program takes x,
// the embedding of the next token, and hid, the model's hidden state (llm.TokenInputs: the
// host fills it per token from the model's output h, not from a table). With a KV cache it also
// outputs h, its own final hidden state, which it reads in place of the model's to guess further
// ahead. The output layer is the model's (shared, as in vLLM).
type MTP struct{ *Config }

// ParseMTP parses config.json for the MTP model and rejects what it does not implement.
func ParseMTP(raw []byte) (*MTP, error) {
	c, err := ParseConfig(raw)
	if err != nil {
		return nil, err
	}
	switch {
	case c.MTPLayers == 0:
		return nil, fmt.Errorf("qwen35: the checkpoint has no multi-token prediction layer (mtp_num_hidden_layers)")
	case c.MTPLayers > 1:
		return nil, fmt.Errorf("qwen35: %d multi-token prediction layers; only 1 is supported", c.MTPLayers)
	case c.MTPOwnEmbeddings:
		return nil, fmt.Errorf("qwen35: multi-token prediction with embeddings of its own (mtp_use_dedicated_embeddings) is not supported")
	}
	return &MTP{c}, nil
}

// Layers is 2: the front and the decoder layer.
func (c *MTP) Layers() int { return 2 }

// TokenInputs is hid, the model's final hidden state at the position, which the host gives per
// token from the model's output h (no table: Table is empty).
func (c *MTP) TokenInputs() []llm.TokenInput {
	return []llm.TokenInput{{Input: "hid", Width: c.HiddenSize}}
}

// mtpLayer is the prefix of the decoder layer's tensors.
const mtpLayer = "mtp.layers.0."

// Tensors lists the tensors the MTP program reads: its own under mtp., and the model's output
// layer. The embedding table is the model's, read on the host.
func (c *MTP) Tensors() map[string][]int {
	d, hd := c.HiddenSize, c.HeadDim
	t := map[string][]int{
		c.Embedding():                      {c.VocabSize, d},
		c.Head():                           {c.VocabSize, d},
		"mtp.pre_fc_norm_embedding.weight": {d},
		"mtp.pre_fc_norm_hidden.weight":    {d},
		"mtp.fc.weight":                    {d, 2 * d},
		"mtp.norm.weight":                  {d},
	}
	p := mtpLayer
	t[p+"input_layernorm.weight"] = []int{d}
	t[p+"post_attention_layernorm.weight"] = []int{d}
	t[p+"mlp.gate_proj.weight"] = []int{c.IntermediateSize, d}
	t[p+"mlp.up_proj.weight"] = []int{c.IntermediateSize, d}
	t[p+"mlp.down_proj.weight"] = []int{d, c.IntermediateSize}
	a := p + "self_attn."
	t[a+"q_proj.weight"] = []int{2 * c.NumHeads * hd, d}
	t[a+"k_proj.weight"] = []int{c.NumKVHeads * hd, d}
	t[a+"v_proj.weight"] = []int{c.NumKVHeads * hd, d}
	t[a+"o_proj.weight"] = []int{d, c.NumHeads * hd}
	t[a+"q_norm.weight"] = []int{hd}
	t[a+"k_norm.weight"] = []int{hd}
	return t
}

// BeforeLayers reports the front's tensors (llm.Prologue): the norms of the inputs and fc are
// before the decoder layer, not after it with the final norm and the output layer.
func (c *MTP) BeforeLayers(name string) bool {
	return strings.HasPrefix(name, "mtp.pre_fc_norm_") || name == "mtp.fc.weight"
}

// Build writes the program without a cache.
func (c *MTP) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Cache > 0 {
		return nil, out, fmt.Errorf("qwen35: a program with a KV cache has more inputs and outputs; use BuildCached")
	}
	all, outs, err := c.BuildCached(b, w, opts)
	if err != nil {
		return nil, out, err
	}
	return all, outs[0], nil
}

// BuildCached writes the program with its state held by the host (llm.Cached): the KV cache of
// layer 1, kv_k_1 and kv_v_1, and the inputs of Config.StepInputs.
func (c *MTP) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	if len(opts.HeadRows) > 0 {
		// the drafts of a part of the vocabulary (llm.Options.HeadRows): the model checks each,
		// so a token outside it only goes undrafted
		w = llm.SelectRows(w, c.Head(), opts.HeadRows)
	}
	return buildCached(b, c.Config, w, opts, true, false)
}

// PagedLayers is the decoder layer (llm.Paged).
func (c *MTP) PagedLayers() []int { return []int{1} }

var (
	_ llm.Paged       = (*MTP)(nil)
	_ llm.TokenInputs = (*MTP)(nil)
	_ llm.Prologue    = (*MTP)(nil)
)
