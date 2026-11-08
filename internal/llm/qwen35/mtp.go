package qwen35

import (
	_ "embed"
	"encoding/binary"
	"fmt"
	"slices"
	"strconv"
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

// draftVocab is the ids of the Qwen3.5 tokenizer's most frequent tokens, the most frequent first,
// as little-endian uint32: the chat format's control tokens, then by their count in Japanese and
// English text (wiki40b-ja, wikitext-2) and in this repository's code and documents
// (tools/draftvocab). The 32768 first cover 98.3% of the tokens counted, the 16384 first 94.3%.
//
//go:embed draft_vocab.bin
var draftVocab []byte

// DraftVocab is the ids of the k most frequent tokens of the Qwen3.5 tokenizer (draftVocab), in
// ascending order: the rows of the multi-token prediction layer's output layer with draft-vocab k.
func DraftVocab(k int) ([]int32, error) {
	if n := len(draftVocab) / 4; k <= 0 || k > n || k%16 != 0 {
		return nil, fmt.Errorf("qwen35: draft-vocab %d is not a multiple of 16 from 16 to %d", k, n)
	}
	ids := make([]int32, k)
	for i := range ids {
		ids[i] = int32(binary.LittleEndian.Uint32(draftVocab[4*i:]))
	}
	slices.Sort(ids)
	return ids, nil
}

// OutputIDs is the tokens of the logits of a program with draft-vocab (llm.OutputSubset), nil
// for the whole vocabulary.
func (c *MTP) OutputIDs(opts llm.Options) ([]int32, error) {
	k, err := strconv.Atoi(MTPArch.Param(opts, "draft-vocab"))
	if err != nil || k < 0 {
		return nil, fmt.Errorf("qwen35: draft-vocab %q is not a number of tokens", MTPArch.Param(opts, "draft-vocab"))
	}
	if k == 0 {
		return nil, nil
	}
	ids, err := DraftVocab(k)
	if err != nil {
		return nil, err
	}
	for _, id := range ids {
		if int(id) >= c.VocabSize {
			return nil, fmt.Errorf("qwen35: draft-vocab: token %d is past the vocabulary of %d", id, c.VocabSize)
		}
	}
	return ids, nil
}

// rowsView is weights whose matrix name has the rows ids alone, in their order: the output layer
// of draft-vocab.
type rowsView struct {
	llm.Weights
	name string
	ids  []int32
}

func (v rowsView) Get(name string) ([]float32, []int, error) {
	if name != v.name {
		return v.Weights.Get(name)
	}
	var out []float32
	d := 0
	for i := 0; i < len(v.ids); {
		j := i + 1 // a run of consecutive ids, read at once
		for j < len(v.ids) && v.ids[j] == v.ids[j-1]+1 {
			j++
		}
		rows, err := v.Weights.Rows(name, int(v.ids[i]), int(v.ids[j-1])+1)
		if err != nil {
			return nil, nil, err
		}
		if d == 0 {
			d = len(rows) / (j - i)
		}
		out = append(out, rows...)
		i = j
	}
	return out, []int{len(v.ids), d}, nil
}

// narrowed is the config and weights of a program built with opts: with draft-vocab, a vocabulary
// of those tokens and the output layer of their rows.
func (c *MTP) narrowed(w llm.Weights, opts llm.Options) (*Config, llm.Weights, error) {
	ids, err := c.OutputIDs(opts)
	if err != nil || ids == nil {
		return c.Config, w, err
	}
	n := *c.Config
	n.VocabSize = len(ids)
	return &n, rowsView{Weights: w, name: c.Head(), ids: ids}, nil
}

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
	cfg, w, err := c.narrowed(w, opts)
	if err != nil {
		return nil, nil, err
	}
	return buildCached(b, cfg, w, opts, true, false)
}

// Narrow is the layer as a program with the output layer of the k most frequent tokens alone
// (draft-vocab k; DraftVocab) lists its tensors: for estimates of its weights.
func (c *MTP) Narrow(k int) (llm.Model, error) {
	ids, err := DraftVocab(k)
	if err != nil {
		return nil, err
	}
	n := *c.Config
	n.VocabSize = len(ids)
	return &MTP{&n}, nil
}

// PagedLayers is the decoder layer (llm.Paged).
func (c *MTP) PagedLayers() []int { return []int{1} }

var (
	_ llm.Paged        = (*MTP)(nil)
	_ llm.TokenInputs  = (*MTP)(nil)
	_ llm.Prologue     = (*MTP)(nil)
	_ llm.OutputSubset = (*MTP)(nil)
)
