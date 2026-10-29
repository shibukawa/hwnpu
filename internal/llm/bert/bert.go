// Package bert converts BERT encoders (Hugging Face `bert` architecture): the sentence embedding
// and classification models built on it, such as all-MiniLM-L6-v2, E5, BGE and GTE.
//
// The structure follows transformers' modeling_bert.py: word, position and token type embeddings
// summed and normalized, then post norm encoder layers of bidirectional multi head attention and
// a GELU MLP. The program's output is the last hidden state; pooling (mean or CLS) and
// normalization are left to the host, where they are a few additions per token.
package bert

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Config is the subset of config.json the conversion needs.
type Config struct {
	Architectures    []string `json:"architectures"`
	VocabSize        int      `json:"vocab_size"`
	HiddenSize       int      `json:"hidden_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	IntermediateSize int      `json:"intermediate_size"`
	HiddenAct        string   `json:"hidden_act"`
	MaxPositions     int      `json:"max_position_embeddings"`
	TypeVocabSize    int      `json:"type_vocab_size"`
	LayerNormEps     float64  `json:"layer_norm_eps"`
	PositionType     string   `json:"position_embedding_type"`
	// Prefix is prepended to every tensor name: "bert." for task models (BertForMaskedLM and
	// the like), which wrap the encoder, and "" for BertModel checkpoints.
	Prefix string `json:"-"`
}

// Arch registers BERT with the whitelist in internal/llm.
var Arch = &llm.Arch{
	Name:          "bert",
	Title:         "BERT",
	ModelTypes:    []string{"bert"},
	Architectures: []string{"BertModel"},
	LoadConfig:    func(raw []byte) (llm.Model, error) { return ParseConfig(raw) },
}

func init() { llm.Register(Arch) }

// LoadConfig reads config.json.
func LoadConfig(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	c, err := ParseConfig(raw)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return c, nil
}

// ParseConfig parses config.json and rejects settings the conversion does not handle.
func ParseConfig(raw []byte) (*Config, error) {
	c := Config{HiddenAct: "gelu", TypeVocabSize: 2, LayerNormEps: 1e-12, PositionType: "absolute"}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("bert: %w", err)
	}
	if c.HiddenSize == 0 || c.NumHeads == 0 || c.HiddenSize%c.NumHeads != 0 {
		return nil, fmt.Errorf("bert: hidden size %d is not a multiple of %d heads", c.HiddenSize, c.NumHeads)
	}
	if c.PositionType != "absolute" {
		return nil, fmt.Errorf("bert: position_embedding_type %q is not supported", c.PositionType)
	}
	switch c.HiddenAct {
	case "gelu", "gelu_new", "gelu_pytorch_tanh", "relu":
	default:
		return nil, fmt.Errorf("bert: hidden_act %q is not supported", c.HiddenAct)
	}
	for _, a := range c.Architectures {
		if a != "BertModel" && strings.HasPrefix(a, "Bert") {
			c.Prefix = "bert."
		}
	}
	return &c, nil
}

// Hidden is the hidden size.
func (c *Config) Hidden() int { return c.HiddenSize }

// Layers is the number of encoder layers.
func (c *Config) Layers() int { return c.NumLayers }

// Output is the last hidden state.
func (c *Config) Output() (int, bool) { return c.HiddenSize, false }

// Embedding names the word embedding table.
func (c *Config) Embedding() string { return c.Prefix + "embeddings.word_embeddings.weight" }

// Tensors lists the tensors the conversion reads (the pooler and task heads are not among them).
func (c *Config) Tensors() map[string][]int {
	d, f, p := c.HiddenSize, c.IntermediateSize, c.Prefix
	t := map[string][]int{
		c.Embedding(): {c.VocabSize, d},
		p + "embeddings.position_embeddings.weight":   {c.MaxPositions, d},
		p + "embeddings.token_type_embeddings.weight": {c.TypeVocabSize, d},
		p + "embeddings.LayerNorm.weight":             {d},
		p + "embeddings.LayerNorm.bias":               {d},
	}
	for i := 0; i < c.NumLayers; i++ {
		l := fmt.Sprintf("%sencoder.layer.%d.", p, i)
		for _, n := range []string{"attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense"} {
			t[l+n+".weight"] = []int{d, d}
			t[l+n+".bias"] = []int{d}
		}
		t[l+"intermediate.dense.weight"] = []int{f, d}
		t[l+"intermediate.dense.bias"] = []int{f}
		t[l+"output.dense.weight"] = []int{d, f}
		t[l+"output.dense.bias"] = []int{d}
		for _, n := range []string{"attention.output.LayerNorm", "output.LayerNorm"} {
			t[l+n+".weight"] = []int{d}
			t[l+n+".bias"] = []int{d}
		}
	}
	return t
}

type builder struct {
	*blocks.Graph
	cfg *Config
	seq int
	d   int
}

// Build writes the program. Its inputs are x, the word embeddings (1, seq, hidden), and mask
// (1, seq), 1 for tokens and 0 for padding; its output y is the last hidden state (1, seq,
// hidden), or the hidden state after the last built layer of a truncated model.
func (c *Config) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = "lut4"
	}
	if opts.Layers <= 0 || opts.Start+opts.Layers > c.NumLayers {
		opts.Layers = c.NumLayers - opts.Start
	}
	if opts.Seq > c.MaxPositions {
		return nil, out, fmt.Errorf("bert: sequence length %d is longer than the %d positions", opts.Seq, c.MaxPositions)
	}
	m := &builder{Graph: &blocks.Graph{Name: "bert", B: b, W: w, Opts: opts}, cfg: c, seq: opts.Seq, d: c.HiddenSize}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		mask := b.Input("mask", mlpackage.Float16, 1, m.seq)
		ins = []mlpackage.Feature{
			{Name: "x", DType: mlpackage.Float16, Shape: x.Shape},
			{Name: "mask", DType: mlpackage.Float16, Shape: mask.Shape},
		}
		h := x
		if opts.Start == 0 {
			h = m.embeddings(h)
		}
		var bias *mlpackage.Var
		if opts.Layers > 0 {
			bias = m.MaskBias(mask)
		}
		for i := opts.Start; i < opts.Start+opts.Layers; i++ {
			h = m.layer(i, h, bias)
		}
		y := b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		out = mlpackage.Feature{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}
	})
	return ins, out, err
}

// embeddings adds the position embeddings of positions 0..seq-1 and the token type 0 embedding
// (one constant, since the sequence length is fixed) and normalizes.
func (m *builder) embeddings(x *mlpackage.Var) *mlpackage.Var {
	p := m.cfg.Prefix + "embeddings."
	pos := m.Get(p+"position_embeddings.weight", m.cfg.MaxPositions, m.d)
	typ := m.Get(p+"token_type_embeddings.weight", m.cfg.TypeVocabSize, m.d)
	add := make([]float32, m.seq*m.d)
	for s := 0; s < m.seq; s++ {
		for j := 0; j < m.d; j++ {
			add[s*m.d+j] = pos[s*m.d+j] + typ[j]
		}
	}
	h := m.Op("add", map[string]any{"x": x, "y": m.F16s(p+"position_and_type", []int{1, m.seq, m.d}, add)}, x.Shape...)
	return m.LayerNorm(h, p+"LayerNorm.weight", p+"LayerNorm.bias", m.d, m.cfg.LayerNormEps)
}

// layer is BertLayer: self attention with a residual and LayerNorm, then the GELU MLP with a
// residual and LayerNorm.
func (m *builder) layer(i int, h, bias *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("%sencoder.layer.%d.", m.cfg.Prefix, i)
	s, d, nh := m.seq, m.d, m.cfg.NumHeads
	hd := d / nh
	heads := func(name string) *mlpackage.Var {
		t := m.LinearBias(h, p+name+".weight", p+name+".bias", "", d)
		t = m.Op("reshape", map[string]any{"x": t, "shape": m.B.Ints(1, int32(s), int32(nh), int32(hd))}, 1, s, nh, hd)
		return m.Op("transpose", map[string]any{"x": t, "perm": m.B.Ints(0, 2, 1, 3)}, 1, nh, s, hd)
	}
	q, k, v := heads("attention.self.query"), heads("attention.self.key"), heads("attention.self.value")
	o := m.SDPA(q, k, v, func() *mlpackage.Var { return bias })
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, s, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(s), int32(d))}, 1, s, d)
	a := m.LinearBias(o, p+"attention.output.dense.weight", p+"attention.output.dense.bias", "", d)
	a = m.Op("add", map[string]any{"x": a, "y": h}, 1, s, d)
	h = m.LayerNorm(a, p+"attention.output.LayerNorm.weight", p+"attention.output.LayerNorm.bias", d, m.cfg.LayerNormEps)

	f := m.LinearBias(h, p+"intermediate.dense.weight", p+"intermediate.dense.bias", "", m.cfg.IntermediateSize)
	f = m.Activation(f, m.cfg.HiddenAct)
	f = m.LinearBias(f, p+"output.dense.weight", p+"output.dense.bias", "", d)
	f = m.Op("add", map[string]any{"x": f, "y": h}, 1, s, d)
	return m.LayerNorm(f, p+"output.LayerNorm.weight", p+"output.LayerNorm.bias", d, m.cfg.LayerNormEps)
}
