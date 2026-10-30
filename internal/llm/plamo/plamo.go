// Package plamo converts PLaMo's bidirectional encoder (PlamoBiModel in the modeling_plamo.py
// that pfnet/plamo-embedding-1b ships): the PLaMo decoder with its attention made bidirectional,
// for sentence embeddings.
//
// The structure follows modeling_plamo.py: pre-norm layers of attention and a SwiGLU MLP, each
// with a residual, and a final RMSNorm. The attention has one fused qkv projection, multi query
// keys and values, an RMS norm of each head's queries and keys with a weight per head, and
// rotate_half RoPE; it sees every token except padding. The program's output is the last hidden
// state; pooling (the mean over the text's own tokens, leaving out BOS and a query's instruction)
// is left to the host.
//
// PLaMo's causal language models (PlamoForCausalLM, PLaMo 2) are other architectures and are
// rejected.
package plamo

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// qkNormEps is the epsilon of the query and key norms, fixed in modeling_plamo.py's Attention.
const qkNormEps = 1e-6

// ropeTheta is the RoPE base: RotaryEmbedding's default, which Attention does not override.
const ropeTheta = 10000

// Config is the subset of config.json the conversion needs.
type Config struct {
	Architectures    []string `json:"architectures"`
	VocabSize        int      `json:"vocab_size"`
	HiddenSize       int      `json:"hidden_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"hidden_size_per_head"`
	IntermediateSize int      `json:"intermediate_size"`
	MaxPositions     int      `json:"max_position_embeddings"`
	NormEps          float64  `json:"rms_norm_eps"`
	// Mixture of experts and evaluation time quantization, which the conversion does not handle.
	NExpert       *int `json:"n_expert"`
	SparseStep    *int `json:"sparse_step"`
	AttentionNBit *int `json:"eval_attention_n_bit"`
	MLPNBit       *int `json:"eval_mlp_n_bit"`
}

// Arch registers PLaMo's encoder with the whitelist in internal/llm. It is selected by the
// architectures value: model_type "plamo" is shared with the causal PLaMo models.
var Arch = &llm.Arch{
	Name:          "plamo-bi",
	Title:         "PLaMo bidirectional encoder",
	Architectures: []string{"PlamoBiModel"},
	LoadConfig:    func(raw []byte) (llm.Model, error) { return ParseConfig(raw) },
	// 4 bits lose too much of the sentence embedding: its cosine with fp16's was 0.91 to 0.94
	// in int4 (per row or in groups), 0.9995 in int8 (PLaMo-Embedding-1B, TestIntelIRCheckpoint)
	IntelFormat: "int8",
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

// ParseConfig parses config.json and rejects settings the conversion does not handle. The
// defaults are PlamoConfig's.
func ParseConfig(raw []byte) (*Config, error) {
	c := Config{VocabSize: 32000, HiddenSize: 4096, IntermediateSize: 13312, NumLayers: 32, NumHeads: 32, NumKVHeads: 4,
		HeadDim: 128, MaxPositions: 2048, NormEps: 1e-6}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("plamo: %w", err)
	}
	for _, a := range c.Architectures {
		if a != "PlamoBiModel" {
			return nil, fmt.Errorf("plamo: architecture %s is not supported (only PlamoBiModel, the bidirectional encoder)", a)
		}
	}
	if c.NumHeads <= 0 || c.NumKVHeads <= 0 || c.NumHeads%c.NumKVHeads != 0 {
		return nil, fmt.Errorf("plamo: %d attention heads are not a multiple of %d key value heads", c.NumHeads, c.NumKVHeads)
	}
	if c.HeadDim <= 0 || c.HeadDim%2 != 0 {
		return nil, fmt.Errorf("plamo: hidden_size_per_head %d is not even", c.HeadDim)
	}
	if c.NExpert != nil || c.SparseStep != nil {
		return nil, fmt.Errorf("plamo: mixture of experts (n_expert, sparse_step) is not supported")
	}
	if c.AttentionNBit != nil || c.MLPNBit != nil {
		return nil, fmt.Errorf("plamo: eval_attention_n_bit and eval_mlp_n_bit are not supported")
	}
	return &c, nil
}

// Hidden is the hidden size.
func (c *Config) Hidden() int { return c.HiddenSize }

// Layers is the number of layers.
func (c *Config) Layers() int { return c.NumLayers }

// Output is the last hidden state, after the final norm.
func (c *Config) Output() (int, bool) { return c.HiddenSize, false }

// Embedding names the token embedding table.
func (c *Config) Embedding() string { return "embed_tokens.weight" }

// Tensors lists the tensors the conversion reads: all of PlamoBiModel's.
func (c *Config) Tensors() map[string][]int {
	d, f, hd := c.HiddenSize, c.IntermediateSize, c.HeadDim
	t := map[string][]int{
		c.Embedding(): {c.VocabSize, d},
		"norm.weight": {d},
	}
	for i := 0; i < c.NumLayers; i++ {
		l := fmt.Sprintf("layers.layers.%d.", i)
		t[l+"norm.weight"] = []int{d}
		t[l+"norm2.weight"] = []int{d}
		t[l+"self_attn.qkv_proj.weight"] = []int{(c.NumHeads + 2*c.NumKVHeads) * hd, d}
		t[l+"self_attn.o_proj.weight"] = []int{d, c.NumHeads * hd}
		t[l+"self_attn.q_weight"] = []int{c.NumHeads, hd}
		t[l+"self_attn.k_weight"] = []int{c.NumKVHeads, hd}
		t[l+"mlp.gate_up_proj.weight"] = []int{2 * f, d}
		t[l+"mlp.down_proj.weight"] = []int{d, f}
	}
	return t
}

// invFreq is RotaryEmbedding's inverse frequencies for head_dim.
func (c *Config) invFreq() []float64 {
	f := make([]float64, c.HeadDim/2)
	for j := range f {
		f[j] = 1 / math.Pow(ropeTheta, float64(2*j)/float64(c.HeadDim))
	}
	return f
}

type builder struct {
	*blocks.Graph
	cfg *Config
	seq int
	d   int
}

// Build writes the program. Its inputs are x, the token embeddings (1, seq, hidden), and mask
// (1, seq), 1 for tokens and 0 for padding; its output y is the last hidden state (1, seq,
// hidden), or the hidden state after the last built layer of a truncated model.
//
// The positions are 0..seq-1 whatever the padding. That matches the reference for a text on its
// own; transformers' left padded batches shift the positions instead, which changes nothing:
// RoPE depends only on the distance between positions, and the attention is bidirectional.
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
	if opts.Cache > 0 {
		return nil, out, fmt.Errorf("plamo: an encoder has no KV cache")
	}
	if opts.Seq > c.MaxPositions {
		return nil, out, fmt.Errorf("plamo: sequence length %d is longer than the %d positions", opts.Seq, c.MaxPositions)
	}
	m := &builder{Graph: &blocks.Graph{Name: "plamo", B: b, W: w, Opts: opts}, cfg: c, seq: opts.Seq, d: c.HiddenSize}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		mask := b.Input("mask", mlpackage.Float16, 1, m.seq)
		ins = []mlpackage.Feature{
			{Name: "x", DType: mlpackage.Float16, Shape: x.Shape},
			{Name: "mask", DType: mlpackage.Float16, Shape: mask.Shape},
		}
		// Nothing comes between the embedding lookup and layer 0.
		h := x
		if opts.Layers > 0 {
			bias := m.MaskBias(mask)
			cos, sin := m.RopeTables("plamo", c.invFreq(), m.seq, c.HeadDim)
			for i := opts.Start; i < opts.Start+opts.Layers; i++ {
				h = m.layer(i, h, bias, cos, sin)
			}
		}
		if opts.Start+opts.Layers == c.NumLayers && !opts.NoHead {
			h = m.RMSNorm(h, "norm.weight", m.d, c.NormEps, 1)
		}
		y := b.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
		out = mlpackage.Feature{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}
	})
	return ins, out, err
}

// layer is PlamoDecoderLayer (ModifiedPlamoDecoderLayer in PlamoBiModel): norm, attention and a
// residual, then norm2, the MLP and a residual.
func (m *builder) layer(i int, h, bias, cos, sin *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("layers.layers.%d.", i)
	a := m.attention(p+"self_attn.", m.RMSNorm(h, p+"norm.weight", m.d, m.cfg.NormEps, 1), bias, cos, sin)
	h = m.Op("add", map[string]any{"x": h, "y": a}, h.Shape...)
	f := m.mlp(p+"mlp.", m.RMSNorm(h, p+"norm2.weight", m.d, m.cfg.NormEps, 1))
	return m.Op("add", map[string]any{"x": h, "y": f}, h.Shape...)
}

// attention is Attention (ModifiedAttention, not causal): the fused qkv projection split into
// queries, keys and values, the per head norms of the queries and keys, RoPE, attention over the
// tokens that are not padding, and o_proj.
func (m *builder) attention(p string, h, bias, cos, sin *mlpackage.Var) *mlpackage.Var {
	s, nh, nkv, hd := m.seq, m.cfg.NumHeads, m.cfg.NumKVHeads, m.cfg.HeadDim
	heads := func(begin, n int) *mlpackage.Var {
		t := m.LinearRows(h, p+"qkv_proj.weight", begin, begin+n*hd)
		return m.Op("reshape", map[string]any{"x": t, "shape": m.B.Ints(1, int32(s), int32(n), int32(hd))}, 1, s, n, hd)
	}
	toHeads := func(t *mlpackage.Var) *mlpackage.Var {
		return m.Op("transpose", map[string]any{"x": t, "perm": m.B.Ints(0, 2, 1, 3)}, t.Shape[0], t.Shape[2], t.Shape[1], t.Shape[3])
	}
	q := heads(0, nh)
	k := heads(nh*hd, nkv)
	v := heads((nh+nkv)*hd, nkv)
	q = toHeads(m.ApplyRope(m.headNorm(q, p+"q_weight"), cos, sin))
	k = toHeads(m.ApplyRope(m.headNorm(k, p+"k_weight"), cos, sin))
	o := m.SDPA(q, k, toHeads(v), func() *mlpackage.Var { return bias })
	o = m.Op("transpose", map[string]any{"x": o, "perm": m.B.Ints(0, 2, 1, 3)}, 1, s, nh, hd)
	o = m.Op("reshape", map[string]any{"x": o, "shape": m.B.Ints(1, int32(s), int32(nh*hd))}, 1, s, nh*hd)
	return m.Linear(o, p+"o_proj.weight", "", m.d)
}

// headNorm is the query or key norm of Attention: _rms_norm without a weight over each head's
// head_dim, then the named weight (heads, head_dim), one per head and dimension. x is (1, seq,
// heads, head_dim).
func (m *builder) headNorm(x *mlpackage.Var, name string) *mlpackage.Var {
	sh := x.Shape
	gamma := m.F16s(name, []int{1, 1, sh[2], sh[3]}, m.Get(name, sh[2], sh[3]))
	msShape := []int{sh[0], sh[1], sh[2], 1}
	sq := m.Op("mul", map[string]any{"x": x, "y": x}, sh...)
	ms := m.Op("reduce_mean", map[string]any{"x": sq, "axes": m.B.Ints(-1), "keep_dims": m.B.ScalarBool(true)}, msShape...)
	r := m.Op("rsqrt", map[string]any{"x": ms, "epsilon": m.B.ScalarF16(qkNormEps)}, msShape...)
	y := m.Op("mul", map[string]any{"x": x, "y": r}, sh...)
	return m.Op("mul", map[string]any{"x": y, "y": gamma}, sh...)
}

// mlp is DenseMLP: gate_up_proj, whose first half is the gate and second half the up
// projection (_swiglu), then down_proj. The halves are two linear layers, which keeps each
// output well inside the Neural Engine's 16384 per dimension.
func (m *builder) mlp(p string, h *mlpackage.Var) *mlpackage.Var {
	f := m.cfg.IntermediateSize
	gate := m.LinearRows(h, p+"gate_up_proj.weight", 0, f)
	up := m.LinearRows(h, p+"gate_up_proj.weight", f, 2*f)
	gate = m.Op("silu", map[string]any{"x": gate}, gate.Shape...)
	g := m.Op("mul", map[string]any{"x": gate, "y": up}, gate.Shape...)
	return m.Linear(g, p+"down_proj.weight", "", m.d)
}
