package qwen35

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/llm/experts"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// MoEArch registers the mixture of experts models of Qwen3.5 and Qwen3.6 (qwen3_5_moe:
// Qwen3.6-35B-A3B, 256 experts of 2048 x 512 a layer, 8 a token, and a shared expert). The layers'
// token mixers are the dense models' (Gated DeltaNet and gated full attention); their MLP is
// Qwen3_5MoeSparseMoeBlock. As gpt-oss's (internal/llm/gptoss), the model is cut at every layer's
// experts (llm.Routed): the Neural Engine runs the mixers, the router's logits and the shared
// expert, and the host picks the experts and runs them on its CPU in MXFP4 (llm.HostRouted),
// mapped from a checkpoint whose experts hwnpu-dev mxfp4 wrote in MXFP4 (DeriveMXFP4): the
// release's are bf16, 64 GB of the 72.
var MoEArch = &llm.Arch{
	Name:          "qwen35moe",
	Title:         "Qwen3.5-MoE",
	ModelTypes:    []string{"qwen3_5_moe", "qwen3_5_moe_text"},
	Architectures: []string{"Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM"},
	Format:        "lut8",
	LastRow:       true,
	// 1: a chunk of one row without the matmul of an inner dimension of 1 (deltaRuleChunks), 2026-11-20
	Revision:   2, // 2: win_conv_N_new is the window after the step's tokens (step_window)
	LoadConfig: func(raw []byte) (llm.Model, error) { return ParseMoE(raw) },
	Params: []llm.Param{
		{Name: "delta", Default: "chunk", Usage: "the Gated DeltaNet's recurrence: chunk (over chunks of rows) or rows (one row after the other)"},
	},
}

func init() { llm.Register(MoEArch) }

// MoE is a Qwen3.5-MoE text model.
type MoE struct{ *Config }

var (
	_ llm.HostDerived = (*MoE)(nil)
	_ llm.Paged       = (*MoE)(nil)
	_ llm.Formatter   = (*MoE)(nil)
)

// ParseMoE parses the config.json of a qwen3_5_moe model.
func ParseMoE(raw []byte) (*MoE, error) {
	c, err := parseConfig(raw, true)
	if err != nil {
		return nil, err
	}
	return &MoE{c}, nil
}

// Format is lut8 for the mixers and the shared experts (the head lut6r, moeFormats); the experts
// are MXFP4 on the host.
func (c *MoE) Format() string { return "lut8" }

// moeFormats are the formats ahead of the caller's FormatFor rules (a later rule wins): the head
// in lut6 with a scale a row, as gpt-oss's (its size is the 248320 rows of the vocabulary).
func moeFormats() []string { return []string{"lm_head=lut6r"} }

// Tensors lists the tensors the programs read: not the experts', which the host runs from the
// checkpoint (HostExperts). The checkpoint also holds the vision encoder (model.visual.) and the
// multi-token prediction layer (mtp.), which the text model does not read (DeriveMXFP4 drops them).
func (c *MoE) Tensors() map[string][]int {
	t := c.Config.Tensors()
	d := c.HiddenSize
	for i := 0; i < c.NumLayers; i++ {
		p := fmt.Sprintf("%slayers.%d.mlp.", c.prefix(), i)
		delete(t, p+"gate_proj.weight")
		delete(t, p+"up_proj.weight")
		delete(t, p+"down_proj.weight")
		t[p+"gate.weight"] = []int{c.NumExperts, d}
		t[p+"shared_expert.gate_proj.weight"] = []int{c.SharedIntermediate, d}
		t[p+"shared_expert.up_proj.weight"] = []int{c.SharedIntermediate, d}
		t[p+"shared_expert.down_proj.weight"] = []int{d, c.SharedIntermediate}
		t[p+"shared_expert_gate.weight"] = []int{1, d}
	}
	return t
}

// RoutedLayers are every layer (llm.Routed).
func (c *MoE) RoutedLayers() []int {
	out := make([]int, c.NumLayers)
	for i := range out {
		out[i] = i
	}
	return out
}

// ExpertShape is the experts' sizes.
func (c *MoE) ExpertShape() experts.Shape {
	return experts.Shape{Hidden: c.HiddenSize, Inter: c.MoEIntermediate, Experts: c.NumExperts, TopK: c.TopK}
}

// Routing is Qwen3_5MoeTopKRouter's: the softmax over every expert's logit, the top k of it
// renormalized to sum to 1.
func (c *MoE) Routing() experts.Routing { return experts.Routing{Score: "softmax", Norm: true} }

// ExpertNames is empty: the experts are MXFP4 tensors of a layer's experts, which the host runs as
// they are (HostExperts).
func (c *MoE) ExpertNames(layer int) experts.Names { return experts.Names{} }

// HostExperts are layer's experts in MXFP4 as DeriveMXFP4 writes them, with the router and the
// norm before it (zero centered: its weight plus 1), from which the host guesses the next layer's
// experts.
func (c *MoE) HostExperts(layer int) experts.MXFP4 {
	p := fmt.Sprintf("%slayers.%d.", c.prefix(), layer)
	e := p + "mlp.experts."
	return experts.MXFP4{
		GateUpBlocks: e + "gate_up_proj_blocks", GateUpScales: e + "gate_up_proj_scales",
		DownBlocks: e + "down_proj_blocks", DownScales: e + "down_proj_scales",
		Router: p + "mlp.gate.weight", Norm: p + "post_attention_layernorm.weight", NormOffset: 1,
	}
}

// ExpertAct is the SwiGLU: silu(gate)·up.
func (c *MoE) ExpertAct() experts.Act { return experts.Act{Kind: experts.ActSiLU} }

// Build writes a program without a KV cache (a part, as llm.Routed cuts it).
func (c *MoE) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Cache > 0 {
		return nil, out, fmt.Errorf("qwen35moe: a program with a KV cache has more inputs and outputs; use BuildCached")
	}
	i, o, err := c.BuildCached(b, w, opts)
	if err != nil {
		return nil, out, err
	}
	return i, o[0], nil
}

// BuildCached writes a part of the model (llm.Cached, llm.Routed): every layer ends with its
// experts, so a part is one layer's step between two of the host's (llm.PlanRouted), as
// gpt-oss's:
//
//   - {0, NoLayers}: layer 0 up to its router
//   - {N, 1}: from layer N's experts' output (moe_o_N, added to x) through layer N+1 up to its
//     router, or up to its attention when the host attends for it (Paged); the last part ends
//     with the final norm and the logits
//   - {N, NoLayers} of a full attention layer N the host attended for: from the attention's
//     output (attn_o_N, with the gate attn_gate_N) up to its router
//
// A part that ends at layer N's router outputs moe_x_N, the post-attention norm's output the
// experts take, and moe_logits_N; its y is the residual plus the shared expert's output, to which
// the host's experts' output is added.
func (c *MoE) BuildCached(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins, outs []mlpackage.Feature, err error) {
	cfg := c.Config
	if opts.Group == 0 {
		opts.Group = 16
	}
	if opts.Format == "" {
		opts.Format = c.Format()
	}
	opts.FormatFor = append(moeFormats(), opts.FormatFor...)
	noLayers := opts.Layers == llm.NoLayers
	first := noLayers && opts.Start == 0
	afterAttn := noLayers && opts.Start > 0
	if !noLayers && (opts.Layers <= 0 || opts.Start+opts.Layers > cfg.NumLayers) {
		opts.Layers = cfg.NumLayers - opts.Start
	}
	if opts.Paged > 0 && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("qwen35moe: a program split at its attention (Paged) needs a KV cache")
	}
	if opts.LastRow && opts.Cache <= 0 {
		return nil, nil, fmt.Errorf("qwen35moe: the logits of the last row alone (LastRow) are of a program with a KV cache")
	}
	if _, chunked := llm.CacheChunks(opts.Cache); opts.Cache > 0 && opts.Paged == 0 && chunked {
		return nil, nil, fmt.Errorf("qwen35moe: a KV cache of %d is over the Neural Engine's dimension limit; split the model at its attention (Paged) for a longer one", opts.Cache)
	}
	start, end := opts.Start, opts.Start+opts.Layers
	m := &builder{Graph: &blocks.Graph{Name: "qwen35moe", B: b, W: w, Opts: opts}, cfg: cfg, opts: opts, seq: opts.Seq, d: cfg.HiddenSize, ins: map[string]*mlpackage.Var{}}
	if afterAttn && !m.paged(start) {
		return nil, nil, fmt.Errorf("qwen35moe: a part of no layers from layer %d is the rest of a full attention layer the host attends for", start)
	}
	err = blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, m.seq, m.d)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		h := x
		var y *mlpackage.Var
		switch {
		case first:
			y = m.ident(m.moeHead(0, m.mixer(0, h)))
		case afterAttn:
			s, nh, hd := m.seq, cfg.NumHeads, cfg.HeadDim
			o := m.input(fmt.Sprintf("attn_o_%d", start), 1, nh, s, hd)
			gate := m.input(fmt.Sprintf("attn_gate_%d", start), 1, s, nh, hd)
			h = m.Op("add", map[string]any{"x": m.attnOutput(start, o, gate), "y": h}, h.Shape...)
			y = m.ident(m.moeHead(start, h))
		default:
			if end != start+1 {
				m.Fail("the part of layers %d to %d holds a layer's experts: a part is one layer's step (llm.PlanRouted)", start, end-1)
			}
			h = m.Op("add", map[string]any{"x": h, "y": m.input(fmt.Sprintf("moe_o_%d", start), 1, m.seq, m.d)}, h.Shape...)
			switch {
			case end < cfg.NumLayers && m.paged(end):
				m.beforeAttention(end, h)
				y = m.ident(h)
			case end < cfg.NumLayers:
				y = m.ident(m.moeHead(end, m.mixer(end, h)))
			case opts.NoHead:
				y = m.ident(h)
			default:
				if opts.LastRow {
					h = m.LastRow(h, m.input("last", 1, 1, m.seq))
				}
				h = m.norm(h, cfg.prefix()+"norm.weight")
				if opts.Hidden {
					y = m.ident(h)
				} else {
					y = m.Linear(h, cfg.Head(), "y", cfg.VocabSize)
				}
			}
		}
		outs = []mlpackage.Feature{{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}}
		for _, v := range m.outList {
			outs = append(outs, mlpackage.Feature{Name: v.Name, DType: mlpackage.Float16, Shape: v.Shape})
		}
		ins = append(ins, m.inList...)
	})
	return ins, outs, err
}

// ident is h as the program's output y.
func (m *builder) ident(h *mlpackage.Var) *mlpackage.Var {
	return m.B.NamedOp("identity", "y", map[string]any{"x": h}, mlpackage.Float16, h.Shape...)
}

// mixer is the first half of Qwen3_5MoeDecoderLayer i: the input norm, the token mixer (Gated
// DeltaNet or full attention) and the residual.
func (m *builder) mixer(i int, h *mlpackage.Var) *mlpackage.Var {
	p := m.layer(i)
	n := m.norm(h, p+"input_layernorm.weight")
	if m.linear(i) {
		n = m.gatedDeltaNet(i, n)
	} else {
		n = m.attention(i, n)
	}
	return m.Op("add", map[string]any{"x": n, "y": h}, h.Shape...)
}

// moeHead is Qwen3_5MoeSparseMoeBlock of layer i on the residual h, but for the routed experts:
// it outputs the post-attention norm's output moe_x_N and the router's logits moe_logits_N (the
// router Qwen3_5MoeTopKRouter's weight in fp16: the choice of the experts turns on small
// differences of the logits), and returns h plus the shared expert's output times the sigmoid of
// its gate, to which the host adds the routed experts' output (moe_o_N).
func (m *builder) moeHead(i int, h *mlpackage.Var) *mlpackage.Var {
	p := m.layer(i) + "mlp."
	c := m.cfg
	f := m.norm(h, m.layer(i)+"post_attention_layernorm.weight")
	m.output(f, fmt.Sprintf("moe_x_%d", i))
	m.output(m.f16Linear(f, p+"gate.weight", c.NumExperts), fmt.Sprintf("moe_logits_%d", i))
	sh := m.silu(m.Linear(f, p+"shared_expert.gate_proj.weight", "", c.SharedIntermediate))
	sh = m.Op("mul", map[string]any{"x": sh, "y": m.Linear(f, p+"shared_expert.up_proj.weight", "", c.SharedIntermediate)}, sh.Shape...)
	sh = m.Linear(sh, p+"shared_expert.down_proj.weight", "", m.d)
	g := m.Op("sigmoid", map[string]any{"x": m.f16Linear(f, p+"shared_expert_gate.weight", 1)}, 1, m.seq, 1)
	sh = m.Op("mul", map[string]any{"x": sh, "y": g}, sh.Shape...)
	return m.Op("add", map[string]any{"x": h, "y": sh}, h.Shape...)
}

// f16Linear is x (1, seq, hidden) times the weight name (out, hidden) kept in fp16.
func (m *builder) f16Linear(x *mlpackage.Var, name string, out int) *mlpackage.Var {
	wv, err := m.B.F16s(name, []int{out, m.d}, m.Get(name, out, m.d))
	if err != nil {
		m.Fail("%v", err)
	}
	return m.Op("linear", map[string]any{"x": x, "weight": wv}, 1, m.seq, out)
}

// Derive writes the checkpoint with the experts in MXFP4 (DeriveMXFP4) of the Hugging Face
// repository repo into dir (llm.HostDerived), read over HTTP a layer at a time; a dir with its
// config.json is whole, and left as it is.
func (c *MoE) Derive(ctx context.Context, repo, dir string, log io.Writer) error {
	if raw, err := os.ReadFile(filepath.Join(dir, "config.json")); err == nil && IsDerived(raw) {
		return nil
	}
	fmt.Fprintf(log, "%s: writing a checkpoint with its experts in MXFP4 into %s, read from Hugging Face a layer at a time (72 GB read, 23 GB written; about 30 min at 40 MB/s)\n", repo, dir)
	base, rev, _ := strings.Cut(repo, "@")
	src := NewHTTPSource(ctx, base)
	src.Revision = rev
	return DeriveMXFP4(src, "hf:"+repo, dir, log)
}
