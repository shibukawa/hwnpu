package mil

import (
	"fmt"
	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// The text encoder's and the adapter's shape (text_encoder/config.json, text_conditioner/config.json).
const (
	TextRows      = 512 // the pipeline's max_sequence_length
	textHidden    = 1024
	qwenLayers    = 28
	qwenHeads     = 16
	qwenKVHeads   = 8
	qwenHeadDim   = 128
	qwenMLP       = 3072
	qwenTheta     = 1e6
	qwenEps       = 1e-6
	adapterLayers = 6
	adapterHeads  = 16
	adapterMLP    = 4096
)

// Text is Qwen3-0.6B over the prompt's rows, then the LLM adapter: inputs qx and tx (1, 512,
// 1024), the embedding rows of the prompt's Qwen3 tokens and of its T5 tokens (zeros past
// them), qbias and tbias (1, 1, 1, 512), 0 for each token and -10000 past them (the adapter's
// masks of the Qwen3 states and of the T5 rows). Output y (1, 512, 1024), the conditioning,
// valid in the T5 tokens' rows.
func Text() *Part {
	return part("text", func(g *graph) {
		L, E := TextRows, textHidden
		qx := g.Input("qx", 1, L, E)
		tx := g.Input("tx", 1, L, E)
		qbias := g.Input("qbias", 1, 1, 1, L)
		tbias := g.Input("tbias", 1, 1, 1, L)
		causal := g.CausalMask("qwen_causal", L, 0)
		cos, sin := g.RopeTables("qwen", milpart.InvFreq(qwenHeadDim, qwenTheta), L, qwenHeadDim)
		h := qx
		for i := 0; i < qwenLayers; i++ {
			h = qwenLayer(g, i, h, causal, cos, sin)
		}
		src := g.ScaledRMSNorm(h, "qwen.norm.weight", qwenEps)
		acos, asin := g.RopeTables("adapter", milpart.InvFreq(E/adapterHeads, 1e4), L, E/adapterHeads)
		x := tx
		for i := 0; i < adapterLayers; i++ {
			x = adapterBlock(g, i, x, src, tbias, qbias, acos, asin)
		}
		x = g.LinearBias(x, "dit.llm_adapter.out_proj.weight", "dit.llm_adapter.out_proj.bias", "", E)
		g.Output(g.ScaledRMSNorm(x, "dit.llm_adapter.norm.weight", 1e-6))
	})
}

func qwenLayer(g *graph, i int, x, causal, cos, sin *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("qwen.layers.%d.", i)
	L, D := x.Shape[1], qwenHeadDim
	h := g.ScaledRMSNorm(x, p+"input_layernorm.weight", qwenEps)
	proj := func(name string, heads int, norm string) *mlpackage.Var {
		y := g.Reshape(g.Linear(h, p+"self_attn."+name+"_proj.weight", "", heads*D), 1, L, heads, D)
		if norm != "" {
			y = g.ApplyRope(g.ScaledRMSNorm(y, p+"self_attn."+norm+".weight", qwenEps), cos, sin)
		}
		return g.Transpose(y, 0, 2, 1, 3)
	}
	q, k, v := proj("q", qwenHeads, "q_norm"), proj("k", qwenKVHeads, "k_norm"), proj("v", qwenKVHeads, "")
	o := g.QChunkAttention(q, k, v, 1/math.Sqrt(float64(D)), causal, 0)
	x = g.Add(x, g.Linear(g.Merge(o), p+"self_attn.o_proj.weight", "", textHidden))
	h = g.ScaledRMSNorm(x, p+"post_attention_layernorm.weight", qwenEps)
	return g.Add(x, g.SwiGLU(h, p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight", qwenMLP, textHidden, 1))
}

func adapterBlock(g *graph, i int, x, src, tbias, qbias, cos, sin *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("dit.llm_adapter.blocks.%d.", i)
	E, H := textHidden, adapterHeads
	D := E / H
	attend := func(a string, h, kv, mask *mlpackage.Var) *mlpackage.Var {
		pa := p + a + "."
		head := func(y *mlpackage.Var, norm string) *mlpackage.Var {
			y = g.Reshape(y, 1, y.Shape[1], H, D)
			if norm != "" {
				y = g.ApplyRope(g.ScaledRMSNorm(y, pa+norm+".weight", 1e-6), cos, sin)
			}
			return g.Transpose(y, 0, 2, 1, 3)
		}
		q := head(g.Linear(h, pa+"q_proj.weight", "", E), "q_norm")
		k := head(g.Linear(kv, pa+"k_proj.weight", "", E), "k_norm")
		v := head(g.Linear(kv, pa+"v_proj.weight", "", E), "")
		o := g.QChunkAttention(q, k, v, 1/math.Sqrt(float64(D)), mask, 0)
		return g.Linear(g.Merge(o), pa+"o_proj.weight", "", E)
	}
	h := g.ScaledRMSNorm(x, p+"norm_self_attn.weight", 1e-6)
	x = g.Add(x, attend("self_attn", h, h, tbias))
	h = g.ScaledRMSNorm(x, p+"norm_cross_attn.weight", 1e-6)
	x = g.Add(x, attend("cross_attn", h, src, qbias))
	h = g.ScaledRMSNorm(x, p+"norm_mlp.weight", 1e-6)
	h = g.Activation(g.LinearBias(h, p+"mlp.0.weight", p+"mlp.0.bias", "", adapterMLP), "gelu")
	return g.Add(x, g.LinearBias(h, p+"mlp.2.weight", p+"mlp.2.bias", "", E))
}
