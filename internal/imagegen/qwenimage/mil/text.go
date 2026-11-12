package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// TextScale is what Qwen3-VL-8B's residual stream is divided by in its programs: the host's
// embedding rows and every layer's output projections are scaled down by it (the norms are scale
// free). From layer 6 on one token's rows reach 13500 in float32 (an attention sink; design.md
// 16.165): 3400 here, whose square a norm never takes (MaxScaled first).
const TextScale = 4

// TextGroup is Qwen3-VL-8B's language model's layers first..first+count-1 over TextLen positions,
// causal: input x (1, TextLen, hidden), the residual stream divided by TextScale (the first
// group's: the embedding rows divided by it, zeros past the prompt); output the same after them.
// The last group (last) then gives the DiT's text rows: txt_in (zero-centred RMS norm, linear, tanh
// GELU, linear) of its output, divided by DitScale (TextBlocks' input). The text's state is the
// last layer's output before the final norm, as the pipeline's hook takes it.
func TextGroup(t Text, d Dit, first, count int, last bool) *Part {
	return part(fmt.Sprintf("text_%d", first), func(g *graph) {
		x := g.Input("x", 1, TextLen, t.Hidden)
		causal := g.CausalMask("qwen_causal", TextLen, 0)
		cos, sin := g.RopeTables("qwen", milpart.InvFreq(t.HeadDim, t.Theta), TextLen, t.HeadDim)
		for i := first; i < first+count; i++ {
			x = qwenLayer(g, t, i, x, causal, cos, sin)
		}
		if last {
			x = textIn(g, d, x)
		}
		g.Output(x)
	})
}

func qwenLayer(g *graph, t Text, i int, x, causal, cos, sin *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("text.layers.%d.", i)
	L, D := x.Shape[1], t.HeadDim
	h := g.ScaledRMSNorm(x, p+"input_layernorm.weight", t.Eps)
	proj := func(name string, heads int, norm string) *mlpackage.Var {
		y := g.Reshape(g.Linear(h, p+"self_attn."+name+"_proj.weight", "", heads*D), 1, L, heads, D)
		if norm != "" {
			y = g.ApplyRope(g.ScaledRMSNorm(y, p+"self_attn."+norm+".weight", t.Eps), cos, sin)
		}
		return g.Transpose(y, 0, 2, 1, 3)
	}
	q, k, v := proj("q", t.Heads, "q_norm"), proj("k", t.KVHeads, "k_norm"), proj("v", t.KVHeads, "")
	o := g.SDPAGrouped(q, k, v, 1/math.Sqrt(float64(D)), func() *mlpackage.Var { return causal })
	x = g.Add(x, g.LinearS(g.Merge(o), p+"self_attn.o_proj.weight", "", t.Hidden, 1.0/TextScale))
	h = g.ScaledRMSNorm(x, p+"post_attention_layernorm.weight", t.Eps)
	return g.Add(x, g.SwiGLU(h, p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight", t.MLP, t.Hidden, 1.0/TextScale))
}

// textIn is the DiT's txt_in of x (the text's state, any scale): the zero-centred RMS norm (its
// weight + 1), in_layer, the tanh GELU (written with exp: the Neural Engine's own gelu is
// inaccurate, new-model-checklist 3.3), out_layer divided by DitScale.
func textIn(g *graph, d Dit, x *mlpackage.Var) *mlpackage.Var {
	E := d.Hidden
	w := g.Get("dit.txt_in.text_norm.weight", E)
	for i := range w {
		w[i]++
	}
	h := g.RMSNormWith(g.MaxScaled(x, -1), "dit.txt_in.text_norm.weight", w, d.Eps, 1)
	h = g.Linear(h, "dit.txt_in.in_layer.weight", "", E)
	h = tanhGELU(g, h)
	return g.LinearS(h, "dit.txt_in.out_layer.weight", "", E, 1.0/DitScale)
}

// tanhGELU is x / (1 + exp(-1.5958 (x + 0.044715 x^3))), the tanh GELU.
func tanhGELU(g *graph, x *mlpackage.Var) *mlpackage.Var {
	x3 := g.Mul(g.Mul(x, x), x)
	z := g.Scale(g.Add(x, g.Scale(x3, 0.044715)), -1.5957691)
	return g.Div(x, g.AddScalar(g.Unary("exp", z), 1))
}
