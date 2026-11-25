package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/bonsaiai/internal/imagegen/milpart"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
)

// TextScale is what Qwen3-4B's residual stream is divided by in its programs: the host's
// embedding rows and every layer's output projections are scaled down by it (the norms are scale
// free). Qwen3's attention sinks reach the ten thousands in a few channels (Qwen3-VL-8B's 13500,
// docs/design/image-apple.md).
const TextScale = 8

// TextGroup is Qwen3-4B's layers first..first+count-1 over the TextRows rows: input x (1, rows,
// hidden), the residual stream divided by TextScale (the first group's: the embedding rows divided
// by it, the pad's past the prompt), and kbias (1, 1, 1, rows), 0 for the prompt's keys and
// -10000 for the pads': the prompt's rows attend causally, the pads to the prompt's keys only
// (transformers' mask: a pad is no key, not even its own). Output the residual stream after them,
// as the pipeline's hidden states (no final norm).
func TextGroup(t Text, first, count int) *Part {
	return part(fmt.Sprintf("text_%d", first), func(g *graph) {
		L := TextRows
		x := g.Input("x", 1, L, t.Hidden)
		kbias := g.Input("kbias", 1, 1, 1, L)
		mask := g.Add(g.CausalMask("qwen_causal", L, 0), kbias)
		cos, sin := g.RopeTables("qwen", milpart.InvFreq(t.HeadDim, t.Theta), L, t.HeadDim)
		for i := first; i < first+count; i++ {
			x = qwenLayer(g, t, i, x, mask, cos, sin)
		}
		g.Output(x)
	})
}

func qwenLayer(g *graph, t Text, i int, x, mask, cos, sin *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("text.model.layers.%d.", i)
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
	o := g.SDPAGrouped(q, k, v, 1/math.Sqrt(float64(D)), func() *mlpackage.Var { return mask })
	x = g.Add(x, g.LinearS(g.Merge(o), p+"self_attn.o_proj.weight", "", t.Hidden, 1.0/TextScale))
	h = g.ScaledRMSNorm(x, p+"post_attention_layernorm.weight", t.Eps)
	return g.Add(x, g.SwiGLU(h, p+"mlp.gate_proj.weight", p+"mlp.up_proj.weight", p+"mlp.down_proj.weight", t.MLP, t.Hidden, 1.0/TextScale))
}
