package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// TextScale is what Qwen3-VL's residual stream is divided by in its programs: the host's embedding
// rows and every layer's output projections are scaled down by it (the norms are scale free), as
// Qwen3's in FLUX.2 klein (its attention sinks reach the ten thousands in a few channels).
const TextScale = 8

// TextGroup is Qwen3-VL's language model's layers first..first+count-1 over rows rows: input x
// (1, rows, hidden), the residual stream divided by TextScale (the first group's: the embedding
// rows divided by it, the pad's past the prompt); kbias (1, 1, 1, rows), 0 for the prompt's keys
// and -10000 for the pads' (the prompt's rows attend causally, the pads to the prompt's keys only:
// transformers' mask); cos and sin (1, rows, 1, head_dim), the rotary embedding of each row's
// position (the prompt's its index, the pads' 1: the pipeline passes the position ids of
// transformers 4.57, which it was trained with). Output the residual stream after them, as the
// pipeline's hidden states (no final norm).
func TextGroup(t Text, rows, first, count int) *Part {
	return part(fmt.Sprintf("text_%d", first), func(g *graph) {
		x := g.Input("x", 1, rows, t.Hidden)
		kbias := g.Input("kbias", 1, 1, 1, rows)
		cos := g.Input("cos", 1, rows, 1, t.HeadDim)
		sin := g.Input("sin", 1, rows, 1, t.HeadDim)
		mask := g.Add(g.CausalMask(fmt.Sprintf("qwen_causal_%d", rows), rows, 0), kbias)
		for i := first; i < first+count; i++ {
			x = qwenLayer(g, t, i, x, mask, cos, sin)
		}
		g.Output(x)
	})
}

func qwenLayer(g *graph, t Text, i int, x, mask, cos, sin *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("text.model.language_model.layers.%d.", i)
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

// TextPositions are the position ids of the text's rows rows with n tokens of prompt: 0..n-1, then
// 1 for each pad (the pipeline's position_ids.masked_fill(attention_mask == 0, 1)).
func TextPositions(n, rows int) []int {
	pos := make([]int, rows)
	for i := range pos {
		pos[i] = 1
		if i < n {
			pos[i] = i
		}
	}
	return pos
}

// TextRope is the rotary embedding of positions as TextGroup's cos and sin take them (each
// (rows, head_dim): the angle of frequency j at j and at j + head_dim / 2, Qwen's rotate_half).
func TextRope(t Text, positions []int) (cos, sin []float32) {
	D, half := t.HeadDim, t.HeadDim/2
	cos, sin = make([]float32, len(positions)*D), make([]float32, len(positions)*D)
	for r, p := range positions {
		for j := 0; j < half; j++ {
			a := float64(p) / math.Pow(t.Theta, float64(2*j)/float64(D))
			c, s := float32(math.Cos(a)), float32(math.Sin(a))
			cos[r*D+j], cos[r*D+j+half] = c, c
			sin[r*D+j], sin[r*D+j+half] = s, s
		}
	}
	return cos, sin
}
