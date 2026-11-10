package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// TextGroups splits Gemma 2's layers into the programs of Text: at most per layers each, as
// even as they go (26 by 6: 6, 5, 5, 5, 5). At int8 a layer is 78 MB and a program past about
// 1 GB of weights is placed on the CPU (design.md 16.85).
func TextGroups(layers, per int) [][2]int {
	n := (layers + per - 1) / per
	var out [][2]int
	first := 0
	for i := 0; i < n; i++ {
		k := layers / n
		if i < layers%n {
			k++
		}
		out = append(out, [2]int{first, k})
		first += k
	}
	return out
}

// Text is Gemma 2's layers first..first+count-1 over the prompt's TextLen rows, causal: input x
// (1, TextLen, hidden), the embedding rows times sqrt(hidden) for the first group, else the
// residual stream; output its residual stream after them, past the last layer the final norm's
// (Gemma 2's last hidden state). The attention's logit softcapping (50) is left out, as on AMD's
// NPU (the rows Sana keeps come out the same, design.md 16.119).
func Text(c Gemma, first, count int) *Part {
	return part(fmt.Sprintf("text_%d", first), func(g *graph) {
		L := TextLen
		x := g.Input("x", 1, L, c.Hidden)
		causal := g.CausalMask("gemma_causal", L, 0)
		cos, sin := g.RopeTables("gemma", milpart.InvFreq(c.HeadDim, c.Theta), L, c.HeadDim)
		for i := first; i < first+count; i++ {
			x = gemmaLayer(g, c, i, x, causal, cos, sin)
		}
		if first+count == c.Layers {
			x = gemmaNorm(g, x, "gemma.norm.weight", c.Eps)
		}
		g.Output(x)
	})
}

// gemmaNorm is Gemma 2's RMS norm, weighted by 1 + w, of x scaled by MaxScaled.
func gemmaNorm(g *graph, x *mlpackage.Var, name string, eps float64) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	w := g.Get(name, n)
	for i := range w {
		w[i]++
	}
	return g.RMSNormWith(g.MaxScaled(x, -1), name, w, eps, 1)
}

func gemmaLayer(g *graph, c Gemma, i int, x, causal, cos, sin *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("gemma.layers.%d.", i)
	L, D, H, G := x.Shape[1], c.HeadDim, c.Heads, c.KVHeads
	h := gemmaNorm(g, x, p+"input_layernorm.weight", c.Eps)
	proj := func(name string, heads int, rope bool) *mlpackage.Var {
		y := g.Reshape(g.Linear(h, p+"self_attn."+name+"_proj.weight", "", heads*D), 1, L, heads, D)
		if rope {
			y = g.ApplyRope(y, cos, sin)
		}
		return g.Transpose(y, 0, 2, 1, 3)
	}
	q, k, v := proj("q", H, true), proj("k", G, true), proj("v", G, false)
	o := g.SDPAGrouped(q, k, v, 1/math.Sqrt(c.QueryScalar), func() *mlpackage.Var { return causal })
	o = g.Linear(g.Merge(o), p+"self_attn.o_proj.weight", "", c.Hidden)
	x = g.Add(x, gemmaNorm(g, o, p+"post_attention_layernorm.weight", c.Eps))
	h = gemmaNorm(g, x, p+"pre_feedforward_layernorm.weight", c.Eps)
	gt := geluTanh(g, g.Linear(h, p+"mlp.gate_proj.weight", "", c.MLP))
	up := g.Linear(h, p+"mlp.up_proj.weight", "", c.MLP)
	f := g.Linear(g.Mul(gt, up), p+"mlp.down_proj.weight", "", c.Hidden)
	return g.Add(x, gemmaNorm(g, f, p+"post_feedforward_layernorm.weight", c.Eps))
}

// geluTanh is the tanh approximation of GELU written as x sigmoid(2 sqrt(2/pi) (x + 0.044715
// x^3)), with exp and real_div as blocks.SiLU: the Neural Engine's own gelu (TANH_APPROXIMATION)
// moved Gemma 2's rows Sana keeps (design.md 16.127).
func geluTanh(g *graph, x *mlpackage.Var) *mlpackage.Var {
	// u = -2 sqrt(2/pi) (x + 0.044715 x^3) = x (a + b x^2)
	a := float32(-2 * math.Sqrt(2/math.Pi))
	u := g.Mul(x, g.AddScalar(g.Scale(g.Mul(x, x), a*0.044715), a))
	// past x = -11 or so exp overflows fp16 to inf and the quotient is 0, as gelu is there
	return g.Div(x, g.AddScalar(g.Unary("exp", u), 1))
}
