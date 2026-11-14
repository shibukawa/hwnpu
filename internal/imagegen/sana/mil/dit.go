package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// The linear attentions' keys are divided by these before their sums over the tokens, and each
// query row (a head's 32 values at a token) by its largest value: N / D does not change, and the
// sums stay within fp16. Measured in fp32 at 1024 x 1024 (design.md 16.145), the DiT's V^T K
// reaches 2.1e4 and N 1.6e5, the VAE's 2.4e5 and 4.5e7.
const (
	DitKeyScale = 64
	VaeKeyScale = 256
)

// captionScale divides the caption projection's second linear (Caption).
const captionScale = 64

// Caption is the caption projection (linear, tanh GELU, linear) and its RMS norm, then every
// block's cross-attention keys and values of it (with their norms): they do not depend on the
// step, so they are computed once a prompt and the steps' programs hold none of their weights
// (new-model-checklist 7.1). Input text (1, TextRows, 2304), the rows of Gemma 2's last hidden
// state Sana keeps (zeros past them); output kv (1, 2 Layers, TextRows, CrossHeads x
// CrossHeadDim): block i's keys at 2i, its values at 2i + 1, what Blocks takes.
func Caption(c Dit) *Part {
	return part("caption", func(g *graph) {
		y := g.Input("text", 1, TextRows, c.Caption)
		h := g.LinearBias(y, "dit.caption_projection.linear_1.weight", "dit.caption_projection.linear_1.bias", "", c.Hidden)
		h = g.Activation(h, "gelu_pytorch_tanh")
		// linear_2 times 1/captionScale: the norm after it does not change. The prompt's first row
		// (Gemma's <bos>) reaches 3.8e4 there in SANA 1.5 (fp32), and on the Neural Engine it came
		// out NaN, so that the unconditional call (that row alone) was off (design.md 16.145)
		b2 := g.Get("dit.caption_projection.linear_2.bias", c.Hidden)
		for i := range b2 {
			b2[i] /= captionScale
		}
		h = g.LinearS(h, "dit.caption_projection.linear_2.weight", "", c.Hidden, 1.0/captionScale)
		h = g.Add(h, g.F16s("dit.caption_projection.linear_2.bias", []int{c.Hidden}, b2))
		ctx := g.ScaledRMSNorm(h, "dit.caption_norm.weight", 1e-5)
		var ctxd *mlpackage.Var
		if !c.QKNorm {
			// every caption row less the first (the prompt's <bos>, always kept): the keys less
			// the first key, so that the scores are near their spread (design.md 16.119)
			ctxd = g.Sub(ctx, g.Slice(ctx, 1, 0, 1))
		}
		HD := c.CrossHeads * c.CrossHeadDim
		var kvs []*mlpackage.Var
		for i := 0; i < c.Layers; i++ {
			a := fmt.Sprintf("dit.transformer_blocks.%d.attn2.", i)
			var k *mlpackage.Var
			if ctxd != nil {
				k = g.Linear(ctxd, a+"to_k.weight", "", HD) // the bias cancels
			} else {
				k = g.LinearBias(ctx, a+"to_k.weight", a+"to_k.bias", "", HD)
			}
			if c.QKNorm {
				k = g.ScaledRMSNorm(k, a+"norm_k.weight", 1e-5)
			}
			v := g.LinearBias(ctx, a+"to_v.weight", a+"to_v.bias", "", HD)
			kvs = append(kvs, g.Reshape(k, 1, 1, TextRows, HD), g.Reshape(v, 1, 1, TextRows, HD))
		}
		g.Output(g.Concat(1, kvs...))
	})
}

// Blocks is the DiT's blocks first..first+count-1: inputs x (1, tokens, hidden), the residual
// stream (for the first group, the latents channel last (1, tokens, 32), which it embeds); t6
// (1, 1, 6 hidden), the time embedding's modulations (the host's, sana.TimeFile); kv (1, 2 count,
// TextRows, CrossHeads x CrossHeadDim), these blocks' keys and values of Caption's output; cbias
// (1, 1, 1, TextRows), 0 for the caption's rows that count and -10000 past them. Output y, the
// residual stream after them.
func Blocks(c Dit, first, count int) *Part {
	return part(fmt.Sprintf("dit_blocks_%d", first), func(g *graph) {
		n, E := c.Tokens(), c.Hidden
		var x *mlpackage.Var
		if first == 0 {
			x = g.LinearBias(g.Input("x", 1, n, c.Latent), "dit.patch_embed.proj.weight", "dit.patch_embed.proj.bias", "", E)
		} else {
			x = g.Input("x", 1, n, E)
		}
		t6 := g.Input("t6", 1, 1, 6*E)
		HD := c.CrossHeads * c.CrossHeadDim
		kv := g.Input("kv", 1, 2*count, TextRows, HD)
		cbias := g.Input("cbias", 1, 1, 1, TextRows)
		for i := first; i < first+count; i++ {
			j := 2 * (i - first)
			k := g.Reshape(g.Slice(kv, 1, j, j+1), 1, TextRows, HD)
			v := g.Reshape(g.Slice(kv, 1, j+1, j+2), 1, TextRows, HD)
			x = block(g, c, i, x, t6, k, v, cbias)
		}
		g.Output(x)
	})
}

func block(g *graph, c Dit, i int, x, t6, kk, vv, cbias *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("dit.transformer_blocks.%d.", i)
	E, n := c.Hidden, c.Tokens()
	sst := g.F16s(p+"scale_shift_table", []int{1, 1, 6 * E}, g.Get(p+"scale_shift_table", 6, E))
	m := g.Add(sst, t6)
	mod := func(j int) *mlpackage.Var { return g.Slice(m, -1, j*E, (j+1)*E) }
	modulate := func(x *mlpackage.Var, shift, scale int) *mlpackage.Var {
		return g.Add(g.Mul(g.ScaledLayerNorm(x, 1e-6), g.AddScalar(mod(scale), 1)), mod(shift))
	}
	// linear self-attention
	h := modulate(x, 0, 1)
	a := p + "attn1."
	q := g.Linear(h, a+"to_q.weight", "", E)
	k := g.Linear(h, a+"to_k.weight", "", E)
	v := g.Linear(h, a+"to_v.weight", "", E)
	if c.QKNorm {
		q = g.ScaledRMSNorm(q, a+"norm_q.weight", 1e-5)
		k = g.ScaledRMSNorm(k, a+"norm_k.weight", 1e-5)
	}
	o := linearAttention(g, g.Unary("relu", q), g.Unary("relu", k), v, c.Heads, n)
	o = g.LinearBias(o, a+"to_out.0.weight", a+"to_out.0.bias", "", E)
	x = g.Add(x, g.Mul(o, mod(2)))
	// cross-attention to the caption (no norm before it), its keys and values Caption's
	a = p + "attn2."
	H, D := c.CrossHeads, c.CrossHeadDim
	q = g.LinearBias(x, a+"to_q.weight", a+"to_q.bias", "", H*D)
	if c.QKNorm {
		q = g.ScaledRMSNorm(q, a+"norm_q.weight", 1e-5)
	}
	o = g.QChunkAttention(g.Heads(q, H, D), g.Heads(kk, H, D), g.Heads(vv, H, D), 1/math.Sqrt(float64(D)), cbias, 0)
	x = g.Add(x, g.LinearBias(g.Merge(o), a+"to_out.0.weight", a+"to_out.0.bias", "", E))
	// GLUMBConv over the token grid
	h = modulate(x, 3, 4)
	f := glumb(g, h, p+"ff.", c.MLP, c.GH(), c.GW())
	f = g.Linear(f, p+"ff.conv_point.weight", "", E)
	return g.Add(x, g.Mul(f, mod(5)))
}

// glumb is GLUMBConv up to its pointwise convolution: of h (1, tokens, in), the inverted 1 x 1
// convolution to 2 ch, SiLU, the 3 x 3 depthwise convolution over the gh x gw grid, the halves
// as x * silu(gate): (1, tokens, ch).
func glumb(g *graph, h *mlpackage.Var, p string, ch, gh, gw int) *mlpackage.Var {
	n := gh * gw
	a := g.SiLU(g.LinearBias(h, p+"conv_inverted.weight", p+"conv_inverted.bias", "", 2*ch))
	a = g.Reshape(g.Transpose(a, 0, 2, 1), 1, 2*ch, gh, gw)
	v, gate := gluDepthwise(g, a, p+"conv_depth", ch)
	f := g.Mul(v, g.SiLU(gate))
	return g.Transpose(g.Reshape(f, 1, ch, n), 0, 2, 1)
}

// maxDepthwise is the most channels of a depthwise convolution the M1's Neural Engine takes:
// 5600 and 5760 are placed on the CPU, 4096 is not.
const maxDepthwise = 4096

// gluDepthwise is GLUMBConv's 3 x 3 depthwise convolution (path) of a (1, 2 ch, H, W) as the
// value's half and the gate's, each in chunks of at most maxDepthwise channels.
func gluDepthwise(g *graph, a *mlpackage.Var, path string, ch int) (v, gate *mlpackage.Var) {
	w, b := g.Get(path+".weight", 2*ch, 1, 3, 3), g.Get(path+".bias", 2*ch)
	half := func(j int) *mlpackage.Var {
		n := (ch + maxDepthwise - 1) / maxDepthwise
		var parts []*mlpackage.Var
		for k := 0; k < n; k++ {
			lo, hi := j*ch+k*ch/n, j*ch+(k+1)*ch/n
			parts = append(parts, g.Conv2D(g.Slice(a, 1, lo, hi), fmt.Sprintf("%s.%d", path, lo), w[lo*9:hi*9], b[lo:hi], hi-lo, 3, hi-lo))
		}
		if len(parts) == 1 {
			return parts[0]
		}
		return g.Concat(1, parts...)
	}
	return half(0), half(1)
}

// linearAttention is Sana's ReLU linear attention of q, k (past their ReLU) and v (1, tokens,
// heads x 32): each head's relu(q) (relu(K)^T V) / (relu(q) . sum relu(K)), (1, tokens, heads
// x 32).
func linearAttention(g *graph, q, k, v *mlpackage.Var, heads, n int) *mlpackage.Var {
	qh := g.Transpose(g.Reshape(q, 1, n, heads, 32), 0, 2, 1, 3)                           // (1, H, n, 32)
	kh := g.Transpose(g.Reshape(g.Scale(k, 1.0/DitKeyScale), 1, n, heads, 32), 0, 2, 3, 1) // (1, H, 32, n)
	vh := g.Transpose(g.Reshape(v, 1, n, heads, 32), 0, 2, 1, 3)                           // (1, H, n, 32)
	o := linAttn(g, qh, kh, vh)
	return g.Reshape(g.Transpose(o, 0, 2, 1, 3), 1, n, heads*32)
}

// linAttn is the linear attention of qh (1, H, n, d), kh (1, H, d, n) and vh (1, H, n, d):
// (1, H, n, d). Each query row is divided by its largest value first (MaxScaled).
func linAttn(g *graph, qh, kh, vh *mlpackage.Var) *mlpackage.Var {
	qh = g.MaxScaled(qh, -1)
	kv := g.MatMul(kh, vh, false, false)  // (1, H, d, d)
	ks := g.Reduce("sum", kh, -1)         // (1, H, d, 1)
	num := g.MatMul(qh, kv, false, false) // (1, H, n, d)
	den := g.MatMul(qh, ks, false, false) // (1, H, n, 1)
	// a head whose denominator is 0 has all its numerators 0 (the reference's eps)
	den = g.Op("maximum", map[string]any{"x": den, "y": g.B.ScalarF16(1e-4)}, den.Shape...)
	return g.Div(num, den)
}

// Final is the final layer: inputs x (1, tokens, hidden), the residual stream, and emb (1, 1,
// hidden), the embedded timestep; output (1, tokens,
// latent), the DiT's output channel last.
func Final(c Dit) *Part {
	return part("dit_final", func(g *graph) {
		E, n := c.Hidden, c.Tokens()
		x := g.Input("x", 1, n, E)
		emb := g.Input("emb", 1, 1, E)
		sst := g.Get("dit.scale_shift_table", 2, E)
		shift := g.Add(g.F16s("dit.final_shift", []int{1, 1, E}, sst[:E]), emb)
		scale := g.Add(g.F16s("dit.final_scale", []int{1, 1, E}, sst[E:]), emb)
		h := g.Add(g.Mul(g.ScaledLayerNorm(x, 1e-6), g.AddScalar(scale, 1)), shift)
		g.Output(g.LinearBias(h, "dit.proj_out.weight", "dit.proj_out.bias", "", c.Latent))
	})
}
