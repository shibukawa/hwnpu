package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// DitScale is what the DiT's residual streams (the image's and the text's) are divided by in its
// programs: img_in's output, txt_in's and the gates the host gives are scaled down by it (the norms
// are scale free). Its largest value in float32 was about 9000 (512 x 512, the first step, text
// and image rows, design.md 16.165): 1100 here, with room for other prompts and sizes.
const DitScale = 8

// maxScoreBytes is the largest fp16 score tensor (heads x queries x keys) the image attention makes
// at once. Past about 64 MiB the Neural Engine runs q k^T and its softmax slower per score (M3,
// design.md 16.85), but more chunks take longer to compile: a group of 4 blocks at 1024 x 1024 (M3,
// lut4) in chunks of 128 queries (38 MB) compiled in 59 s and ran in 1.49 s, of 512 (151 MB) in 33
// s and 1.57 s, all 4096 at once in 35 s and 2.2 s (design.md 16.165). The compile is paid again
// whenever the Neural Engine's cache has dropped the program.
const maxScoreBytes = 160 << 20

// QChunk is the image attention's queries a chunk (of the image's rows): halved from qchunk (0: all
// of them) until a chunk's scores over the image's and the text's keys are within maxScoreBytes.
func QChunk(c Dit, qchunk int) int {
	n := c.Tokens()
	if qchunk <= 0 || qchunk > n {
		qchunk = n
	}
	for qchunk > 1 && c.Heads*qchunk*(n+TextRows)*2 > maxScoreBytes {
		qchunk /= 2
	}
	return qchunk
}

// headPerm is the order of a head's features that puts Qwen-Image's RoPE pair (2i, 2i + 1) at
// blocks.ApplyRope's (i, i + D/2) (iron_qwenimage.head_perm): q and k permuted alike, q . k is
// unchanged.
func headPerm(D int) []int {
	p := make([]int, 0, D)
	for i := 0; i < D; i += 2 {
		p = append(p, i)
	}
	for i := 1; i < D; i += 2 {
		p = append(p, i)
	}
	return p
}

// permRows is the weight (heads x D, in) with each head's rows in headPerm's order.
func permRows(w []float32, heads, D, in int) []float32 {
	p := headPerm(D)
	out := make([]float32, len(w))
	for h := 0; h < heads; h++ {
		for i, src := range p {
			copy(out[(h*D+i)*in:(h*D+i+1)*in], w[(h*D+src)*in:(h*D+src+1)*in])
		}
	}
	return out
}

// RopeTables are the cos and sin (rows x HeadDim, in blocks.ApplyRope's halves) of the positions
// pos (rows of frame, height and width): each axis's frequencies theta^(-2k / d), in the order of
// QwenImage21Rope (the frame's, the height's, the width's).
func RopeTables(c Dit, pos [][3]float64) (cos, sin []float32) {
	var freqs []float64
	var axis []int
	for a, d := range c.Axes {
		for k := 0; k < d; k += 2 {
			freqs = append(freqs, 1/math.Pow(c.Theta, float64(k)/float64(d)))
			axis = append(axis, a)
		}
	}
	D, half := c.HeadDim, c.HeadDim/2
	cos, sin = make([]float32, len(pos)*D), make([]float32, len(pos)*D)
	for r, ps := range pos {
		for i, f := range freqs {
			a := ps[axis[i]] * f
			cs, sn := float32(math.Cos(a)), float32(math.Sin(a))
			cos[r*D+i], cos[r*D+i+half] = cs, cs
			sin[r*D+i], sin[r*D+i+half] = sn, sn
		}
	}
	return cos, sin
}

// TextPositions are the text's rows' positions: p on all three axes.
func TextPositions(n int) [][3]float64 {
	out := make([][3]float64, n)
	for i := range out {
		out[i] = [3]float64{float64(i), float64(i), float64(i)}
	}
	return out
}

// ImagePositions are the image's rows' positions after a text of textLen rows: the frame at
// textLen, the height and width centred on zero (QwenImage21Rope).
func ImagePositions(c Dit, textLen int) [][3]float64 {
	h, w := c.GH(), c.GW()
	out := make([][3]float64, 0, h*w)
	for y := -(h - h/2); y < h/2; y++ {
		for x := -(w - w/2); x < w/2; x++ {
			out = append(out, [3]float64{float64(textLen), float64(y), float64(x)})
		}
	}
	return out
}

// block is a QwenImage21TransformerBlock's weights' names.
type block struct{ p string }

func blk(i int) block { return block{fmt.Sprintf("dit.transformer_blocks.%d.", i)} }

// qkv is the modulated norm of x (1, n, E) and the attention's q, k (normed, turned) and v, each
// (1, n, H, D).
func (b block) qkv(g *graph, c Dit, x, m1, cos, sin *mlpackage.Var) (q, k, v *mlpackage.Var) {
	E, H, D := c.Hidden, c.Heads, c.HeadDim
	n := x.Shape[1]
	h := g.Mul(g.ScaledLayerNorm(x, c.Eps), m1)
	p := b.p + "attn."
	proj := func(name, norm string) *mlpackage.Var {
		var y *mlpackage.Var
		if norm == "" {
			y = g.Linear(h, p+name+".weight", "", E)
		} else {
			w := permRows(g.Get(p+name+".weight", E, E), H, D, E)
			y = g.Op("linear", map[string]any{"x": h, "weight": g.Matrix(p+name+".weight", w, E, E)}, 1, n, E)
		}
		y = g.Reshape(y, 1, n, H, D)
		if norm == "" {
			return y
		}
		nw := g.Get(p+norm+".weight", D)
		pw := make([]float32, D)
		for i, src := range headPerm(D) {
			pw[i] = nw[src]
		}
		y = g.RMSNormWith(g.MaxScaled(y, -1), p+norm+".weight", pw, c.Eps, 1)
		return g.ApplyRope(y, cos, sin)
	}
	return proj("to_q", "norm_q"), proj("to_k", "norm_k"), proj("to_v", "")
}

// rest is the attention's output projection o (1, n, E) gated into x, and the SwiGLU MLP's.
func (b block) rest(g *graph, c Dit, x, o, g1, m2, g2 *mlpackage.Var) *mlpackage.Var {
	E := c.Hidden
	x = g.Add(x, g.Mul(g.Linear(o, b.p+"attn.to_out.0.weight", "", E), g1))
	h := g.Mul(g.ScaledLayerNorm(x, c.Eps), m2)
	p := b.p + "img_mlp."
	f := g.Mul(g.SiLU(g.Linear(h, p+"gate_layer.weight", "", c.MLP)), g.Linear(h, p+"proj.weight", "", c.MLP))
	return g.Add(x, g.Mul(g.Linear(f, p+"out.weight", "", E), g2))
}

// modRows are the shared modulation's four rows of mod (1, 1, 4 E): 1 + scale1, tanh gate1 /
// DitScale, 1 + scale2, tanh gate2 / DitScale (the host's, Modulation).
func modRows(g *graph, mod *mlpackage.Var, E int) (m1, g1, m2, g2 *mlpackage.Var) {
	r := func(i int) *mlpackage.Var { return g.Slice(mod, -1, i*E, (i+1)*E) }
	return r(0), r(1), r(2), r(3)
}

// Blocks is the DiT's blocks first..first+count-1 over the image's rows: inputs x (1, tokens,
// hidden), the residual stream divided by DitScale (the first group's: the latents channel last
// (1, tokens, 64), which it embeds); mod (1, 1, 4 hidden), the step's modulation rows; cos and sin
// (1, tokens, 1, head_dim), the image rows' RoPE (their frame is the text's length); tk and tv (1,
// count x TextRows, hidden), the text's keys (after their norm and RoPE) and values of these
// blocks (Text's outputs); kbias (1, 1, 1, tokens + TextRows), 0 for the image's keys and the
// text's that count, -10000 past them. The attention runs qchunk queries at a time (0: all; QChunk
// is the default). The last group (final) also takes scale (1, 1, hidden),
// norm_out's 1 + scale, and gives the velocity (1, tokens, 64); the others the residual stream
// after them.
func Blocks(c Dit, first, count, qchunk int, final bool) *Part {
	return part(fmt.Sprintf("dit_blocks_%d", first), func(g *graph) {
		n, E, H, D := c.Tokens(), c.Hidden, c.Heads, c.HeadDim
		var x *mlpackage.Var
		if first == 0 {
			x = g.LinearS(g.Input("x", 1, n, c.Latent), "dit.img_in.weight", "", E, 1.0/DitScale)
		} else {
			x = g.Input("x", 1, n, E)
		}
		mod := g.Input("mod", 1, 1, 4*E)
		cos, sin := g.Input("cos", 1, n, 1, D), g.Input("sin", 1, n, 1, D)
		tk := g.Input("tk", 1, count*TextRows, E)
		tv := g.Input("tv", 1, count*TextRows, E)
		kbias := g.Input("kbias", 1, 1, 1, n+TextRows)
		var scale *mlpackage.Var
		if final {
			scale = g.Input("scale", 1, 1, E)
		}
		m1, g1, m2, g2 := modRows(g, mod, E)
		for j := 0; j < count; j++ {
			b := blk(first + j)
			q, k, v := b.qkv(g, c, x, m1, cos, sin)
			text := func(t *mlpackage.Var) *mlpackage.Var {
				return g.Reshape(g.Slice(t, 1, j*TextRows, (j+1)*TextRows), 1, TextRows, H, D)
			}
			heads := func(y *mlpackage.Var) *mlpackage.Var { return g.Transpose(y, 0, 2, 1, 3) }
			keys := heads(g.Concat(1, k, text(tk)))
			vals := heads(g.Concat(1, v, text(tv)))
			o := g.QChunkAttention(heads(q), keys, vals, 1/math.Sqrt(float64(D)), kbias, qchunk)
			x = b.rest(g, c, x, g.Merge(o), g1, m2, g2)
		}
		if final {
			h := g.Mul(g.ScaledLayerNorm(x, c.Eps), scale)
			x = g.Linear(h, "dit.proj_out.weight", "", c.Latent)
		}
		g.Output(x)
	})
}

// TextBlocks is the DiT's blocks first..first+count-1 over the text's rows at t = 0 (diffusers'
// causal_condition: their keys and values are the same at every step): inputs x (1, TextRows,
// hidden), the text's residual stream divided by DitScale (the first group's: TextIn's output),
// mod (1, 1, 4 hidden), t = 0's modulation rows. The text is causal (its padding, after it, is
// never a key of a row that counts). Output (1, (1 + 2 count) TextRows, hidden): the rows after
// the blocks, then each block's keys (after their norm and RoPE) and values.
func TextBlocks(c Dit, first, count int) *Part {
	return part(fmt.Sprintf("dit_text_%d", first), func(g *graph) {
		E, D := c.Hidden, c.HeadDim
		x := g.Input("x", 1, TextRows, E)
		mod := g.Input("mod", 1, 1, 4*E)
		ct, st := RopeTables(c, TextPositions(TextRows))
		cos := g.F16s("text_rope_cos", []int{1, TextRows, 1, D}, ct)
		sin := g.F16s("text_rope_sin", []int{1, TextRows, 1, D}, st)
		causal := g.CausalMask("text_causal", TextRows, 0)
		m1, g1, m2, g2 := modRows(g, mod, E)
		outs := []*mlpackage.Var{nil}
		for j := 0; j < count; j++ {
			b := blk(first + j)
			q, k, v := b.qkv(g, c, x, m1, cos, sin)
			heads := func(y *mlpackage.Var) *mlpackage.Var { return g.Transpose(y, 0, 2, 1, 3) }
			o := g.QChunkAttention(heads(q), heads(k), heads(v), 1/math.Sqrt(float64(D)), causal, 0)
			outs = append(outs, g.Reshape(k, 1, TextRows, E), g.Reshape(v, 1, TextRows, E))
			x = b.rest(g, c, x, g.Merge(o), g1, m2, g2)
		}
		outs[0] = x
		g.Output(g.Concat(1, outs...))
	})
}
