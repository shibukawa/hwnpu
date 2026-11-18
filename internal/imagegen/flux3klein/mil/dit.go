package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// DitScale divides the image's residual stream in fp16, as FLUX.2 klein's: x_embedders.image and
// every block's output projections carry it; the norms do not see it.
const DitScale = 32

// TextStreamScale divides the text's residual stream: it reaches 1.2e5 out of context_embedder and
// 9.7e6 in the last single blocks (a few rows' channels; the image's 2900), past fp16's 65504 at
// DitScale. A block's output, made at DitScale, is added to the text's rows with its gate times
// DitScale / TextStreamScale.
const TextStreamScale = 1024

// HeadChunk is the queries the attention takes at a time, a head at a time (milpart.HeadAttention).
const HeadChunk = 1024

// Position is a row's position ids (t, h, w, l) for the rotary embedding.
type Position [4]float64

// TextPositions are the text's rows' ids: (0, 0, 0, i) (Flux3TextRoPEInputsStep).
func (c Dit) TextIDs() []Position {
	out := make([]Position, c.TextRows)
	for i := range out {
		out[i] = Position{0, 0, 0, float64(i)}
	}
	return out
}

// ImageIDs are the image's rows' ids: (0, row / 2, column / 2, 0) (ImageRopeScale).
func (c Dit) ImageIDs() []Position {
	out := make([]Position, c.Tokens())
	for r := range out {
		out[r] = Position{0, float64(r/c.GW()) * ImageRopeScale, float64(r%c.GW()) * ImageRopeScale, 0}
	}
	return out
}

// RopeValues is the rotary embedding of rows at ids as blocks.ApplyRope takes it in RopePerm's
// order: (rows, head_dim) each of cos and sin, pair i (values i and i + d/2) at frequency i of the
// axes' (four of 32, theta 10000: Flux3PosEmbed).
func RopeValues(c Dit, ids []Position) (cos, sin []float32) {
	D := c.HeadDim
	half := D / 2
	type freq struct {
		axis int
		f    float64
	}
	var fs []freq
	for a, dim := range c.Axes {
		for k := 0; k < dim/2; k++ {
			fs = append(fs, freq{a, 1 / math.Pow(c.Theta, float64(2*k)/float64(dim))})
		}
	}
	cos, sin = make([]float32, len(ids)*D), make([]float32, len(ids)*D)
	for r, pos := range ids {
		for i, f := range fs {
			a := pos[f.axis] * f.f
			cs, sn := float32(math.Cos(a)), float32(math.Sin(a))
			cos[r*D+i], cos[r*D+i+half] = cs, cs
			sin[r*D+i], sin[r*D+i+half] = sn, sn
		}
	}
	return cos, sin
}

// ditGraph holds what the blocks of a program share.
type ditGraph struct {
	*graph
	c        Dit
	cos, sin *mlpackage.Var // (1, rows, 1, head_dim) of the program's rows
	m        *mlpackage.Var // (1, mods, hidden): the program's modulation rows
}

func newDitGraph(g *graph, c Dit, name string, ids []Position, mods int) *ditGraph {
	d := &ditGraph{graph: g, c: c}
	cs, sn := RopeValues(c, ids)
	R := len(ids)
	d.cos = g.F16s(fmt.Sprintf("rope_cos_%s_%d", name, R), []int{1, R, 1, c.HeadDim}, cs)
	d.sin = g.F16s(fmt.Sprintf("rope_sin_%s_%d", name, R), []int{1, R, 1, c.HeadDim}, sn)
	d.m = g.Input("mods", 1, mods, c.Hidden)
	return d
}

// mod is modulation row r of the program's (1, 1, hidden).
func (d *ditGraph) mod(r int) *mlpackage.Var { return d.Slice(d.m, 1, r, r+1) }

// modulate is the norm of x (no weight, eps) times 1 + scale, plus shift.
func (d *ditGraph) modulate(x, shift, scale *mlpackage.Var) *mlpackage.Var {
	return d.Add(d.Mul(d.AbsScaledLayerNorm(x, d.c.Eps), d.AddScalar(scale, 1)), shift)
}

// qkv is a block's q and k (normed, turned by the program's rotary embedding) and v of h's rows,
// each (1, rows, heads, head_dim). q holds the attention's scale, 1 / sqrt(head_dim), in its norm's
// weight: the attention then runs at scale 1.
func (d *ditGraph) qkv(p string, h *mlpackage.Var) (q, k, v *mlpackage.Var) {
	c := d.c
	n, H, D := h.Shape[1], c.Heads, c.HeadDim
	E := H * D
	head := func(name, norm string, sc float32) *mlpackage.Var {
		y := d.Reshape(d.Linear(h, p+name+".weight", "", E), 1, n, H, D)
		if norm != "" {
			w := append([]float32(nil), d.Get(p+norm+".weight", D)...)
			for i := range w {
				w[i] *= sc
			}
			y = d.ApplyRope(d.RMSNormWith(d.AbsScaled(y, -1), p+norm+".weight", w, c.Eps, 1), d.cos, d.sin)
		}
		return y
	}
	return head("attn.to_q", "attn.norm_q", float32(1/math.Sqrt(float64(D)))), head("attn.to_k", "attn.norm_k", 1), head("attn.to_v", "", 1)
}

// attend is the attention of q, k, v (1, rows, heads, head_dim) over all the program's rows:
// (1, rows, hidden), a head at a time, HeadChunk queries at a time; split, when not 0, cuts the
// keys' scores in two tensors there (the text's and the image's).
func (d *ditGraph) attend(q, k, v *mlpackage.Var, split int) *mlpackage.Var {
	t := func(x *mlpackage.Var) *mlpackage.Var { return d.Transpose(x, 0, 2, 1, 3) }
	return d.Merge(d.HeadAttention(t(q), t(k), t(v), milpart.HeadOptions{Chunk: HeadChunk, Split: split}))
}

// parallel is a block's attention and feedforward of the same modulated rows h, summed (each
// divided by DitScale): Flux3ModeTransformerBlock's and Flux3SingleTransformerBlock's
// attn_output + ff_output.
func (d *ditGraph) parallel(p string, h *mlpackage.Var, split int) *mlpackage.Var {
	c := d.c
	q, k, v := d.qkv(p, h)
	o := d.LinearS(d.attend(q, k, v, split), p+"attn.to_out.0.weight", "", c.Hidden, 1.0/DitScale)
	ff := d.SwiGLU(h, p+"ff.linear_in.weight#gate", p+"ff.linear_in.weight#up", p+"ff.linear_out.weight", c.MLP, c.Hidden, 1.0/DitScale)
	return d.Add(o, ff)
}

// gate is modulation row r as the gate of a block's output (made at DitScale) into a stream held
// divided by scale.
func (d *ditGraph) gate(r int, scale float32) *mlpackage.Var {
	if scale == DitScale {
		return d.mod(r)
	}
	return d.Scale(d.mod(r), DitScale/scale)
}

// mode is Flux3ModeTransformerBlock p over x (a stream held divided by scale), its modulation rows
// (shift, scale, gate) the program's from r: x + gate (attention + feedforward) of the modulated
// rows, the stream attending to itself.
func (d *ditGraph) mode(p string, x *mlpackage.Var, r int, scale float32) *mlpackage.Var {
	h := d.modulate(x, d.mod(r), d.mod(r+1))
	return d.Add(x, d.Mul(d.parallel(p, h, 0), d.gate(r+2, scale)))
}

// single is Flux3SingleTransformerBlock i over x (the text's rows divided by TextStreamScale, then
// the image's by DitScale): each stream's rows modulated by its own rows (the text's from 0, the
// image's from 3), the attention and feedforward over them together, each stream's output gated by
// its own.
func (d *ditGraph) single(i int, x *mlpackage.Var) *mlpackage.Var {
	c := d.c
	L, R := c.TextRows, c.Rows()
	p := fmt.Sprintf("dit.single_transformer_blocks.%d.", i)
	xt, xi := d.Slice(x, 1, 0, L), d.Slice(x, 1, L, R)
	h := d.Concat(1, d.modulate(xt, d.mod(0), d.mod(1)), d.modulate(xi, d.mod(3), d.mod(4)))
	o := d.parallel(p, h, L)
	yt := d.Add(xt, d.Mul(d.Slice(o, 1, 0, L), d.gate(2, TextStreamScale)))
	yi := d.Add(xi, d.Mul(d.Slice(o, 1, L, R), d.gate(5, DitScale)))
	return d.Concat(1, yt, yi)
}

// TextModes is the text's mode blocks first..first+count-1 (txt_mode_blocks) over the text's rows:
// input x (1, text rows, hidden), Ctx's output or the group before's, divided by TextStreamScale; mods
// (1, 3, hidden) the text's early modulation rows (shift, scale, gate). Output the text's stream.
func TextModes(c Dit, first, count int) *Part {
	return part(fmt.Sprintf("dit_text_%d", first), func(g *graph) {
		x := g.Input("x", 1, c.TextRows, c.Hidden)
		d := newDitGraph(g, c, "text", c.TextIDs(), 3)
		for j := first; j < first+count; j++ {
			x = d.mode(fmt.Sprintf("dit.txt_mode_blocks.%d.", j), x, 0, TextStreamScale)
		}
		g.Output(x)
	})
}

// ImageModes is the image's mode blocks first..first+count-1 (mode_blocks.image) over its rows:
// input x (1, tokens, hidden) divided by DitScale, for the first group the latents' patches (1,
// tokens, 128), which it embeds (x_embedders.image); mods (1, 3, hidden) the image's early
// modulation rows. Output the image's stream.
func ImageModes(c Dit, first, count int) *Part {
	return part(fmt.Sprintf("dit_image_%d", first), func(g *graph) {
		N, E := c.Tokens(), c.Hidden
		var x *mlpackage.Var
		if first == 0 {
			x = g.LinearS(g.Input("x", 1, N, c.Latent), "dit.x_embedders.image.weight", "", E, 1.0/DitScale)
		} else {
			x = g.Input("x", 1, N, E)
		}
		d := newDitGraph(g, c, "image", c.ImageIDs(), 3)
		for j := first; j < first+count; j++ {
			x = d.mode(fmt.Sprintf("dit.mode_blocks.image.%d.", j), x, 0, DitScale)
		}
		g.Output(x)
	})
}

// Singles is the single blocks first..first+count-1 over x (1, text rows + tokens, hidden): the
// text's stream (TextModes' output, divided by TextStreamScale), then the image's (ImageModes',
// divided by DitScale); mods
// (1, 6, hidden) the single blocks' text modulation rows, then their image's. The last group ends
// with norm_out and proj_out of the image's rows, its mods 2 rows more (norm_out's scale and shift):
// output (1, tokens, 128), the velocity of the patches. Else the output is the streams.
func Singles(c Dit, first, count int) *Part {
	return part(fmt.Sprintf("dit_single_%d", first), func(g *graph) {
		last := first+count == c.Singles
		mods := 6
		if last {
			mods += 2
		}
		x := g.Input("x", 1, c.Rows(), c.Hidden)
		ids := append(c.TextIDs(), c.ImageIDs()...)
		d := newDitGraph(g, c, "joint", ids, mods)
		for j := first; j < first+count; j++ {
			x = d.single(j, x)
		}
		if last {
			h := d.modulate(g.Slice(x, 1, c.TextRows, c.Rows()), d.mod(7), d.mod(6))
			x = g.Linear(h, "dit.proj_out.weight", "", c.Latent)
		}
		g.Output(x)
	})
}

// Ctx is context_embedder of the text's rows: inputs x0 and x1 (1, text rows, 4 x Qwen3-VL's
// width), the hidden states after TextTaps' first four layers and after their last four, side by
// side, divided by TextScale (20480 in-features are past the Neural Engine's 16384 a dimension:
// two linears of its halves of columns, summed); output (1, text rows, hidden) divided by
// TextStreamScale (TextModes' x).
func Ctx(c Dit) *Part {
	return part("ctx", func(g *graph) {
		half := c.Context / 2
		x0 := g.Input("x0", 1, c.TextRows, half)
		x1 := g.Input("x1", 1, c.TextRows, half)
		s := float32(TextScale) / TextStreamScale
		g.Output(g.Add(g.LinearS(x0, "dit.context_embedder.weight#c0", "", c.Hidden, s),
			g.LinearS(x1, "dit.context_embedder.weight#c1", "", c.Hidden, s)))
	})
}
