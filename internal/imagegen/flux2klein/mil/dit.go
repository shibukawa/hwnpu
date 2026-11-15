package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// DitScale divides the DiT's residual streams (FLUX's reach past fp16 in the later single blocks:
// diffusers clips them to 65504 in fp16). x_embedder, context_embedder and every block's output
// projections carry it; the norms do not see it.
const DitScale = 32

// RopeValues is the rotary embedding of the DiT's rows (the image's at (0, row, column, 0), then
// the text's at (0, 0, 0, position)) as blocks.ApplyRope takes it in RopePerm's order: (rows,
// head_dim) each of cos and sin, pair i (values i and i + d/2) at frequency i of the axes' (FLUX.2:
// four of 32, theta 2000).
func RopeValues(c Dit) (cos, sin []float32) {
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
	R := c.Rows()
	cos, sin = make([]float32, R*D), make([]float32, R*D)
	for r := 0; r < R; r++ {
		pos := make([]float64, len(c.Axes))
		if r < c.Tokens() {
			pos[1], pos[2] = float64(r/c.GW()), float64(r%c.GW())
		} else {
			pos[3] = float64(r - c.Tokens())
		}
		for i, f := range fs {
			a := pos[f.axis] * f.f
			cs, sn := float32(math.Cos(a)), float32(math.Sin(a))
			cos[r*D+i], cos[r*D+i+half] = cs, cs
			sin[r*D+i], sin[r*D+i+half] = sn, sn
		}
	}
	return cos, sin
}

// HeadChunk is the queries the attention takes at a time, a head at a time
// (milpart.HeadAttention), as Chroma's of the same rows (design.md 16.211).
const HeadChunk = 1024

// stream is a block's projections of one stream's rows.
type stream struct {
	q, k, v, qn, kn string // q, k and v's weights; the q and k norms' weights
	o               string // the attention's output projection (double blocks)
	gate, up, down  string // the SwiGLU MLP (double blocks)
}

func imageStream(p string) stream {
	return stream{p + "attn.to_q.weight", p + "attn.to_k.weight", p + "attn.to_v.weight", p + "attn.norm_q.weight", p + "attn.norm_k.weight",
		p + "attn.to_out.0.weight", p + "ff.linear_in.weight#gate", p + "ff.linear_in.weight#up", p + "ff.linear_out.weight"}
}

func textStream(p string) stream {
	return stream{p + "attn.add_q_proj.weight", p + "attn.add_k_proj.weight", p + "attn.add_v_proj.weight", p + "attn.norm_added_q.weight",
		p + "attn.norm_added_k.weight", p + "attn.to_add_out.weight", p + "ff_context.linear_in.weight#gate",
		p + "ff_context.linear_in.weight#up", p + "ff_context.linear_out.weight"}
}

// ditGraph holds what the blocks of a program share.
type ditGraph struct {
	*graph
	c        Dit
	cos, sin *mlpackage.Var // (1, rows, 1, head_dim)
	m        *mlpackage.Var // (1, mods, hidden): the program's modulation rows
}

func newDitGraph(g *graph, c Dit, mods int) *ditGraph {
	d := &ditGraph{graph: g, c: c}
	cs, sn := RopeValues(c)
	R := c.Rows()
	d.cos = g.F16s(fmt.Sprintf("dit_rope_cos_%d", R), []int{1, R, 1, c.HeadDim}, cs)
	d.sin = g.F16s(fmt.Sprintf("dit_rope_sin_%d", R), []int{1, R, 1, c.HeadDim}, sn)
	d.m = g.Input("mods", 1, mods, c.Hidden)
	return d
}

// mod is modulation row r of the program's (1, 1, hidden).
func (d *ditGraph) mod(r int) *mlpackage.Var { return d.Slice(d.m, 1, r, r+1) }

// modulate is the norm of x (no weight, eps) times 1 + scale, plus shift.
func (d *ditGraph) modulate(x, shift, scale *mlpackage.Var) *mlpackage.Var {
	return d.Add(d.Mul(d.AbsScaledLayerNorm(x, d.c.Eps), d.AddScalar(scale, 1)), shift)
}

// qkv is a stream's q and k (normed, turned by the rows' rotary embedding from row lo) and v of
// h's rows, each (1, n, heads, head_dim). q holds the attention's scale, 1 / sqrt(head_dim), in
// its norm's weight: the attention then runs at scale 1 (design.md 16.201).
func (d *ditGraph) qkv(s stream, h *mlpackage.Var, lo int) (q, k, v *mlpackage.Var) {
	c := d.c
	n, H, D := h.Shape[1], c.Heads, c.HeadDim
	E := H * D
	cos, sin := d.Slice(d.cos, 1, lo, lo+n), d.Slice(d.sin, 1, lo, lo+n)
	if lo == 0 && n == c.Rows() {
		cos, sin = d.cos, d.sin
	}
	head := func(name, norm string, sc float32) *mlpackage.Var {
		y := d.Reshape(d.Linear(h, name, "", E), 1, n, H, D)
		if norm != "" {
			w := append([]float32(nil), d.Get(norm, D)...)
			for i := range w {
				w[i] *= sc
			}
			y = d.ApplyRope(d.RMSNormWith(d.AbsScaled(y, -1), norm, w, c.Eps, 1), cos, sin)
		}
		return y
	}
	return head(s.q, s.qn, float32(1/math.Sqrt(float64(D)))), head(s.k, s.kn, 1), head(s.v, "", 1)
}

// attend is the attention of q, k, v (1, rows, heads, head_dim) over all the rows (the DiT attends
// to the text's pads too): (1, rows, hidden). A head at a time, HeadChunk queries at a time, the
// image's keys and the text's as score tensors of their own (design.md 16.211).
func (d *ditGraph) attend(q, k, v *mlpackage.Var) *mlpackage.Var {
	t := func(x *mlpackage.Var) *mlpackage.Var { return d.Transpose(x, 0, 2, 1, 3) }
	return d.Merge(d.HeadAttention(t(q), t(k), t(v), milpart.HeadOptions{Chunk: HeadChunk, Split: d.c.Tokens()}))
}

// swiglu is a SwiGLU MLP's output of h, divided by DitScale.
func (d *ditGraph) swiglu(h *mlpackage.Var, gate, up, down string) *mlpackage.Var {
	return d.SwiGLU(h, gate, up, down, d.c.MLP, d.c.Hidden, 1.0/DitScale)
}

// double is Flux2TransformerBlock i over x (1, rows, hidden): the image's rows, then the text's.
// The modulation rows are the program's: the image's 6 (ModImage), then the text's (ModText).
func (d *ditGraph) double(i int, x *mlpackage.Var) *mlpackage.Var {
	c := d.c
	N, R, E := c.Tokens(), c.Rows(), c.Hidden
	p := fmt.Sprintf("dit.transformer_blocks.%d.", i)
	parts := []struct {
		s      stream
		lo, hi int
		r      int
	}{{imageStream(p), 0, N, ModImage}, {textStream(p), N, R, ModText}}
	var qs, ks, vs, xs []*mlpackage.Var
	for _, pt := range parts {
		xi := d.Slice(x, 1, pt.lo, pt.hi)
		xs = append(xs, xi)
		q, k, v := d.qkv(pt.s, d.modulate(xi, d.mod(pt.r), d.mod(pt.r+1)), pt.lo)
		qs, ks, vs = append(qs, q), append(ks, k), append(vs, v)
	}
	o := d.attend(d.Concat(1, qs...), d.Concat(1, ks...), d.Concat(1, vs...))
	var outs []*mlpackage.Var
	for j, pt := range parts {
		y := d.Add(xs[j], d.Mul(d.LinearS(d.Slice(o, 1, pt.lo, pt.hi), pt.s.o, "", E, 1.0/DitScale), d.mod(pt.r+2)))
		h := d.modulate(y, d.mod(pt.r+3), d.mod(pt.r+4))
		y = d.Add(y, d.Mul(d.swiglu(h, pt.s.gate, pt.s.up, pt.s.down), d.mod(pt.r+5)))
		outs = append(outs, y)
	}
	return d.Concat(1, outs...)
}

// single is Flux2SingleTransformerBlock i over x; its modulation rows (shift, scale, gate) are the
// program's from r.
func (d *ditGraph) single(i int, x *mlpackage.Var, r int) *mlpackage.Var {
	c := d.c
	E := c.Hidden
	p := fmt.Sprintf("dit.single_transformer_blocks.%d.attn.", i)
	fused := p + "to_qkv_mlp_proj.weight#"
	h := d.modulate(x, d.mod(r), d.mod(r+1))
	q, k, v := d.qkv(stream{q: fused + "q", k: fused + "k", v: fused + "v", qn: p + "norm_q.weight", kn: p + "norm_k.weight"}, h, 0)
	o := d.attend(q, k, v)
	gv := d.SiLU(d.Linear(h, fused+"gate", "", c.MLP))
	mlp := d.Mul(gv, d.Linear(h, fused+"up", "", c.MLP))
	// to_out over [attention, MLP] as two linears
	y := d.Add(d.LinearS(o, p+"to_out.weight#attn", "", E, 1.0/DitScale), d.LinearS(mlp, p+"to_out.weight#mlp", "", E, 1.0/DitScale))
	return d.Add(x, d.Mul(y, d.mod(r+2)))
}

// Doubles is the double blocks first..first+count-1: input x (1, rows, hidden), the residual
// streams (the image's rows, then the text's) divided by DitScale; for the first group x is the
// latents' patches (1, tokens, 128), which it embeds, and ctx (1, text rows, hidden) Ctx's
// output. mods (1, 12, hidden): the double blocks' image modulation rows, then their text's (the
// same for every block). Output y, the residual streams after them.
func Doubles(c Dit, first, count int) *Part {
	return part(fmt.Sprintf("dit_double_%d", first), func(g *graph) {
		N, E := c.Tokens(), c.Hidden
		var x *mlpackage.Var
		if first == 0 {
			p := g.Input("x", 1, N, c.Latent)
			ctx := g.Input("ctx", 1, c.TextRows, E)
			x = g.Concat(1, g.LinearS(p, "dit.x_embedder.weight", "", E, 1.0/DitScale), ctx)
		} else {
			x = g.Input("x", 1, c.Rows(), E)
		}
		d := newDitGraph(g, c, ModSingle)
		for j := 0; j < count; j++ {
			x = d.double(first+j, x)
		}
		g.Output(x)
	})
}

// Singles is the single blocks first..first+count-1 over x as Doubles' output; mods (1, 3,
// hidden) their shift, scale and gate rows. The last group ends with norm_out and proj_out of the
// image's rows, its mods 2 rows more (norm_out's scale and shift): output (1, tokens, 128), the
// velocity of the patches. Else the output is the residual streams.
func Singles(c Dit, first, count int) *Part {
	return part(fmt.Sprintf("dit_single_%d", first), func(g *graph) {
		last := first+count == c.Singles
		mods := 3
		if last {
			mods += 2
		}
		x := g.Input("x", 1, c.Rows(), c.Hidden)
		d := newDitGraph(g, c, mods)
		for j := 0; j < count; j++ {
			x = d.single(first+j, x, 0)
		}
		if last {
			h := d.modulate(g.Slice(x, 1, 0, c.Tokens()), d.mod(4), d.mod(3))
			x = g.Linear(h, "dit.proj_out.weight", "", c.Latent)
		}
		g.Output(x)
	})
}

// Ctx is context_embedder of the text's rows: input x (1, text rows, 3 x Qwen3's width), the
// hidden states after TextTaps' layers side by side, divided by TextScale; output (1, text rows,
// hidden) divided by DitScale (Doubles' ctx).
func Ctx(c Dit) *Part {
	return part("ctx", func(g *graph) {
		x := g.Input("x", 1, c.TextRows, c.Context)
		g.Output(g.LinearS(x, "dit.context_embedder.weight", "", c.Hidden, float32(TextScale)/DitScale))
	})
}
