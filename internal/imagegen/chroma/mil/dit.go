package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// linearW is a linear of x with the weight w (out, in) and the bias b (nil: none), both times s
// (folded into them); name, the weight's, picks its format (llm.Options.FormatOf).
func linearW(g *graph, x *mlpackage.Var, name string, w []float32, out int, b []float32, s float32) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	if s != 1 {
		for i := range w {
			w[i] *= s
		}
	}
	inputs := map[string]any{"x": x, "weight": g.Matrix(name, w, out, in)}
	if b != nil {
		bs := make([]float32, len(b))
		for i, v := range b {
			bs[i] = v * s
		}
		inputs["bias"] = g.F16s(name+".bias", []int{out}, bs)
	}
	shape := append(append([]int{}, x.Shape[:len(x.Shape)-1]...), out)
	return g.Op("linear", inputs, shape...)
}

// linearBiasS is the named linear (prefix.weight, prefix.bias) of x times s.
func linearBiasS(g *graph, x *mlpackage.Var, prefix string, out int, s float32) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	return linearW(g, x, prefix+".weight", g.Get(prefix+".weight", out, in), out, g.Get(prefix+".bias", out), s)
}

// Mods is the approximator (distilled_guidance_layer): input x (1, Mods, 64), each row the
// timestep's sinusoid (16), a zero guidance's (16) and the row's own index's (32: of 1000 k), as
// ChromaCombinedTimestepTextProjEmbeddings makes them (the host does); output (1, Mods, hidden),
// the modulation rows. It depends on the timestep alone: the host runs it once a sigma.
func Mods(c Dit) *Part {
	return part("mods", func(g *graph) {
		p := "dit.distilled_guidance_layer."
		x := g.Input("x", 1, c.Mods(), c.ApproxIn)
		h := g.LinearBias(x, p+"in_proj.weight", p+"in_proj.bias", "", c.ApproxHidden)
		for i := 0; i < c.ApproxLayers; i++ {
			// nn.RMSNorm without eps: float32's machine epsilon
			t := g.ScaledRMSNorm(h, fmt.Sprintf("%snorms.%d.weight", p, i), 1.1920929e-07)
			l := fmt.Sprintf("%slayers.%d.", p, i)
			t = g.SiLU(g.LinearBias(t, l+"linear_1.weight", l+"linear_1.bias", "", c.ApproxHidden))
			h = g.Add(h, g.LinearBias(t, l+"linear_2.weight", l+"linear_2.bias", "", c.ApproxHidden))
		}
		g.Output(g.LinearBias(h, p+"out_proj.weight", p+"out_proj.bias", "", c.Hidden))
	})
}

// RopeValues is the rotary embedding of the DiT's rows (the image's at (0, row, column), then the
// text's at (0, 0, 0)) as blocks.ApplyRope takes it in RopePerm's order: (rows, head_dim) each of
// cos and sin, pair i (values i and i + d/2) at frequency i of FLUX's axes (16, 56, 56).
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
			fs = append(fs, freq{a, 1 / math.Pow(10000, float64(2*k)/float64(dim))})
		}
	}
	R := c.Rows()
	cos, sin = make([]float32, R*D), make([]float32, R*D)
	for r := 0; r < R; r++ {
		pos := [3]float64{}
		if r < c.Tokens() {
			pos[1], pos[2] = float64(r/c.GW()), float64(r%c.GW())
		}
		for i, f := range fs {
			// float32 as diffusers' get_1d_rotary_pos_embed (freqs in float64, then cast)
			a := pos[f.axis] * f.f
			cs, sn := float32(math.Cos(a)), float32(math.Sin(a))
			cos[r*D+i], cos[r*D+i+half] = cs, cs
			sin[r*D+i], sin[r*D+i+half] = sn, sn
		}
	}
	return cos, sin
}

// maxScoreBytes is the largest fp16 score tensor (heads x queries x keys) an attention should
// make at once: past about 64 MiB the Neural Engine runs q k^T and its softmax 2 to 3 times
// slower per score (design.md 16.85).
const maxScoreBytes = 64 << 20

// QChunk is the queries an attention over the DiT's rows takes at a time: halved from all of them
// until the scores of a chunk are within maxScoreBytes, then up to a multiple of 64.
func QChunk(c Dit) int {
	R := c.Rows()
	q := R
	for q > 64 && c.Heads*q*R*2 > maxScoreBytes {
		q = (q + 1) / 2
	}
	return q
}

// stream is a block's projections of one stream's rows.
type stream struct {
	q, k, v, qn, kn string // q, k and v's weight and bias prefix; the q and k norms' weights
	o               string // the attention's output projection (double blocks)
	ff1, ff2        string // the MLP (double blocks)
}

func imageStream(p string) stream {
	return stream{p + "attn.to_q", p + "attn.to_k", p + "attn.to_v", p + "attn.norm_q.weight", p + "attn.norm_k.weight",
		p + "attn.to_out.0", p + "ff.net.0.proj", p + "ff.net.2"}
}

func textStream(p string) stream {
	return stream{p + "attn.add_q_proj", p + "attn.add_k_proj", p + "attn.add_v_proj", p + "attn.norm_added_q.weight",
		p + "attn.norm_added_k.weight", p + "attn.to_add_out", p + "ff_context.net.0.proj", p + "ff_context.net.2"}
}

// ditGraph holds what the blocks of a program share.
type ditGraph struct {
	*graph
	c        Dit
	cos, sin *mlpackage.Var // (1, rows, 1, head_dim)
	kbias    *mlpackage.Var // (1, 1, 1, rows)
	m        *mlpackage.Var // (1, mods, hidden): the program's modulation rows
	qchunk   int
}

func newDitGraph(g *graph, c Dit, mods int) *ditGraph {
	d := &ditGraph{graph: g, c: c, qchunk: QChunk(c)}
	cs, sn := RopeValues(c)
	R := c.Rows()
	d.cos = g.F16s(fmt.Sprintf("dit_rope_cos_%d", R), []int{1, R, 1, c.HeadDim}, cs)
	d.sin = g.F16s(fmt.Sprintf("dit_rope_sin_%d", R), []int{1, R, 1, c.HeadDim}, sn)
	d.m = g.Input("mods", 1, mods, c.Hidden)
	d.kbias = g.Input("kbias", 1, 1, 1, R)
	return d
}

// mod is modulation row r of the program's (1, 1, hidden).
func (d *ditGraph) mod(r int) *mlpackage.Var { return d.Slice(d.m, 1, r, r+1) }

// modulate is the norm of x (no weight, eps 1e-6) times 1 + scale, plus shift.
func (d *ditGraph) modulate(x, shift, scale *mlpackage.Var) *mlpackage.Var {
	return d.Add(d.Mul(d.ScaledLayerNorm(x, 1e-6), d.AddScalar(scale, 1)), shift)
}

// qkv is a stream's q and k (normed, turned by the rows' rotary embedding from row lo) and v of
// h's rows, each (1, n, heads, head_dim).
func (d *ditGraph) qkv(s stream, h *mlpackage.Var, lo int) (q, k, v *mlpackage.Var) {
	c := d.c
	n, H, D := h.Shape[1], c.Heads, c.HeadDim
	E := H * D
	cos, sin := d.Slice(d.cos, 1, lo, lo+n), d.Slice(d.sin, 1, lo, lo+n)
	if lo == 0 && n == c.Rows() {
		cos, sin = d.cos, d.sin
	}
	head := func(name, norm string) *mlpackage.Var {
		y := d.Reshape(d.LinearBias(h, name+".weight", name+".bias", "", E), 1, n, H, D)
		if norm != "" {
			y = d.ApplyRope(d.ScaledRMSNorm(y, norm, 1e-6), cos, sin)
		}
		return y
	}
	return head(s.q, s.qn), head(s.k, s.kn), head(s.v, "")
}

// attend is the attention of q, k, v (1, rows, heads, head_dim) over the keys kbias keeps:
// (1, rows, hidden).
func (d *ditGraph) attend(q, k, v *mlpackage.Var) *mlpackage.Var {
	t := func(x *mlpackage.Var) *mlpackage.Var { return d.Transpose(x, 0, 2, 1, 3) }
	o := d.QChunkAttention(t(q), t(k), t(v), 1/math.Sqrt(float64(d.c.HeadDim)), d.kbias, d.qchunk)
	return d.Merge(o)
}

// gelu is the MLP's tanh GELU.
func (d *ditGraph) gelu(x *mlpackage.Var) *mlpackage.Var { return GeluTanh(d.graph, x) }

// double is ChromaTransformerBlock i over x (1, rows, hidden): the image's rows, then the text's.
// Its modulation rows are the program's from r: the image's 6, then the text's 6.
func (d *ditGraph) double(i int, x *mlpackage.Var, r int) *mlpackage.Var {
	c := d.c
	N, R, E := c.Tokens(), c.Rows(), c.Hidden
	p := fmt.Sprintf("dit.transformer_blocks.%d.", i)
	parts := []struct {
		s      stream
		lo, hi int
		r      int
	}{{imageStream(p), 0, N, r}, {textStream(p), N, R, r + 6}}
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
		y := d.Add(xs[j], d.Mul(linearBiasS(d.graph, d.Slice(o, 1, pt.lo, pt.hi), pt.s.o, E, 1.0/DitScale), d.mod(pt.r+2)))
		h := d.modulate(y, d.mod(pt.r+3), d.mod(pt.r+4))
		h = d.gelu(d.LinearBias(h, pt.s.ff1+".weight", pt.s.ff1+".bias", "", c.MLP))
		y = d.Add(y, d.Mul(linearBiasS(d.graph, h, pt.s.ff2, E, 1.0/DitScale), d.mod(pt.r+5)))
		outs = append(outs, y)
	}
	return d.Concat(1, outs...)
}

// single is ChromaSingleTransformerBlock i over x; its modulation rows (shift, scale, gate) are
// the program's from r.
func (d *ditGraph) single(i int, x *mlpackage.Var, r int) *mlpackage.Var {
	c := d.c
	E := c.Hidden
	p := fmt.Sprintf("dit.single_transformer_blocks.%d.", i)
	h := d.modulate(x, d.mod(r), d.mod(r+1))
	mlp := d.gelu(d.LinearBias(h, p+"proj_mlp.weight", p+"proj_mlp.bias", "", c.MLP))
	q, k, v := d.qkv(stream{q: p + "attn.to_q", k: p + "attn.to_k", v: p + "attn.to_v", qn: p + "attn.norm_q.weight", kn: p + "attn.norm_k.weight"}, h, 0)
	o := d.attend(q, k, v)
	// proj_out over [attention, MLP] as two linears
	po := d.Get(p+"proj_out.weight", E, E+c.MLP)
	wa, wm := make([]float32, E*E), make([]float32, E*c.MLP)
	for j := 0; j < E; j++ {
		copy(wa[j*E:(j+1)*E], po[j*(E+c.MLP):])
		copy(wm[j*c.MLP:(j+1)*c.MLP], po[j*(E+c.MLP)+E:])
	}
	ya := linearW(d.graph, o, p+"proj_out.weight.attn", wa, E, nil, 1.0/DitScale)
	y := d.Add(ya, linearW(d.graph, mlp, p+"proj_out.weight.mlp", wm, E, d.Get(p+"proj_out.bias", E), 1.0/DitScale))
	return d.Add(x, d.Mul(y, d.mod(r+2)))
}

// DoubleGroups and SingleGroups split the DiT's blocks into programs of at most per blocks each,
// as even as they go (19 by 4: 4, 4, 4, 4, 3). At 8 bits a double block is 226 MB, a single
// block 113 MB; a program past about 1 GB of weights is placed on the CPU.
func DoubleGroups(c Dit, per int) [][2]int { return TextGroups(c.Doubles, per) }

// SingleGroups is DoubleGroups of the single blocks.
func SingleGroups(c Dit, per int) [][2]int { return TextGroups(c.Singles, per) }

// Doubles is the double blocks first..first+count-1: input x (1, rows, hidden), the residual
// streams (the image's rows, then the text's) divided by DitScale; for the first group x is the
// latents' patches (1, tokens, 64), which it embeds, and ctx (1, text rows, hidden) Text's
// output. mods (1, 12 count, hidden): each block's image modulation rows, then its text's; kbias
// (1, 1, 1, rows): 0 for the keys attended to (the image's, the prompt's and one pad), -10000 for
// the others. Output y, the residual streams after them.
func Doubles(c Dit, first, count int) *Part {
	return part(fmt.Sprintf("dit_double_%d", first), func(g *graph) {
		N, E := c.Tokens(), c.Hidden
		var x *mlpackage.Var
		if first == 0 {
			p := g.Input("x", 1, N, c.Latent)
			ctx := g.Input("ctx", 1, c.TextRows, E)
			x = g.Concat(1, linearBiasS(g, p, "dit.x_embedder", E, 1.0/DitScale), ctx)
		} else {
			x = g.Input("x", 1, c.Rows(), E)
		}
		d := newDitGraph(g, c, 12*count)
		for j := 0; j < count; j++ {
			x = d.double(first+j, x, 12*j)
		}
		g.Output(x)
	})
}

// Singles is the single blocks first..first+count-1 over x as Doubles' output; mods (1, 3 count,
// hidden) their shift, scale and gate rows. The last group ends with norm_out and proj_out of the
// image's rows, its mods 2 rows more (the final norm's shift and scale): output (1, tokens, 64),
// the velocity of the patches. Else the output is the residual streams.
func Singles(c Dit, first, count int) *Part {
	return part(fmt.Sprintf("dit_single_%d", first), func(g *graph) {
		last := first+count == c.Singles
		mods := 3 * count
		if last {
			mods += 2
		}
		x := g.Input("x", 1, c.Rows(), c.Hidden)
		d := newDitGraph(g, c, mods)
		for j := 0; j < count; j++ {
			x = d.single(first+j, x, 3*j)
		}
		if last {
			h := d.modulate(g.Slice(x, 1, 0, c.Tokens()), d.mod(3*count), d.mod(3*count+1))
			x = g.LinearBias(h, "dit.proj_out.weight", "dit.proj_out.bias", "", c.Latent)
		}
		g.Output(x)
	})
}
