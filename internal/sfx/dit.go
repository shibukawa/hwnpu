package sfx

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// The DiT's shape (transformer/config.json: WanAudioModel, dit_variant 1.3B, vae_type dac).
const (
	DitDim     = 1536
	DitLayers  = 30
	ditHeads   = 12
	ditHeadDim = 128
	ditFFN     = 8960
	FreqDim    = 256
	TextDim    = 2048
	// TextRows is the prompter's text_len: the context the cross-attention reads, the prompt's
	// rows and the rows of zeros after them.
	TextRows = 512
	// Frames are the latents the pipeline denoises, 30 seconds (max_inference_seconds) whatever
	// the duration asked for; the samples are cut from the decoded 30 seconds.
	Frames = 30 * LatentRate
	ditEps = 1e-6
)

// DitScale is what the residual stream is divided by in the block programs: its largest value
// measured is 1.4e4 (the last block's feed-forward, the first step of the reference's dog
// barking), 4.6 times under fp16's limit; divided by 4, 18 times (the gates and the
// cross-attention's output are divided by it on the way in).
const DitScale = 4.0

// Context is the text embedding: input ctx (1, 512, 2048), the text encoder's rows (zeros past
// the prompt); output y (1, 512, 1536), the context the blocks' cross-attention reads.
func Context() *Part {
	return &Part{Name: "dit_context", build: func(g *graph) {
		x := g.input("ctx", 1, TextRows, TextDim)
		d := &dit{graph: g}
		h := d.Activation(d.linear(x, "dit.condition_embedder.text_embedder.linear_1", DitDim, 1), "gelu_pytorch_tanh")
		g.output(d.linear(h, "dit.condition_embedder.text_embedder.linear_2", DitDim, 1))
	}}
}

// ModRows are the rows of a group's input mod: six a block (shift_msa, scale_msa + 1,
// gate_msa / DitScale, shift_mlp, scale_mlp + 1, gate_mlp / DitScale: the block's
// scale_shift_table plus the step's time projection), and for the last group the head's two
// (shift, scale + 1: scale_shift_table plus the time embedding).
func ModRows(first, count int) int {
	n := 6 * count
	if first+count == DitLayers {
		n += 2
	}
	return n
}

// Blocks is the DiT's blocks first to first + count - 1: inputs x, for the first group the
// latents (1, Frames, 128) as rows (the patch embedding, a 1x1 convolution, is its head),
// for the others the residual stream divided by DitScale (1, Frames, 1536); ctx (1, 512, 1536),
// Context's output; mod (1, ModRows, 1536). Output y, the residual stream after them, or for the
// last group the velocity (1, Frames, 128) as rows.
func Blocks(first, count int) *Part {
	return &Part{Name: fmt.Sprintf("dit_blocks_%d", first), build: func(g *graph) {
		d := &dit{graph: g}
		F := Frames
		var x *mlpackage.Var
		if first == 0 {
			x = d.linear(g.input("x", 1, F, LatentDim), "dit.patch_embedding", DitDim, 1/DitScale)
		} else {
			x = g.input("x", 1, F, DitDim)
		}
		ctx := g.input("ctx", 1, TextRows, DitDim)
		mod := g.input("mod", 1, ModRows(first, count), DitDim)
		freqs := make([]float64, ditHeadDim/2)
		for j := range freqs {
			freqs[j] = math.Pow(10000, -float64(2*j)/ditHeadDim)
		}
		cos, sin := g.RopeTables("dit", freqs, F, ditHeadDim)
		row := func(i int) *mlpackage.Var { return g.slice(mod, 1, i, i+1) }
		for j := range count {
			x = d.block(first+j, x, ctx, cos, sin, func(k int) *mlpackage.Var { return row(6*j + k) })
		}
		if first+count == DitLayers {
			h := g.add(g.mul(d.layerNorm(x), row(6*count+1)), row(6*count))
			x = d.linear(h, "dit.proj_out", LatentDim, 1)
		}
		g.output(x)
	}}
}

type dit struct{ *graph }

// linear is the named Linear (weight and bias) with its output scaled by s.
func (d *dit) linear(x *mlpackage.Var, name string, out int, s float32) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	w := d.Get(name + ".weight")
	if len(w) != out*in {
		d.Fail("%s: %d values for %d x %d", name, len(w), out, in)
	}
	return d.linearW(x, name, w, d.Get(name+".bias", out), out, s)
}

// linearW is a Linear of the given weight (out, in) and bias, its output scaled by s.
func (d *dit) linearW(x *mlpackage.Var, name string, w, bias []float32, out int, s float32) *mlpackage.Var {
	in := x.Shape[len(x.Shape)-1]
	b := bias
	if s != 1 {
		b = make([]float32, len(bias))
		for i, v := range bias {
			b[i] = v * s
		}
		ws := make([]float32, len(w))
		for i, v := range w {
			ws[i] = v * s
		}
		w = ws
	}
	shape := append(append([]int(nil), x.Shape[:len(x.Shape)-1]...), out)
	return d.op("linear", map[string]any{"x": x, "weight": d.Matrix(name+".weight", w, out, in), "bias": d.F16s(name+".bias", []int{out}, b)}, shape...)
}

// ropeOrder is the order of a head's features the programs keep its queries and keys in: Wan
// rotates the pairs (2j, 2j + 1) (rope_apply), blocks.ApplyRope the pairs (j, j + 64), so the
// projections' rows (and the bias and the norm's weight) are permuted: feature j of a head is
// Wan's 2j, feature j + 64 Wan's 2j + 1. The scores do not change.
func ropeOrder(v []float32, rowLen int) []float32 {
	out := make([]float32, len(v))
	half := ditHeadDim / 2
	for h := range DitDim / ditHeadDim {
		for j := range ditHeadDim {
			src := 2 * j
			if j >= half {
				src = 2*(j-half) + 1
			}
			copy(out[(h*ditHeadDim+j)*rowLen:(h*ditHeadDim+j+1)*rowLen], v[(h*ditHeadDim+src)*rowLen:(h*ditHeadDim+src+1)*rowLen])
		}
	}
	return out
}

// projection is a query or key projection and its RMSNorm (over all 1536 features), with the
// features in ropeOrder when rope.
func (d *dit) projection(x *mlpackage.Var, p, which string, rope bool) *mlpackage.Var {
	w := d.Get(p + "to_" + which + ".weight")
	b := d.Get(p+"to_"+which+".bias", DitDim)
	nw := d.Get(p+"norm_"+which+".weight", DitDim)
	if rope {
		w, b, nw = ropeOrder(w, DitDim), ropeOrder(b, 1), ropeOrder(nw, 1)
	}
	y := d.linearW(x, p+"to_"+which, w, b, DitDim, 1)
	return d.RMSNormWith(d.maxScaled(y, -1), p+"norm_"+which+".weight", nw, ditEps, 1)
}

// heads is x (1, n, 1536) as (1, 12, n, 128).
func (d *dit) heads(x *mlpackage.Var) *mlpackage.Var {
	n := x.Shape[1]
	return d.transpose(d.reshape(x, 1, n, ditHeads, ditHeadDim), 0, 2, 1, 3)
}

// merge is o (1, 12, n, 128) as (1, n, 1536).
func (d *dit) merge(o *mlpackage.Var) *mlpackage.Var {
	n := o.Shape[2]
	return d.reshape(d.transpose(o, 0, 2, 1, 3), 1, n, DitDim)
}

func (d *dit) block(i int, x, ctx, cos, sin *mlpackage.Var, mod func(int) *mlpackage.Var) *mlpackage.Var {
	p := fmt.Sprintf("dit.blocks.%d.", i)
	n := x.Shape[1]
	sc := 1 / math.Sqrt(ditHeadDim)
	// self-attention
	h := d.add(d.mul(d.layerNorm(x), mod(1)), mod(0))
	a := p + "attn1."
	q := d.reshape(d.projection(h, a, "q", true), 1, n, ditHeads, ditHeadDim)
	k := d.reshape(d.projection(h, a, "k", true), 1, n, ditHeads, ditHeadDim)
	q = d.transpose(d.ApplyRope(q, cos, sin), 0, 2, 1, 3)
	k = d.transpose(d.ApplyRope(k, cos, sin), 0, 2, 1, 3)
	v := d.heads(d.linear(h, a+"to_v", DitDim, 1))
	o := d.linear(d.merge(d.SDPAScale(q, k, v, sc, nil)), a+"to_out.0", DitDim, 1)
	x = d.add(x, d.mul(o, mod(2)))
	// cross-attention over the text (norm2 is WanModel's norm3, the affine one)
	nw := d.Get(p+"norm2.weight", DitDim)
	nb := d.Get(p+"norm2.bias", DitDim)
	h = d.add(d.mul(d.layerNorm(x), d.F16s(p+"norm2.weight", []int{DitDim}, nw)), d.F16s(p+"norm2.bias", []int{DitDim}, nb))
	a = p + "attn2."
	q = d.heads(d.projection(h, a, "q", false))
	k = d.heads(d.projection(ctx, a, "k", false))
	v = d.heads(d.linear(ctx, a+"to_v", DitDim, 1))
	o = d.linear(d.merge(d.SDPAScale(q, k, v, sc, nil)), a+"to_out.0", DitDim, 1/DitScale)
	x = d.add(x, o)
	// feed-forward
	h = d.add(d.mul(d.layerNorm(x), mod(4)), mod(3))
	f := d.Activation(d.linear(h, p+"ffn.net.0.proj", ditFFN, 1), "gelu_pytorch_tanh")
	f = d.linear(f, p+"ffn.net.2", DitDim, 1)
	return d.add(x, d.mul(f, mod(5)))
}

// maxScaled divides each row of x (over axis) by its largest magnitude (at least 1e-3), so a
// norm after it squares nothing larger than 1: a norm does not change with its input's scale.
func (d *dit) maxScaled(x *mlpackage.Var, axis int) *mlpackage.Var {
	abs := d.op("maximum", map[string]any{"x": x, "y": d.scale(x, -1)}, x.Shape...)
	rs := append([]int(nil), x.Shape...)
	a := axis
	if a < 0 {
		a += len(rs)
	}
	rs[a] = 1
	m := d.op("reduce_max", map[string]any{"x": abs, "axes": d.B.Ints(int32(axis)), "keep_dims": d.B.ScalarBool(true)}, rs...)
	m = d.op("maximum", map[string]any{"x": m, "y": d.B.ScalarF16(1e-3)}, rs...)
	return d.op("real_div", map[string]any{"x": x, "y": m}, x.Shape...)
}

// layerNorm is a LayerNorm over the last axis without weight and bias (eps 1e-6), of x scaled by
// maxScaled.
func (d *dit) layerNorm(x *mlpackage.Var) *mlpackage.Var {
	n := x.Shape[len(x.Shape)-1]
	ones, zeros := make([]float32, n), make([]float32, n)
	for i := range ones {
		ones[i] = 1
	}
	gamma := d.F16s(fmt.Sprintf("ln_ones_%d", n), []int{n}, ones)
	beta := d.F16s(fmt.Sprintf("ln_zeros_%d", n), []int{n}, zeros)
	return d.op("layer_norm", map[string]any{"x": d.maxScaled(x, -1), "axes": d.B.Ints(-1), "gamma": gamma, "beta": beta,
		"epsilon": d.B.ScalarF16(ditEps)}, x.Shape...)
}
