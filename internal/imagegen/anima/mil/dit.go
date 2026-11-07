package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// The Cosmos-Predict2 2B DiT's shape (transformer/config.json).
const (
	DitHidden  = 2048
	DitLayers  = 28
	ditHeads   = 16
	ditHeadDim = 128
	ditMLP     = 8192
	ditLoRA    = 256
	// PatchDim is a patch's features: 16 latent channels by 2 x 2, channel major (the padding
	// mask's channel is zero for a whole image, its four columns of the embedding left out).
	PatchDim = 64
	// OutDim is a patch's velocity: 2 x 2 x 16, ordered dy * 32 + dx * 16 + c.
	OutDim = 64
	// TimeDim is the time part's output: temb (3 x 2048) then the normed sinusoid (2048).
	TimeDim = 4 * DitHidden
)

// Grid is the patches down and across of an image of w x h pixels (8 x 2 pixels a patch).
func Grid(w, h int) (gh, gw int) { return h / 16, w / 16 }

// TimeRows is the rows of Time's input: the Neural Engine takes the time part only with 8 rows or
// more (with one, Core ML places all of it on the CPU), so a call carries a step's sinusoid in
// row 0 and zeros after it.
const TimeRows = 8

// Time is the timestep embedding: input x (1, TimeRows, 2048), sinusoids of sigma (diffusers'
// Timesteps, cos first); output y (1, TimeRows, 8192), t_embedder's output then
// t_embedding_norm's, row by row.
func Time() *Part {
	return &Part{Name: "dit_time", build: func(g *graph) {
		x := g.input("x", 1, TimeRows, DitHidden)
		temb := g.Linear(g.op("silu", map[string]any{"x": g.Linear(x, "dit.t_embedder.1.linear_1.weight", "", DitHidden)}, 1, TimeRows, DitHidden),
			"dit.t_embedder.1.linear_2.weight", "", 3*DitHidden)
		et := g.rmsNorm(x, "dit.t_embedding_norm.weight", 1e-6)
		g.output(g.concat(-1, temb, et))
	}}
}

// patchEmbed is the patch embedding of x (1, patches, 64): (1, patches, 2048), the residual
// stream divided by DitScale. It runs at the head of the first group of blocks: alone, as a
// program of its own, Core ML places it on the CPU (at 512 x 512 and 1024 x 1024).
func patchEmbed(g *graph, x *mlpackage.Var) *mlpackage.Var {
	raw := g.Get("dit.x_embedder.proj.1.weight", DitHidden, (16+1)*4)
	wt := make([]float32, DitHidden*PatchDim)
	for o := 0; o < DitHidden; o++ {
		for i := 0; i < PatchDim; i++ {
			wt[o*PatchDim+i] = raw[o*68+i] / DitScale
		}
	}
	return g.linearConst(x, "dit.x_embedder", wt, DitHidden, nil)
}

// maxScoreBytes is the largest fp16 score tensor (heads x queries x keys) an attention should
// make at once: past about 64 MiB the Neural Engine runs q k^T and its softmax 2 to 3 times
// slower per score (M3, design.md 16.85; 65 MiB fast, 68 MiB slow).
const maxScoreBytes = 64 << 20

// DitQChunk is qchunk, halved until the self-attention's scores of a chunk (16 heads by qchunk
// by the patches of a w x h image, fp16) are within maxScoreBytes.
func DitQChunk(w, h, qchunk int) int {
	gh, gw := Grid(w, h)
	n := gh * gw
	if qchunk <= 0 || qchunk > n {
		qchunk = n
	}
	for qchunk > 1 && ditHeads*qchunk*n*2 > maxScoreBytes {
		qchunk /= 2
	}
	return qchunk
}

// Blocks is the DiT's blocks first..first+count-1: inputs x (1, patches, 2048), the residual
// stream divided by DitScale (for the first group, the patches (1, patches, 64), which it
// embeds); t (1, 1, 8192), a row of Time's output; ctx (1, 512, 1024), the text's
// conditioning. Output y, the residual stream after them. Self-attention runs qchunk queries at
// a time (0: all at once).
func Blocks(w, h, first, count, qchunk int) *Part {
	return &Part{Name: fmt.Sprintf("dit_blocks_%d", first), build: func(g *graph) {
		gh, gw := Grid(w, h)
		n := gh * gw
		var x *mlpackage.Var
		if first == 0 {
			x = patchEmbed(g, g.input("x", 1, n, PatchDim))
		} else {
			x = g.input("x", 1, n, DitHidden)
		}
		t := g.input("t", 1, 1, TimeDim)
		ctx := g.input("ctx", 1, TextRows, textHidden)
		temb := g.slice(t, -1, 0, 3*DitHidden)
		act := g.op("silu", map[string]any{"x": g.slice(t, -1, 3*DitHidden, TimeDim)}, 1, 1, DitHidden)
		cos, sin := ropeTables(g, gh, gw)
		for i := first; i < first+count; i++ {
			x = ditBlock(g, i, x, act, temb, ctx, cos, sin, qchunk)
		}
		g.output(x)
	}}
}

// Final is the final layer: inputs x and t as Blocks'; output y (1, patches, 64), each patch's
// velocity.
func Final(w, h int) *Part {
	return &Part{Name: "dit_final", build: func(g *graph) {
		gh, gw := Grid(w, h)
		n := gh * gw
		x := g.input("x", 1, n, DitHidden)
		t := g.input("t", 1, 1, TimeDim)
		act := g.op("silu", map[string]any{"x": g.slice(t, -1, 3*DitHidden, TimeDim)}, 1, 1, DitHidden)
		m := g.Linear(g.Linear(act, "dit.final_layer.adaln_modulation.1.weight", "", ditLoRA), "dit.final_layer.adaln_modulation.2.weight", "", 2*DitHidden)
		m = g.add(m, g.slice(t, -1, 0, 2*DitHidden))
		shift, scale := g.slice(m, -1, 0, DitHidden), g.slice(m, -1, DitHidden, 2*DitHidden)
		hh := g.add(g.mul(g.layerNorm(x), g.op("add", map[string]any{"x": scale, "y": g.B.ScalarF16(1)}, scale.Shape...)), shift)
		g.output(g.Linear(hh, "dit.final_layer.linear.weight", "", OutDim))
	}}
}

// modulation is an adaln_modulation of a block: shift, scale + 1, and gate / DitScale.
func modulation(g *graph, p string, act, temb *mlpackage.Var) (shift, scale1, gate *mlpackage.Var) {
	m := g.Linear(g.Linear(act, p+".1.weight", "", ditLoRA), p+".2.weight", "", 3*DitHidden)
	m = g.add(m, temb)
	shift = g.slice(m, -1, 0, DitHidden)
	scale1 = g.op("add", map[string]any{"x": g.slice(m, -1, DitHidden, 2*DitHidden), "y": g.B.ScalarF16(1)}, 1, 1, DitHidden)
	gate = g.scale(g.slice(m, -1, 2*DitHidden, 3*DitHidden), 1.0/DitScale)
	return shift, scale1, gate
}

func ditBlock(g *graph, i int, x, act, temb, ctx, cos, sin *mlpackage.Var, qchunk int) *mlpackage.Var {
	p := fmt.Sprintf("dit.blocks.%d.", i)
	E, H, D := DitHidden, ditHeads, ditHeadDim
	sc := 1 / math.Sqrt(float64(D))
	norm := func(name string) (*mlpackage.Var, *mlpackage.Var) {
		shift, scale1, gate := modulation(g, p+"adaln_modulation_"+name, act, temb)
		return g.add(g.mul(g.layerNorm(x), scale1), shift), gate
	}
	head := func(y *mlpackage.Var, normName string, rope bool) *mlpackage.Var {
		y = g.reshape(y, 1, y.Shape[1], H, D)
		if normName != "" {
			y = g.rmsNorm(y, normName, 1e-6)
		}
		if rope {
			y = g.ApplyRope(y, cos, sin)
		}
		return g.transpose(y, 0, 2, 1, 3)
	}
	// self-attention
	h, gate := norm("self_attn")
	a := p + "self_attn."
	q := head(g.Linear(h, a+"q_proj.weight", "", E), a+"q_norm.weight", true)
	k := head(g.Linear(h, a+"k_proj.weight", "", E), a+"k_norm.weight", true)
	v := head(g.Linear(h, a+"v_proj.weight", "", E), "", false)
	o := g.Linear(g.merge(g.attention(q, k, v, sc, nil, qchunk)), a+"output_proj.weight", "", E)
	x = g.add(x, g.mul(o, gate))
	// cross-attention over the text
	h, gate = norm("cross_attn")
	a = p + "cross_attn."
	q = head(g.Linear(h, a+"q_proj.weight", "", E), a+"q_norm.weight", false)
	k = head(g.Linear(ctx, a+"k_proj.weight", "", E), a+"k_norm.weight", false)
	v = head(g.Linear(ctx, a+"v_proj.weight", "", E), "", false)
	o = g.Linear(g.merge(g.attention(q, k, v, sc, nil, qchunk)), a+"output_proj.weight", "", E)
	x = g.add(x, g.mul(o, gate))
	// MLP
	h, gate = norm("mlp")
	f := g.Activation(g.Linear(h, p+"mlp.layer1.weight", "", ditMLP), "gelu")
	f = g.Linear(f, p+"mlp.layer2.weight", "", E)
	return g.add(x, g.mul(f, gate))
}

// ropeTables is CosmosRotaryPosEmbed's cos and sin (1, patches, 1, 128) as blocks.ApplyRope
// takes them: pair j (x[j] with x[j+64]) turns by the frame (22 pairs, all at frame 0), the row
// (21) and the column (21), each axis at its NTK-scaled theta (rope_scale 1, 4, 4).
func ropeTables(g *graph, gh, gw int) (cos, sin *mlpackage.Var) {
	c, s := RopeValues(gh, gw)
	n := gh * gw
	return g.F16s(fmt.Sprintf("dit_rope_cos_%dx%d", gh, gw), []int{1, n, 1, ditHeadDim}, c),
		g.F16s(fmt.Sprintf("dit_rope_sin_%dx%d", gh, gw), []int{1, n, 1, ditHeadDim}, s)
}

// RopeValues is ropeTables' values.
func RopeValues(gh, gw int) (cos, sin []float32) {
	D := ditHeadDim
	dimH := D / 6 * 2
	dimW := dimH
	dimT := D - dimH - dimW
	hTheta := 10000 * math.Pow(4, float64(dimH)/float64(dimH-2))
	wTheta := 10000 * math.Pow(4, float64(dimW)/float64(dimW-2))
	freqs := func(dim int, theta float64) []float64 {
		f := make([]float64, dim/2)
		for j := range f {
			f[j] = 1 / math.Pow(theta, float64(2*j)/float64(dim))
		}
		return f
	}
	fh, fw := freqs(dimH, hTheta), freqs(dimW, wTheta)
	n := gh * gw
	cos, sin = make([]float32, n*D), make([]float32, n*D)
	half := D / 2
	for py := 0; py < gh; py++ {
		for px := 0; px < gw; px++ {
			i := py*gw + px
			for j := 0; j < half; j++ {
				var a float64
				switch {
				case j < dimT/2:
					a = 0 // the frame, 0 for an image
				case j < dimT/2+dimH/2:
					a = float64(py) * fh[j-dimT/2]
				default:
					a = float64(px) * fw[j-dimT/2-dimH/2]
				}
				c, s := float32(math.Cos(a)), float32(math.Sin(a))
				cos[i*D+j], cos[i*D+j+half] = c, c
				sin[i*D+j], sin[i*D+j+half] = s, s
			}
		}
	}
	return cos, sin
}
