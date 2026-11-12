package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// VaeOp is one module of the decoder a stage runs: "in" (the latents unscaled, z / scaling_factor
// + shift_factor, and conv_in), "res" (a ResnetBlock2D of Path, In to Out channels), "attn" (the
// mid block's attention), "up" (nearest 2x and the upsampler's 3 x 3 convolution) or "head"
// (conv_norm_out, SiLU, conv_out to RGB).
type VaeOp struct {
	Kind    string
	Path    string
	In, Out int
}

// VaeStage is a stage of the decoder: its modules and its input's shape (channels, rows,
// columns).
type VaeStage struct {
	Ops     []VaeOp
	C, H, W int
}

// VaeStages is the decoder of a w x h picture in stages, each a program of its own (as
// tools/amd/iron_chroma_parts.py's vae_stages): the mid block, the first two up blocks whole, then
// a residual block a stage, the upsampler with the last of its block.
func VaeStages(v Vae, w, h int) []VaeStage {
	H, W := h/8, w/8
	d := "vae.decoder"
	ch := make([]int, len(v.Channels))
	for i := range ch {
		ch[i] = v.Channels[len(ch)-1-i] // 512 512 256 128
	}
	n := v.Layers + 1
	mid := ch[0]
	s := []VaeStage{{[]VaeOp{{Kind: "in", Out: mid}, {"res", d + ".mid_block.resnets.0", mid, mid}, {Kind: "attn", Path: d + ".mid_block.attentions.0"},
		{"res", d + ".mid_block.resnets.1", mid, mid}}, v.Latent, H, W}}
	prev := mid
	for i, cout := range ch {
		p := fmt.Sprintf("%s.up_blocks.%d", d, i)
		var res []VaeOp
		for j := 0; j < n; j++ {
			in := cout
			if j == 0 {
				in = prev
			}
			res = append(res, VaeOp{"res", fmt.Sprintf("%s.resnets.%d", p, j), in, cout})
		}
		var up []VaeOp
		if i < len(ch)-1 {
			up = []VaeOp{{"up", p + ".upsamplers.0.conv", cout, cout}}
		}
		if i < 2 {
			s = append(s, VaeStage{append(res, up...), prev, H, W})
		} else {
			for j, r := range res {
				ops := []VaeOp{r}
				if j == n-1 {
					ops = append(ops, up...)
				}
				s = append(s, VaeStage{ops, r.In, H, W})
			}
		}
		prev = cout
		if up != nil {
			H, W = 2*H, 2*W
		}
	}
	return append(s, VaeStage{[]VaeOp{{Kind: "head"}}, ch[len(ch)-1], H, W})
}

// VaeProgram is stage i of VaeStages(v, w, h): input x (1, C, H, W) (the latents, (1, 16, h/8, w/8), for
// the first); output the next stage's input, or the picture (1, 3, h, w) in [-1, 1] roughly (the
// host clamps).
func VaeProgram(v Vae, w, h, i int) *Part {
	st := VaeStages(v, w, h)[i]
	return part(fmt.Sprintf("vae_s%d", i), func(g *graph) {
		x := g.Input("x", 1, st.C, st.H, st.W)
		for _, op := range st.Ops {
			switch op.Kind {
			case "in":
				x = g.AddScalar(g.Scale(x, float32(1/v.Scaling)), float32(v.Shift))
				x = conv(g, x, "vae.decoder.conv_in", op.Out, 3)
			case "res":
				x = vaeRes(g, x, op.Path, op.Out, v.Groups)
			case "attn":
				x = vaeAttn(g, x, op.Path, v.Groups)
			case "up":
				C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
				u := g.Op("upsample_nearest_neighbor", map[string]any{"x": x, "scale_factor_height": g.B.ScalarI32(2),
					"scale_factor_width": g.B.ScalarI32(2)}, 1, C, 2*H, 2*W)
				x = conv(g, u, op.Path, op.Out, 3)
			case "head":
				x = g.SiLU(GroupNorm(g, x, "vae.decoder.conv_norm_out", v.Groups, 1e-6))
				x = conv(g, x, "vae.decoder.conv_out", 3, 3)
			}
		}
		g.Output(x)
	})
}

// conv is the named convolution (path.weight (cout, cin, k, k), path.bias) of x, stride 1, zero
// padded to the same size.
func conv(g *graph, x *mlpackage.Var, path string, cout, k int) *mlpackage.Var {
	cin := x.Shape[1]
	shape := []int{cout, cin, k, k}
	if k == 1 {
		shape = shape[:2]
	}
	return g.Conv2D(x, path, g.Get(path+".weight", shape...), g.Get(path+".bias", cout), cout, k, 1)
}

// GroupNorm is torch's GroupNorm (groups, eps, path.weight and path.bias) of x (1, C, H, W). A
// group's values are contiguous: viewed as (1, groups, a, b) with no axis past 16384 (the Neural
// Engine places an operation on a wider axis on the CPU), they are divided by their largest
// magnitude (so that the squares fit fp16), less their mean, then divided by the root of their
// mean square: the mean is taken out before the squares, as a group's |mean| / std reaches 6.3
// in this decoder (tools/amd/iron_chroma_parts.py).
func GroupNorm(g *graph, x *mlpackage.Var, path string, groups int, eps float64) *mlpackage.Var {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	cg := C / groups
	n := cg * H * W
	b := W
	for k := 2; k <= H; k++ {
		if H%k == 0 && W*k <= 16384 && n/(W*k) <= 16384 {
			b = W * k
		}
	}
	if b > 16384 || n/b > 16384 {
		g.Fail("%s: a group of %d values in rows of %d", path, n, b)
	}
	t := g.Reshape(x, 1, groups, n/b, b)
	reduce := func(typ string, y *mlpackage.Var) *mlpackage.Var { return g.Reduce(typ, g.Reduce(typ, y, 3), 2) }
	abs := g.Op("maximum", map[string]any{"x": t, "y": g.Scale(t, -1)}, t.Shape...)
	m := g.Op("maximum", map[string]any{"x": reduce("max", abs), "y": g.B.ScalarF16(1e-3)}, 1, groups, 1, 1)
	t = g.Div(t, m)
	d := g.Sub(t, reduce("mean", t))
	v := reduce("mean", g.Mul(d, d))
	// eps of the scaled values, eps / m^2, as (sqrt(eps) / m)^2 (no constant under fp16's normal
	// range; it is 0 where it is that small next to the variance)
	e := g.Op("real_div", map[string]any{"x": g.B.ScalarF16(float32(math.Sqrt(eps))), "y": m}, m.Shape...)
	r := g.Op("rsqrt", map[string]any{"x": g.Add(v, g.Mul(e, e)), "epsilon": g.B.ScalarF16(0)}, 1, groups, 1, 1)
	y := g.Reshape(g.Mul(d, r), 1, C, H, W)
	gm := g.F16s(path+".weight", []int{1, C, 1, 1}, g.Get(path+".weight", C))
	bt := g.F16s(path+".bias", []int{1, C, 1, 1}, g.Get(path+".bias", C))
	return g.Add(g.Mul(y, gm), bt)
}

// vaeRes is ResnetBlock2D: GroupNorm, SiLU, conv1, GroupNorm, SiLU, conv2, plus x (through
// conv_shortcut when the channels change).
func vaeRes(g *graph, x *mlpackage.Var, path string, cout, groups int) *mlpackage.Var {
	sc := x
	if x.Shape[1] != cout {
		sc = conv(g, x, path+".conv_shortcut", cout, 1)
	}
	a := g.SiLU(GroupNorm(g, x, path+".norm1", groups, 1e-6))
	a = conv(g, a, path+".conv1", cout, 3)
	a = g.SiLU(GroupNorm(g, a, path+".norm2", groups, 1e-6))
	a = conv(g, a, path+".conv2", cout, 3)
	return g.Add(sc, a)
}

// vaeAttn is the mid block's attention: GroupNorm, one head over the pixels (q, k and v 1 x 1
// convolutions of the linears' weights), to_out, plus x.
func vaeAttn(g *graph, x *mlpackage.Var, path string, groups int) *mlpackage.Var {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	n := H * W
	h := GroupNorm(g, x, path+".group_norm", groups, 1e-6)
	proj := func(name string) *mlpackage.Var {
		y := conv(g, h, path+"."+name, C, 1)
		return g.Reshape(g.Transpose(g.Reshape(y, 1, C, n), 0, 2, 1), 1, 1, n, C) // (1, 1, pixels, C)
	}
	q, k, v := proj("to_q"), proj("to_k"), proj("to_v")
	chunk := n
	for chunk > 64 && chunk*n*2 > maxScoreBytes {
		chunk /= 2
	}
	o := g.QChunkAttention(q, k, v, 1/math.Sqrt(float64(C)), nil, chunk)
	o = g.Reshape(g.Transpose(g.Reshape(o, 1, n, C), 0, 2, 1), 1, C, H, W)
	return g.Add(x, conv(g, o, path+".to_out.0", C, 1))
}
