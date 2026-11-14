package mil

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// VaeChannels is DC-AE f32c32's decoder_block_out_channels.
var VaeChannels = []int{128, 256, 512, 512, 1024, 1024}

// VaeOp is one module of the decoder a stage runs: "in" (the latents divided by the scaling
// factor, conv_in and its shortcut, the channels repeated), "vit" (an EfficientViT block of
// Path, C channels), "up" (nearest 2x, the 3 x 3 convolution of Path to Out channels and the
// shortcut), "res" (a ResBlock) or "head" (norm_out, ReLU, conv_out to RGB).
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

// VaeStages is the decoder of a w x h picture in the stages of tools/amd/iron_sana.py
// (vae_stages), each a program of its own.
func VaeStages(w, h int) []VaeStage {
	gh, gw := h/VAEScale, w/VAEScale
	d := "vae.decoder.up_blocks"
	C := VaeChannels
	vit := func(i, j int) VaeOp { return VaeOp{"vit", fmt.Sprintf("%s.%d.%d", d, i, j), C[i], C[i]} }
	res := func(i, j int) VaeOp { return VaeOp{"res", fmt.Sprintf("%s.%d.%d", d, i, j), C[i], C[i]} }
	up := func(i int) VaeOp { return VaeOp{"up", fmt.Sprintf("%s.%d.0", d, i), C[i+1], C[i]} }
	st := func(s, c int, ops ...VaeOp) VaeStage { return VaeStage{ops, c, s * gh, s * gw} }
	return []VaeStage{
		st(1, 32, VaeOp{Kind: "in"}, vit(5, 0), vit(5, 1), vit(5, 2)),
		st(1, 1024, up(4), vit(4, 1), vit(4, 2), vit(4, 3)),
		st(2, 1024, up(3), vit(3, 1)),
		st(4, 512, vit(3, 2)),
		st(4, 512, vit(3, 3)),
		st(4, 512, up(2), res(2, 1), res(2, 2), res(2, 3)),
		st(8, 512, up(1), res(1, 1)),
		st(16, 256, res(1, 2), res(1, 3)),
		st(16, 256, up(0)),
		st(32, 128, res(0, 1)),
		st(32, 128, res(0, 2)),
		st(32, 128, res(0, 3)),
		st(32, 128, VaeOp{Kind: "head"}),
	}
}

// Vae is stage i of VaeStages(w, h): input x (1, C, H, W) (the latents for the first); output the
// next stage's input, or the picture (1, 3, h, w) in [-1, 1] roughly (the host clamps).
// scaling is the VAE's scaling_factor (vae/config.json).
func Vae(w, h, i int, scaling float64) *Part {
	st := VaeStages(w, h)[i]
	return part(fmt.Sprintf("vae_s%d", i), func(g *graph) {
		x := g.Input("x", 1, st.C, st.H, st.W)
		for _, op := range st.Ops {
			switch op.Kind {
			case "in":
				x = vaeIn(g, x, scaling)
			case "vit":
				x = vaeViT(g, x, op.Path)
			case "up":
				x = vaeUp(g, x, op.Path, op.Out)
			case "res":
				x = vaeRes(g, x, op.Path)
			case "head":
				x = g.Unary("relu", chanNorm(g, x, "vae.decoder.norm_out"))
				x = conv(g, x, "vae.decoder.conv_out", 3, 3, true)
			}
		}
		g.Output(x)
	})
}

// conv is the named convolution (path.weight, (cout, cin, k, k) or (cout, cin) for 1 x 1, and
// path.bias when bias) of x, stride 1, zero padded to the same size.
func conv(g *graph, x *mlpackage.Var, path string, cout, k int, bias bool) *mlpackage.Var {
	cin := x.Shape[1]
	shape := []int{cout, cin, k, k}
	if k == 1 {
		shape = shape[:2]
	}
	var b []float32
	if bias {
		b = g.Get(path+".bias", cout)
	}
	return g.Conv2D(x, path, g.Get(path+".weight", shape...), b, cout, k, 1)
}

// chanNorm is DC-AE's RMS norm over the channels with weight and bias (eps 1e-5), of x scaled by
// MaxScaled over the channels.
func chanNorm(g *graph, x *mlpackage.Var, path string) *mlpackage.Var {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	xs := g.MaxScaled(x, 1)
	ms := g.Reduce("mean", g.Mul(xs, xs), 1)
	r := g.Op("rsqrt", map[string]any{"x": ms, "epsilon": g.B.ScalarF16(1e-5)}, 1, 1, H, W)
	gm := g.F16s(path+".weight", []int{1, C, 1, 1}, g.Get(path+".weight", C))
	bt := g.F16s(path+".bias", []int{1, C, 1, 1}, g.Get(path+".bias", C))
	return g.Add(g.Mul(g.Mul(xs, r), gm), bt)
}

// repeatChannels is x (1, C, H, W) with each channel repeated r times (repeat_interleave), as a
// 1 x 1 convolution of 0/1 weights (exact in fp16): written with tile, the first stage at 1024 x
// 1024 (32 x 32 latents) failed the Neural Engine's compiler on Core AI ("empty reset layer
// candidates in SS cost model") once the q/k/v convolution followed it (design.md 16.189).
func repeatChannels(g *graph, x *mlpackage.Var, r int) *mlpackage.Var {
	C := x.Shape[1]
	w := make([]float32, C*r*C)
	for o := 0; o < C*r; o++ {
		w[o*C+o/r] = 1
	}
	return g.Conv2D(x, "vae.decoder.repeat", w, nil, C*r, 1, 1)
}

func vaeIn(g *graph, z *mlpackage.Var, scaling float64) *mlpackage.Var {
	z = g.Scale(z, float32(1/scaling))
	co := VaeChannels[len(VaeChannels)-1]
	y := conv(g, z, "vae.decoder.conv_in", co, 3, true)
	return g.Add(y, repeatChannels(g, z, co/z.Shape[1]))
}

// vaeUp is DCUpBlock2d with interpolate: the 3 x 3 convolution of nearest 2x, plus the shortcut
// (the input's channels repeated and shuffled into the pixels: nearest 2x when the channels stay,
// and when they halve, output channel c at a row of parity i takes input channel 2c + i).
func vaeUp(g *graph, x *mlpackage.Var, path string, cout int) *mlpackage.Var {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	u := g.Op("upsample_nearest_neighbor", map[string]any{"x": x, "scale_factor_height": g.B.ScalarI32(2),
		"scale_factor_width": g.B.ScalarI32(2)}, 1, C, 2*H, 2*W)
	y := conv(g, u, path+".conv", cout, 3, true)
	switch C {
	case cout:
		return g.Add(y, u)
	case 2 * cout:
		// (2c + i, y, x) -> (c, 2y + i, x), then twice across
		s := g.Reshape(g.Transpose(g.Reshape(x, cout, 2, H, W), 0, 2, 1, 3), 1, cout, 2*H, W)
		s = g.Op("upsample_nearest_neighbor", map[string]any{"x": s, "scale_factor_height": g.B.ScalarI32(1),
			"scale_factor_width": g.B.ScalarI32(2)}, 1, cout, 2*H, 2*W)
		return g.Add(y, s)
	}
	g.Fail("%s: %d to %d channels", path, C, cout)
	return nil
}

// vaeRes is a ResBlock: conv1, SiLU, conv2 (no bias), the RMS norm, plus x.
func vaeRes(g *graph, x *mlpackage.Var, path string) *mlpackage.Var {
	C := x.Shape[1]
	a := g.Unary("silu", conv(g, x, path+".conv1", C, 3, true))
	a = conv(g, a, path+".conv2", C, 3, false)
	return g.Add(chanNorm(g, a, path+".norm"), x)
}

// vaeViT is an EfficientViT block: the multi-scale linear attention (heads of 32; q, k and v of
// a head the 96 channels of the concatenated projections, and of the 5 x 5 depthwise and grouped
// 1 x 1 convolutions of them), then GLUMBConv, each normed and added to its input.
func vaeViT(g *graph, x *mlpackage.Var, path string) *mlpackage.Var {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	n, heads := H*W, C/32
	a := path + ".attn."
	wqkv := append(append(g.Get(a+"to_q.weight", C, C), g.Get(a+"to_k.weight", C, C)...), g.Get(a+"to_v.weight", C, C)...)
	qkv := g.Conv2D(x, a+"to_qkv", wqkv, nil, 3*C, 1, 1) // (1, 3C, H, W)
	ms := a + "to_qkv_multiscale.0."
	m := g.Conv2D(qkv, ms+"proj_in", g.Get(ms+"proj_in.weight", 3*C, 1, 5, 5), nil, 3*C, 5, 3*C)
	m = g.Conv2D(m, ms+"proj_out", g.Get(ms+"proj_out.weight", 3*C, 32), nil, 3*C, 1, 3*C/32)
	attend := func(t *mlpackage.Var) *mlpackage.Var {
		t = g.Reshape(t, 1, heads, 96, n)
		q := g.Unary("relu", g.Slice(t, 2, 0, 32))
		k := g.Scale(g.Unary("relu", g.Slice(t, 2, 32, 64)), 1.0/VaeKeyScale)
		v := g.Slice(t, 2, 64, 96)
		// (1, heads, 32, n) each: the attention of the transposes
		o := linAttn(g, g.Transpose(q, 0, 1, 3, 2), k, g.Transpose(v, 0, 1, 3, 2)) // (1, heads, n, 32)
		return g.Reshape(g.Transpose(o, 0, 1, 3, 2), 1, C, H, W)
	}
	o := g.Concat(1, attend(qkv), attend(m))
	o = conv(g, o, a+"to_out", C, 1, false)
	x = g.Add(chanNorm(g, o, a+"norm_out"), x)
	// GLUMBConv
	f := path + ".conv_out."
	ch := 4 * C
	h := g.Unary("silu", conv(g, x, f+"conv_inverted", 2*ch, 1, true))
	v, gate := gluDepthwise(g, h, f+"conv_depth", ch)
	h = g.Mul(v, g.Unary("silu", gate))
	h = conv(g, h, f+"conv_point", C, 1, false)
	return g.Add(chanNorm(g, h, f+"norm"), x)
}
