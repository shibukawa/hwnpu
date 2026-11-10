package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// VaeOp is one module of the VAE decoder a stage runs: "in" (the latents' unnormalization and
// post_quant_conv, as one 1 x 1 convolution, then conv_in), "res" (a residual block of Path,
// In to Out channels), "attn" (the middle attention), "up" (nearest 2x and the 3 x 3
// convolution of Path's upsampler to Out channels) or "head" (norm_out, SiLU, conv_out).
type VaeOp struct {
	Kind    string
	Path    string
	In, Out int
}

// VaeStage is a stage of the decoder: its modules and its input's shape (channels, rows, columns).
type VaeStage struct {
	Ops     []VaeOp
	C, H, W int
}

// VaeStages is the decoder of an image of w x h pixels in the stages of tools/amd/iron_anima.py
// (vae_stages): each a program of its own.
func VaeStages(w, h int) []VaeStage {
	h8, w8 := h/8, w/8
	d := "vae.decoder."
	res := func(path string, in, out int) VaeOp { return VaeOp{"res", d + path, in, out} }
	up := func(path string, in, out int) VaeOp { return VaeOp{"up", d + path, in, out} }
	s := []VaeStage{
		{[]VaeOp{{Kind: "in"}, res("mid_block.resnets.0", 384, 384), {Kind: "attn", Path: d + "mid_block.attentions.0"}, res("mid_block.resnets.1", 384, 384)}, 16, h8, w8},
		{[]VaeOp{res("up_blocks.0.resnets.0", 384, 384), res("up_blocks.0.resnets.1", 384, 384), res("up_blocks.0.resnets.2", 384, 384), up("up_blocks.0", 384, 192)}, 384, h8, w8},
		{[]VaeOp{res("up_blocks.1.resnets.0", 192, 384), res("up_blocks.1.resnets.1", 384, 384), res("up_blocks.1.resnets.2", 384, 384)}, 192, 2 * h8, 2 * w8},
		{[]VaeOp{up("up_blocks.1", 384, 192), res("up_blocks.2.resnets.0", 192, 192)}, 384, 2 * h8, 2 * w8},
		{[]VaeOp{res("up_blocks.2.resnets.1", 192, 192), res("up_blocks.2.resnets.2", 192, 192)}, 192, 4 * h8, 4 * w8},
		{[]VaeOp{up("up_blocks.2", 192, 96)}, 192, 4 * h8, 4 * w8},
	}
	for i := 0; i < 3; i++ {
		s = append(s, VaeStage{[]VaeOp{res(fmt.Sprintf("up_blocks.3.resnets.%d", i), 96, 96)}, 96, 8 * h8, 8 * w8})
	}
	return append(s, VaeStage{[]VaeOp{{Kind: "head"}}, 96, 8 * h8, 8 * w8})
}

// Vae is stage i of VaeStages(w, h): input x (1, C, H, W) (the DiT's latents, 16 channels, for
// the first); output y, the next stage's input, or the image (1, 3, h, w) in [-1, 1] roughly
// (the host clamps). The latents' mean and std (vae/config.json) go into the first stage.
func Vae(w, h, i int, mean, std []float64, qchunk int) *Part {
	st := VaeStages(w, h)[i]
	return part(fmt.Sprintf("vae_s%d", i), func(g *graph) {
		x := g.Input("x", 1, st.C, st.H, st.W)
		for _, op := range st.Ops {
			switch op.Kind {
			case "in":
				x = vaeIn(g, x, mean, std)
			case "res":
				x = vaeRes(g, x, op.Path, op.Out)
			case "attn":
				x = vaeAttn(g, x, op.Path, qchunk)
			case "up":
				x = vaeUp(g, x, op.Path, op.Out)
			case "head":
				x = g.Op("silu", map[string]any{"x": chanNorm(g, x, "vae.decoder.norm_out.gamma")}, x.Shape...)
				x = conv(g, x, "vae.decoder.conv_out", 3)
			}
		}
		g.Output(x)
	})
}

// conv is a causal 3 x 3 x 3 convolution's last temporal tap (the frames before the first are
// zeros) or a 1 x 1 x 1 one (k 1), named path (.weight, .bias), zero padded to the same size.
func conv(g *graph, x *mlpackage.Var, path string, cout int) *mlpackage.Var {
	raw, shape, err := g.W.Get(path + ".weight")
	if err != nil {
		g.Fail("%v", err)
	}
	cin := x.Shape[1]
	var k, kt int
	switch len(shape) {
	case 5: // (cout, cin, t, k, k)
		k, kt = shape[3], shape[2]
	case 4: // (cout, cin, k, k): the upsamplers' 2 D convolutions
		k, kt = shape[2], 1
	default:
		g.Fail("%s: weight shape %v", path, shape)
	}
	if shape[0] != cout || shape[1] != cin {
		g.Fail("%s: weight %v for %d to %d channels", path, shape, cin, cout)
	}
	wt := make([]float32, cout*cin*k*k)
	for o := 0; o < cout*cin; o++ {
		copy(wt[o*k*k:(o+1)*k*k], raw[(o*kt+kt-1)*k*k:][:k*k])
	}
	return convConst(g, x, path, wt, g.Get(path+".bias", cout), cout, k)
}

func convConst(g *graph, x *mlpackage.Var, name string, wt, bias []float32, cout, k int) *mlpackage.Var {
	return g.Conv2D(x, name, wt, bias, cout, k, 1)
}

// chanNorm is the VAE's RMS_norm over the channels: x / ||x|| * sqrt(C) * gamma, which is x /
// rms(x) * gamma (its activations are under 8, so their squares fit fp16).
func chanNorm(g *graph, x *mlpackage.Var, gamma string) *mlpackage.Var {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	gm := g.F16s(gamma, []int{1, C, 1, 1}, g.Get(gamma))
	sq := g.Mul(x, x)
	ms := g.Op("reduce_mean", map[string]any{"x": sq, "axes": g.B.Ints(1), "keep_dims": g.B.ScalarBool(true)}, 1, 1, H, W)
	r := g.Op("rsqrt", map[string]any{"x": ms, "epsilon": g.B.ScalarF16(1e-6)}, 1, 1, H, W)
	return g.Mul(g.Mul(x, r), gm)
}

func vaeIn(g *graph, z *mlpackage.Var, mean, std []float64) *mlpackage.Var {
	pq := g.Get("vae.post_quant_conv.weight", 16, 16, 1, 1, 1)
	pb := g.Get("vae.post_quant_conv.bias", 16)
	wt := make([]float32, 16*16)
	b := make([]float32, 16)
	for o := 0; o < 16; o++ {
		acc := float64(pb[o])
		for i := 0; i < 16; i++ {
			wt[o*16+i] = pq[o*16+i] * float32(std[i])
			acc += float64(pq[o*16+i]) * mean[i]
		}
		b[o] = float32(acc)
	}
	x := convConst(g, z, "vae.in", wt, b, 16, 1)
	return conv(g, x, "vae.decoder.conv_in", 384)
}

func vaeRes(g *graph, x *mlpackage.Var, path string, cout int) *mlpackage.Var {
	h := x
	if x.Shape[1] != cout {
		h = conv(g, x, path+".conv_shortcut", cout)
	}
	// the Neural Engine's own silu, unlike the DiT's (blocks.SiLU): written with exp and real_div,
	// no image changed (PSNR within 0.04 dB of diffusers' fp32) and the VAE took 15 to 20 % longer
	// (design.md 16.95)
	a := g.Op("silu", map[string]any{"x": chanNorm(g, x, path+".norm1.gamma")}, x.Shape...)
	a = conv(g, a, path+".conv1", cout)
	a = g.Op("silu", map[string]any{"x": chanNorm(g, a, path+".norm2.gamma")}, a.Shape...)
	a = conv(g, a, path+".conv2", cout)
	return g.Add(a, h)
}

// vaeAttn is the middle attention: one head over the pixels, its q, k and v a 1 x 1
// convolution, then proj, added to x.
func vaeAttn(g *graph, x *mlpackage.Var, path string, qchunk int) *mlpackage.Var {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	n := H * W
	h := chanNorm(g, x, path+".norm.gamma")
	qkvw := g.Get(path+".to_qkv.weight", 3*C, C, 1, 1)
	qkvb := g.Get(path+".to_qkv.bias", 3*C)
	part := func(j int) *mlpackage.Var {
		y := convConst(g, h, fmt.Sprintf("%s.to_qkv.%d", path, j), qkvw[j*C*C:(j+1)*C*C], qkvb[j*C:(j+1)*C], C, 1)
		return g.Reshape(g.Transpose(g.Reshape(y, 1, C, n), 0, 2, 1), 1, 1, n, C) // (1, 1, pixels, C)
	}
	q, k, v := part(0), part(1), part(2)
	o := g.QChunkAttention(q, k, v, 1/math.Sqrt(float64(C)), nil, qchunk)
	o = g.Reshape(g.Transpose(g.Reshape(o, 1, n, C), 0, 2, 1), 1, C, H, W)
	pw := g.Get(path+".proj.weight", C, C, 1, 1)
	o = convConst(g, o, path+".proj", pw, g.Get(path+".proj.bias", C), C, 1)
	return g.Add(x, o)
}

// vaeUp is nearest 2x upsampling and the upsampler's 3 x 3 convolution. upsample_nearest_neighbor
// is 3 times faster on the Neural Engine than the same as reshapes and tiles (stage 5 at 1024 x
// 1024 on M3: 412 ms against 1275 ms); OpenVINO IR lowers it to those.
func vaeUp(g *graph, x *mlpackage.Var, path string, cout int) *mlpackage.Var {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	u := g.Op("upsample_nearest_neighbor", map[string]any{"x": x, "scale_factor_height": g.B.ScalarI32(2),
		"scale_factor_width": g.B.ScalarI32(2)}, 1, C, 2*H, 2*W)
	return conv(g, u, path+".upsamplers.0.resample.1", cout)
}
