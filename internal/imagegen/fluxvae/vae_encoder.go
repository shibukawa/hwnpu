package fluxvae

import "fmt"

// VaeEncStages is the VAE's encoder (diffusers' Encoder: conv_in, a down block per channel count
// of layers_per_block residual blocks with a stride-2 downsampler after all but the last, the mid
// block with its one-head attention, conv_norm_out, SiLU, conv_out) of a w x h picture in stages,
// each a program of its own, as the decoder's (VaeStages) mirrored: at the picture's size and at
// half of it a residual block a stage (conv_in with the first, a downsampler with the last of its
// block), then the other down blocks whole, then the mid block with the head. Its ops are those
// of VaeStage: "in" (conv_in), "res", "down" (the downsampler of Path), "attn" and "head".
func VaeEncStages(v Vae, w, h int) []VaeStage {
	H, W := h, w
	e := "vae.encoder"
	var s []VaeStage
	prev, C := 3, 3
	for i, cout := range v.Channels {
		p := fmt.Sprintf("%s.down_blocks.%d", e, i)
		var ops []VaeOp
		if i == 0 {
			ops = append(ops, VaeOp{"in", e + ".conv_in", 3, cout})
			prev = cout
		}
		for j := 0; j < v.Layers; j++ {
			ops = append(ops, VaeOp{"res", fmt.Sprintf("%s.resnets.%d", p, j), prev, cout})
			prev = cout
		}
		last := i == len(v.Channels)-1
		if !last {
			ops = append(ops, VaeOp{"down", p + ".downsamplers.0.conv", cout, cout})
		}
		if i < 2 {
			// a residual block a stage: the largest activations (128 or 256 channels at the
			// picture's size or half of it)
			at := 0
			for j, op := range ops {
				if op.Kind != "res" || j == len(ops)-1 || ops[j+1].Kind == "down" {
					continue
				}
				s = append(s, VaeStage{ops[at : j+1], C, H, W})
				C, at = op.Out, j+1
			}
			s = append(s, VaeStage{ops[at:], C, H, W})
		} else {
			s = append(s, VaeStage{ops, C, H, W})
		}
		C = cout
		if !last {
			H, W = (H+1)/2, (W+1)/2
		}
	}
	mid := v.Channels[len(v.Channels)-1]
	return append(s, VaeStage{[]VaeOp{{"res", e + ".mid_block.resnets.0", mid, mid}, {Kind: "attn", Path: e + ".mid_block.attentions.0"},
		{"res", e + ".mid_block.resnets.1", mid, mid}, {Kind: "head"}}, mid, H, W})
}

// VaeEncProgram is stage i of VaeEncStages(v, w, h): input x (1, C, H, W), the picture (1, 3, h,
// w) in [-1, 1] for the first; output the next stage's input, or the latents' means (1, latent,
// h/8, w/8) for the last: conv_out's first half (the means of the latent distribution, after
// quant_conv when the VAE has it, folded into conv_out's weights).
func VaeEncProgram(v Vae, w, h, i int) *Part {
	st := VaeEncStages(v, w, h)[i]
	return part(fmt.Sprintf("vae_enc_s%d", i), func(g *graph) {
		x := g.Input("x", 1, st.C, st.H, st.W)
		for _, op := range st.Ops {
			switch op.Kind {
			case "in":
				x = conv(g, x, op.Path, op.Out, 3)
			case "res":
				x = vaeRes(g, x, op.Path, op.Out, v.Groups)
			case "down":
				// Downsample2D: padded by one row and column at the bottom and the right, then a 3 x 3
				// convolution of stride 2
				C := x.Shape[1]
				x = g.Conv2DPad(x, op.Path, g.Get(op.Path+".weight", op.Out, C, 3, 3), g.Get(op.Path+".bias", op.Out), op.Out, 3, 1, 2, [4]int{0, 1, 0, 1})
			case "attn":
				x = vaeAttn(g, x, op.Path, v.Groups)
			case "head":
				x = g.SiLU(GroupNorm(g, x, "vae.encoder.conv_norm_out", v.Groups, 1e-6))
				wt, b := encoderMeans(g, v, x.Shape[1])
				x = g.Conv2D(x, "vae.encoder.conv_out#mean", wt, b, v.Latent, 3, 1)
			}
		}
		g.Output(x)
	})
}

// encoderMeans are the weights (latent, cin, 3, 3) and bias of the means of the latent
// distribution: conv_out's first latent channels, or, with quant_conv (a 1 x 1 convolution of
// conv_out's 2 x latent channels), the first latent channels of quant_conv over conv_out.
func encoderMeans(g *graph, v Vae, cin int) (wt, bias []float32) {
	L, k := v.Latent, cin*9
	w := g.Get("vae.encoder.conv_out.weight", 2*L, cin, 3, 3)
	b := g.Get("vae.encoder.conv_out.bias", 2*L)
	if !v.Quant {
		return w[:L*k], b[:L]
	}
	q := g.Get("vae.quant_conv.weight", 2*L, 2*L)
	qb := g.Get("vae.quant_conv.bias", 2*L)
	wt, bias = make([]float32, L*k), make([]float32, L)
	for o := 0; o < L; o++ {
		acc := float64(qb[o])
		for m := 0; m < 2*L; m++ {
			c := q[o*2*L+m]
			acc += float64(c) * float64(b[m])
			row := w[m*k : (m+1)*k]
			out := wt[o*k : (o+1)*k]
			for j, x := range row {
				out[j] += c * x
			}
		}
		bias[o] = float32(acc)
	}
	return wt, bias
}
