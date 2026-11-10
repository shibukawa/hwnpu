package mil

import (
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

const headPrefix = "model.head."

// Head is DPT's head (DualDPT's depth branch for the any-view models) on the ViT's output:
// input feats (1, tokens, 4 FeatDim); output (1, 2, H, W), channel 0 depth's logit (the host
// takes exp), channel 1 the sky's logit (DPT) or the confidence's (DualDPT).
func Head(c Config) *Part {
	return part("head", func(g *graph) {
		gh, gw, T, D := c.GH(), c.GW(), c.Tokens(), c.FeatDim()
		oc, Fe := c.OutChannels, c.Features
		feats := g.Input("feats", 1, T, 4*D)
		var lv [4]*mlpackage.Var
		for i := 0; i < 4; i++ {
			f := g.Slice(g.Slice(feats, 1, 1, T), -1, i*D, (i+1)*D) // the patches' rows
			if c.Dual {
				f = layerNorm(g, f, headPrefix+"norm", 1e-5)
			}
			p := fmt.Sprintf("%sprojects.%d", headPrefix, i)
			x := g.LinearConst(f, p, g.Get(p+".weight", oc[i], D), oc[i], g.Get(p+".bias", oc[i])) // (1, P, oc)
			if c.Dual {
				x = g.Add(x, g.F16s(fmt.Sprintf("pe%d", i), []int{1, gh * gw, oc[i]}, UVEmbed(gw, gh, oc[i], float64(c.Width)/float64(c.Height))))
			}
			r := fmt.Sprintf("%sresize_layers.%d", headPrefix, i)
			switch i {
			case 0, 1:
				lv[i] = deconv(g, x, r, oc[i], 4>>i, gh, gw)
			case 2:
				lv[i] = g.Reshape(g.Transpose(x, 0, 2, 1), 1, oc[i], gh, gw)
			case 3:
				x = g.Reshape(g.Transpose(x, 0, 2, 1), 1, oc[i], gh, gw)
				lv[i] = g.Conv2DStride(x, r, g.Get(r+".weight", oc[i], oc[i], 3, 3), g.Get(r+".bias", oc[i]), oc[i], 3, 1, 2)
			}
		}
		var rn [4]*mlpackage.Var
		for i, x := range lv {
			p := fmt.Sprintf("%sscratch.layer%d_rn", headPrefix, i+1)
			rn[i] = g.Conv2D(x, p, g.Get(p+".weight", Fe, oc[i], 3, 3), nil, Fe, 3, 1)
		}
		size := func(x *mlpackage.Var) [2]int { return [2]int{x.Shape[2], x.Shape[3]} }
		d := refine(g, 4, rn[3], nil, size(rn[2]))
		d = refine(g, 3, d, rn[2], size(rn[1]))
		d = refine(g, 2, d, rn[1], size(rn[0]))
		d = refine(g, 1, d, rn[0], [2]int{2 * rn[0].Shape[2], 2 * rn[0].Shape[3]})
		s := headPrefix + "scratch."
		d = conv(g, d, s+"output_conv1", Fe/2, 3, true)
		d = resize(g, d, c.Height, c.Width)
		if c.Dual {
			pe := UVEmbed(c.Width, c.Height, Fe/2, float64(c.Width)/float64(c.Height)) // (H W, C)
			t := make([]float32, len(pe))
			n := c.Width * c.Height
			for p := 0; p < n; p++ {
				for ch := 0; ch < Fe/2; ch++ {
					t[ch*n+p] = pe[p*(Fe/2)+ch]
				}
			}
			d = g.Add(d, g.F16s("pe_out", []int{1, Fe / 2, c.Height, c.Width}, t))
		}
		out := func(p string, n int) *mlpackage.Var {
			h := g.Unary("relu", conv(g, d, p+".0", 32, 3, true))
			return conv(g, h, p+".2", n, 1, true)
		}
		if c.Dual {
			g.Output(out(s+"output_conv2", 2))
			return
		}
		g.Output(g.Concat(1, out(s+"output_conv2", 1), out(s+"sky_output_conv2", 1)))
	})
}

// conv is the named convolution (path.weight (cout, cin, k, k), or (cout, cin) for 1 x 1;
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

// deconv is ConvTranspose2d with kernel = stride = k (path's weight (cin, cout, k, k)) of x (1,
// gh gw, cin), the tokens of a gh x gw grid: one linear to cout k k channels per token, then
// PixelShuffle (1, cout, k gh, k gw).
func deconv(g *graph, x *mlpackage.Var, path string, cout, k, gh, gw int) *mlpackage.Var {
	cin := x.Shape[2]
	raw, b := g.Get(path+".weight", cin, cout, k, k), g.Get(path+".bias", cout)
	wt := make([]float32, cout*k*k*cin) // rows (c, ky, kx)
	bias := make([]float32, cout*k*k)
	for ci := 0; ci < cin; ci++ {
		for c := 0; c < cout; c++ {
			for j := 0; j < k*k; j++ {
				wt[(c*k*k+j)*cin+ci] = raw[(ci*cout+c)*k*k+j]
			}
		}
	}
	for c := 0; c < cout; c++ {
		for j := 0; j < k*k; j++ {
			bias[c*k*k+j] = b[c]
		}
	}
	y := g.LinearConst(x, path, wt, cout*k*k, bias)
	y = g.Reshape(g.Transpose(y, 0, 2, 1), 1, cout*k*k, gh, gw)
	return g.Op("pixel_shuffle", map[string]any{"x": y, "upscale_factor": g.B.ScalarI32(int32(k))}, 1, cout, k*gh, k*gw)
}

// rcu is a residual conv unit: ReLU, conv, ReLU, conv, plus x.
func rcu(g *graph, x *mlpackage.Var, p string) *mlpackage.Var {
	C := x.Shape[1]
	a := conv(g, g.Unary("relu", x), p+".conv1", C, 3, true)
	a = conv(g, g.Unary("relu", a), p+".conv2", C, 3, true)
	return g.Add(a, x)
}

// refine is fusion block r: (x + RCU1(lateral)), RCU2, the bilinear resize (align_corners) to
// size, the 1 x 1 out_conv.
func refine(g *graph, r int, x, lateral *mlpackage.Var, size [2]int) *mlpackage.Var {
	p := fmt.Sprintf("%sscratch.refinenet%d", headPrefix, r)
	if lateral != nil {
		x = g.Add(x, rcu(g, lateral, p+".resConfUnit1"))
	}
	x = rcu(g, x, p+".resConfUnit2")
	x = resize(g, x, size[0], size[1])
	return conv(g, x, p+".out_conv", x.Shape[1], 1, true)
}

// resize is the bilinear resize (align_corners) of x (1, C, H, W) to (1, C, h, w): two matrix
// products with the interpolation matrices.
func resize(g *graph, x *mlpackage.Var, h, w int) *mlpackage.Var {
	H, W := x.Shape[2], x.Shape[3]
	if W != w {
		m := Interp(w, W)
		t := make([]float32, W*w) // (W, w): the transpose
		for o := 0; o < w; o++ {
			for i := 0; i < W; i++ {
				t[i*w+o] = m[o*W+i]
			}
		}
		x = g.MatMul(x, g.F16s(fmt.Sprintf("interp_%dto%d", W, w), []int{W, w}, t), false, false)
	}
	if H != h {
		x = g.MatMul(g.F16s(fmt.Sprintf("interp_%dto%d", H, h), []int{1, 1, h, H}, Interp(h, H)), x, false, false)
	}
	return x
}

// Interp is bilinear interpolation with align_corners as a matrix (out, in).
func Interp(out, in int) []float32 {
	m := make([]float32, out*in)
	for o := 0; o < out; o++ {
		s := 0.0
		if out > 1 {
			s = float64(o) * float64(in-1) / float64(out-1)
		}
		i0 := min(int(math.Floor(s)), in-1)
		t := s - float64(i0)
		i1 := min(i0+1, in-1)
		m[o*in+i0] += float32(1 - t)
		m[o*in+i1] += float32(t)
	}
	return m
}

// UVEmbed is DualDPT's _add_pos_embed (create_uv_grid, position_grid_to_embed, ratio 0.1) of a
// pw x ph map of C channels: (ph pw, C).
func UVEmbed(pw, ph, C int, aspect float64) []float32 {
	diag := math.Sqrt(aspect*aspect + 1)
	sx, sy := aspect/diag, 1/diag
	lin := func(span float64, n int) []float64 {
		v := make([]float64, n)
		lo, hi := -span*float64(n-1)/float64(n), span*float64(n-1)/float64(n)
		for i := range v {
			if n > 1 {
				v[i] = float32To64(lo + (hi-lo)*float64(i)/float64(n-1))
			}
		}
		return v
	}
	xs, ys := lin(sx, pw), lin(sy, ph)
	D := C / 2
	omega := make([]float64, D/2)
	for i := range omega {
		omega[i] = float32To64(1 / math.Pow(100, float64(i)/(float64(D)/2)))
	}
	out := make([]float32, ph*pw*C)
	for y := 0; y < ph; y++ {
		for x := 0; x < pw; x++ {
			row := out[(y*pw+x)*C:]
			for a, pos := range []float64{xs[x], ys[y]} {
				for i, om := range omega {
					ang := pos * om
					row[a*D+i] = float32(0.1 * math.Sin(ang))
					row[a*D+D/2+i] = float32(0.1 * math.Cos(ang))
				}
			}
		}
	}
	return out
}

// float32To64 rounds x to float32 as the reference's float32 grids are.
func float32To64(x float64) float64 { return float64(float32(x)) }
