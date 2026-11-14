// Package mil describes MoGe-2 (Ruicheng/moge-2-vit{s,b,l}-normal) as MIL programs, which hwnpu
// writes as Core ML packages for Apple's Neural Engine (llm.WritePart), for one picture size
// each: the ViT (DINOv2, depth/mil's, its layers' projections summed and the scale head on the
// class token) and the rest (the ConvStack neck over five scales and the points, normal and mask
// heads) as one program.
package mil

import (
	"fmt"
	"math"
	"slices"

	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/torchzip"
	dmil "github.com/shibukawa/hwnpu/internal/vision/depth/mil"
)

type graph = milpart.Graph

func part(name string, body func(g *graph)) *milpart.Part {
	return &milpart.Part{Name: name, Model: "moge", Body: body}
}

// Config is a MoGe-2's shape at one picture size (iron_moge.Config).
type Config struct {
	dmil.Config
	BaseW, BaseH int
	Dim          int    // the encoder's projections
	Neck         [5]int // the channels of each scale
	NeckRes      [5]int // the neck's residual blocks a scale
	HeadRes      [5]int // the heads'
	Resamplers   [4]string
	Scale        []int // the scale head's dims
}

// Level is scale i's rows and columns (2^i times the patch grid).
func (c Config) Level(i int) (int, int) { return c.BaseH << i, c.BaseW << i }

// Weights reads model.pt (torchzip): "model.*" by name; a 1 x 1 convolution's weight (out, in,
// 1, 1) reads as the matrix (out, in).
type Weights struct {
	f    *torchzip.File
	path string
}

// OpenWeights opens a MoGe-2 checkpoint (model.pt).
func OpenWeights(path string) (*Weights, error) {
	f, err := torchzip.Open(path)
	if err != nil {
		return nil, err
	}
	return &Weights{f: f, path: path}, nil
}

// Get returns a tensor as float32 and its shape.
func (w *Weights) Get(name string) ([]float32, []int, error) {
	t, ok := w.f.Lookup(name)
	if !ok {
		return nil, nil, fmt.Errorf("moge: %s has no tensor %s", w.path, name)
	}
	v, err := w.f.Float32(name)
	shape := t.Shape
	if len(shape) == 4 && shape[2] == 1 && shape[3] == 1 {
		shape = shape[:2]
	}
	return v, shape, err
}

// Rows is not used by the parts.
func (w *Weights) Rows(name string, begin, end int) ([]float32, error) {
	return nil, fmt.Errorf("moge: rows of %s", name)
}

// Source names the checkpoint.
func (w *Weights) Source() string { return "torchzip model.pt" }

// Close closes the file.
func (w *Weights) Close() error { return w.f.Close() }

// ReadConfig reads model.pt's model_config for a picture of bw x bh patches.
func (w *Weights) ReadConfig(bw, bh int) (Config, error) {
	root, ok := w.f.Root.(map[any]any)
	if !ok {
		return Config{}, fmt.Errorf("moge: %s holds no model_config", w.path)
	}
	mc, ok := root["model_config"].(map[any]any)
	if !ok {
		return Config{}, fmt.Errorf("moge: %s holds no model_config", w.path)
	}
	get := func(m map[any]any, keys ...string) any {
		var v any = m
		for _, k := range keys {
			mm, ok := v.(map[any]any)
			if !ok {
				return nil
			}
			v = mm[k]
		}
		return v
	}
	ints := func(v any) []int {
		var out []int
		l, _ := v.([]any)
		for _, x := range l {
			if n, ok := toInt(x); ok {
				out = append(out, n)
			}
		}
		return out
	}
	vit := map[string][3]int{"dinov2_vits14": {384, 6, 12}, "dinov2_vitb14": {768, 12, 12}, "dinov2_vitl14": {1024, 16, 24}}
	name, _ := get(mc, "encoder", "backbone").(string)
	v, ok := vit[name]
	if !ok {
		return Config{}, fmt.Errorf("moge: backbone %q is not supported", name)
	}
	if get(mc, "remap_output") != "exp" {
		return Config{}, fmt.Errorf("moge: remap_output %v is not supported", get(mc, "remap_output"))
	}
	out := ints(get(mc, "encoder", "intermediate_layers"))
	if n, ok := toInt(get(mc, "encoder", "intermediate_layers")); ok {
		for i := v[2] - n; i < v[2]; i++ {
			out = append(out, i)
		}
	}
	dim, _ := toInt(get(mc, "encoder", "dim_out"))
	c := Config{BaseW: bw, BaseH: bh, Dim: dim, Scale: ints(get(mc, "scale_head", "dims"))}
	c.Config = dmil.Config{Width: dmil.Patch * bw, Height: dmil.Patch * bh, Hidden: v[0], Heads: v[1], MLP: 4 * v[0],
		OutLayers: out, AltStart: -1, QKNormStart: -1, RopeStart: -1, Prefix: "model.encoder.backbone."}
	c.Layers = slices.Max(out) + 1
	copy(c.Neck[:], ints(get(mc, "neck", "dim_res_blocks")))
	copy(c.NeckRes[:], ints(get(mc, "neck", "num_res_blocks")))
	copy(c.HeadRes[:], ints(get(mc, "points_head", "num_res_blocks")))
	rs, _ := get(mc, "neck", "resamplers").([]any)
	for i := range c.Resamplers {
		if i < len(rs) {
			c.Resamplers[i], _ = rs[i].(string)
		}
	}
	for _, h := range []string{"neck", "points_head", "normal_head", "mask_head"} {
		for _, k := range []string{"res_block_in_norm", "res_block_hidden_norm"} {
			if x := get(mc, h, k); x != nil && x != "none" {
				return Config{}, fmt.Errorf("moge: %s %s %v is not supported", h, k, x)
			}
		}
	}
	if c.Dim == 0 || len(c.Scale) < 2 || len(out) == 0 {
		return Config{}, fmt.Errorf("moge: %s: an incomplete model_config", w.path)
	}
	return c, nil
}

// Vit is the patch embedding, the blocks, the encoder's projections and the scale head: input
// patches (1, tokens, depth/mil.PatchIn) as depth's; output (1, tokens, dim), the summed 1 x 1
// projections of the layers, and in row 0 (the class token's, which the rest does not read)
// the metric scale's logit first.
func Vit(c Config) *milpart.Part {
	return part("vit", func(g *graph) {
		T := c.Tokens()
		feats := dmil.FeatList(g, c.Config, g.Input("patches", 1, T, dmil.PatchIn))
		var f *mlpackage.Var
		bias := make([]float32, c.Dim)
		for i, n := range feats {
			p := fmt.Sprintf("model.encoder.output_projections.%d", i)
			y := g.LinearConst(n, p, g.Get(p+".weight", c.Dim, c.Hidden), c.Dim, nil)
			for j, b := range g.Get(p+".bias", c.Dim) {
				bias[j] += b
			}
			if f == nil {
				f = y
			} else {
				f = g.Add(f, y)
			}
		}
		f = g.Add(f, g.F16s("proj_bias", []int{c.Dim}, bias))
		// the scale head on the last layer's normed class token
		h := g.Slice(feats[len(feats)-1], 1, 0, 1)
		for j := 0; j+1 < len(c.Scale); j++ {
			p := fmt.Sprintf("model.scale_head.%d", 2*j)
			h = g.LinearConst(h, p, g.Get(p+".weight", c.Scale[j+1], c.Scale[j]), c.Scale[j+1], g.Get(p+".bias", c.Scale[j+1]))
			if j+2 < len(c.Scale) {
				h = g.Unary("relu", h)
			}
		}
		pad := g.F16s("scale_pad", []int{1, 1, c.Dim - h.Shape[2]}, make([]float32, c.Dim-h.Shape[2]))
		g.Output(g.Concat(1, g.Concat(-1, h, pad), g.Slice(f, 1, 1, T)))
	})
}

// Dense is the neck and the three heads: input f (1, tokens, dim), Vit's output; output (1, 7,
// H, W) at the last scale (16 times the patch grid): the points' 3 channels, the normal's 3,
// the mask's logit.
func Dense(c Config) *milpart.Part {
	return part("dense", func(g *graph) {
		T := c.Tokens()
		f := g.Slice(g.Input("f", 1, T, c.Dim), 1, 1, T)
		neck := stack(g, c, "model.neck", c.NeckRes, func(i int, x *mlpackage.Var) *mlpackage.Var {
			H, W := c.Level(i)
			ch := c.Neck[i]
			wt := g.Get(fmt.Sprintf("model.neck.input_blocks.%d.weight", i))
			b := g.Get(fmt.Sprintf("model.neck.input_blocks.%d.bias", i), ch)
			cin := len(wt) / ch
			uv := uvConst(W, H, wt, b, ch, cin) // the view plane's UV's share and the bias
			if i == 0 {
				w0 := make([]float32, ch*c.Dim)
				for o := 0; o < ch; o++ {
					copy(w0[o*c.Dim:], wt[o*cin:o*cin+c.Dim])
				}
				y := g.LinearConst(f, "neck_in0", w0, ch, nil)
				y = g.Reshape(g.Transpose(y, 0, 2, 1), 1, ch, H, W)
				return g.Add(y, g.F16s("neck_uv0", []int{1, ch, H, W}, uv))
			}
			return g.Add(x, g.F16s(fmt.Sprintf("neck_uv%d", i), []int{1, ch, H, W}, uv))
		})
		var outs []*mlpackage.Var
		for _, h := range []string{"points", "normal", "mask"} {
			p := "model." + h + "_head"
			levels := stack(g, c, p, c.HeadRes, func(i int, x *mlpackage.Var) *mlpackage.Var {
				y := conv(g, neck[i], fmt.Sprintf("%s.input_blocks.%d", p, i), c.Neck[i], 1, false)
				if x == nil {
					return y
				}
				return g.Add(x, y)
			})
			n := []int{3, 3, 1}[len(outs)]
			outs = append(outs, conv(g, levels[4], p+".output_blocks.4", n, 1, false))
		}
		g.Output(g.Concat(1, outs...))
	})
}

// stack is a ConvStack (path): at each scale input(i, x) is the running features with the
// scale's input added, then the residual blocks; then the resampler to the next. It returns each
// scale's features before its resampler.
func stack(g *graph, c Config, path string, nres [5]int, input func(i int, x *mlpackage.Var) *mlpackage.Var) []*mlpackage.Var {
	var x *mlpackage.Var
	var outs []*mlpackage.Var
	for i := 0; i < 5; i++ {
		x = input(i, x)
		for r := 0; r < nres[i]; r++ {
			p := fmt.Sprintf("%s.res_blocks.%d.%d.layers", path, i, r)
			y := conv(g, g.Unary("relu", x), p+".2", c.Neck[i], 3, true)
			y = conv(g, g.Unary("relu", y), p+".5", c.Neck[i], 3, true)
			x = g.Add(x, y)
		}
		outs = append(outs, x)
		if i == 4 {
			break
		}
		p := fmt.Sprintf("%s.resamplers.%d", path, i)
		switch c.Resamplers[i] {
		case "conv_transpose":
			x = deconv2(g, x, p+".0", c.Neck[i+1])
		case "bilinear":
			H, W := c.Level(i + 1)
			x = resize(g, x, H, W)
		default:
			g.Fail("resampler %q is not supported", c.Resamplers[i])
		}
		x = conv(g, x, p+".1", c.Neck[i+1], 3, true)
	}
	return outs
}

// conv is the named convolution of x (path.weight (cout, cin, k, k), or (cout, cin) for 1 x 1;
// path.bias), stride 1; a 3 x 3 pads by replicating the edge (padding_mode="replicate").
// replicate is true for that padding.
func conv(g *graph, x *mlpackage.Var, path string, cout, k int, replicate bool) *mlpackage.Var {
	cin := x.Shape[1]
	shape := []int{cout, cin, k, k}
	if k == 1 {
		shape = shape[:2]
	}
	wt, b := g.Get(path+".weight", shape...), g.Get(path+".bias", cout)
	if k == 1 || !replicate {
		return g.Conv2D(x, path, wt, b, cout, k, 1)
	}
	p := int32(k / 2)
	H, W := x.Shape[2], x.Shape[3]
	xp := g.Op("pad", map[string]any{"x": x, "pad": g.B.Ints(p, p, p, p), "mode": g.Str("replicate"),
		"constant_val": g.B.ScalarF16(0)}, 1, cin, H+2*int(p), W+2*int(p))
	inputs := map[string]any{
		"x": xp, "weight": g.F16s(path+".k", []int{cout, cin, k, k}, wt), "bias": g.F16s(path+".b", []int{cout}, b),
		"strides": g.B.Ints(1, 1), "pad_type": g.Str("custom"), "pad": g.B.Ints(0, 0, 0, 0), "dilations": g.B.Ints(1, 1), "groups": g.B.ScalarI32(1),
	}
	return g.Op("conv", inputs, 1, cout, H, W)
}

// shuffleRows is the most output rows deconv2 shuffles at a time.
const shuffleRows = 256

// deconv2 is ConvTranspose2d with kernel = stride = 2 (path: weight (cin, cout, 2, 2), bias)
// of x (1, cin, H, W): a 1 x 1 convolution to cout x 4 channels, then PixelShuffle.
func deconv2(g *graph, x *mlpackage.Var, path string, cout int) *mlpackage.Var {
	cin, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	raw, b := g.Get(path+".weight", cin, cout, 2, 2), g.Get(path+".bias", cout)
	wt := make([]float32, cout*4*cin)
	bias := make([]float32, cout*4)
	for ci := 0; ci < cin; ci++ {
		for o := 0; o < cout; o++ {
			for j := 0; j < 4; j++ {
				wt[(o*4+j)*cin+ci] = raw[(ci*cout+o)*4+j]
			}
		}
	}
	for o := 0; o < cout; o++ {
		for j := 0; j < 4; j++ {
			bias[o*4+j] = b[o]
		}
	}
	y := g.Conv2D(x, path, wt, bias, cout*4, 1, 1)
	// the shuffle in bands of at most shuffleRows output rows, concatenated (the same values): the
	// whole of 64 x 400 x 400 put the M1's dense head off the Neural Engine on Core AI ("empty
	// reset layer candidates in SS cost model", design.md 16.190), two bands of 200 kept it on
	rows := (2*H + shuffleRows - 1) / shuffleRows
	if rows <= 1 {
		return g.Op("pixel_shuffle", map[string]any{"x": y, "upscale_factor": g.B.ScalarI32(2)}, 1, cout, 2*H, 2*W)
	}
	var parts []*mlpackage.Var
	for r := 0; r < rows; r++ {
		lo, hi := r*H/rows, (r+1)*H/rows
		parts = append(parts, g.Op("pixel_shuffle", map[string]any{"x": g.Slice(y, 2, lo, hi), "upscale_factor": g.B.ScalarI32(2)}, 1, cout, 2*(hi-lo), 2*W))
	}
	return g.Concat(2, parts...)
}

// resize is torch's bilinear interpolate (align_corners=False) of x (1, C, H, W) to (1, C, h,
// w): two matrix products.
func resize(g *graph, x *mlpackage.Var, h, w int) *mlpackage.Var {
	H, W := x.Shape[2], x.Shape[3]
	mw := interp(w, W)
	t := make([]float32, W*w)
	for o := 0; o < w; o++ {
		for i := 0; i < W; i++ {
			t[i*w+o] = mw[o*W+i]
		}
	}
	x = g.MatMul(x, g.F16s(fmt.Sprintf("interpf_%dto%d", W, w), []int{W, w}, t), false, false)
	return g.MatMul(g.F16s(fmt.Sprintf("interpf_%dto%d", H, h), []int{1, 1, h, H}, interp(h, H)), x, false, false)
}

// interp is bilinear interpolation with align_corners=False as a matrix (out, in): source (o +
// 0.5) in / out - 0.5, clamped at 0.
func interp(out, in int) []float32 {
	m := make([]float32, out*in)
	for o := 0; o < out; o++ {
		s := math.Max(0, (float64(o)+0.5)*float64(in)/float64(out)-0.5)
		i0 := min(int(math.Floor(s)), in-1)
		t := s - float64(i0)
		i1 := min(i0+1, in-1)
		m[o*in+i0] += float32(1 - t)
		m[o*in+i1] += float32(t)
	}
	return m
}

// uvConst is a 1 x 1 convolution's share of the view plane's UV (its last two input channels)
// plus its bias, at every pixel of a w x h scale: (ch, h, w).
func uvConst(w, h int, wt, b []float32, ch, cin int) []float32 {
	a := float64(w) / float64(h)
	sx, sy := a/math.Sqrt(1+a*a), 1/math.Sqrt(1+a*a)
	lin := func(span float64, n, i int) float64 {
		lo, hi := float32(-span*float64(n-1)/float64(n)), float32(span*float64(n-1)/float64(n))
		if n == 1 {
			return float64(lo)
		}
		return float64(float32(float64(lo) + float64(hi-lo)*float64(i)/float64(n-1)))
	}
	out := make([]float32, ch*h*w)
	for y := 0; y < h; y++ {
		v := lin(sy, h, y)
		for x := 0; x < w; x++ {
			u := lin(sx, w, x)
			for o := 0; o < ch; o++ {
				out[(o*h+y)*w+x] = float32(float64(b[o]) + float64(wt[o*cin+cin-2])*u + float64(wt[o*cin+cin-1])*v)
			}
		}
	}
	return out
}

// toInt is an integer of the unpickled config, whatever its Go type.
func toInt(v any) (int, bool) {
	switch n := v.(type) {
	case int:
		return n, true
	case int64:
		return int(n), true
	case int32:
		return int(n), true
	case uint8:
		return int(n), true
	case uint16:
		return int(n), true
	case int8:
		return int(n), true
	case float64:
		return int(n), n == math.Trunc(n)
	}
	return 0, false
}
