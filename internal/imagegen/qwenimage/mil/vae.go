package mil

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Vae is AutoencoderKLQwenImage21's shape (vae/config.json).
type Vae struct {
	Base     int       `json:"decoder_base_dim"`
	Mult     []int     `json:"dim_mult"`
	Res      int       `json:"num_res_blocks"`
	TDown    []bool    `json:"temperal_downsample"`
	Z        int       `json:"z_dim"`
	Out      int       `json:"out_channels"`
	Residual bool      `json:"is_residual"`
	Attn     []float64 `json:"attn_scales"`
	Patch    *int      `json:"patch_size"`
	Mean     []float64 `json:"latents_mean"`
	Std      []float64 `json:"latents_std"`
}

// ReadVae reads root/vae/config.json.
func ReadVae(root string) (Vae, error) {
	b, err := os.ReadFile(filepath.Join(root, "vae", "config.json"))
	if err != nil {
		return Vae{}, err
	}
	var v Vae
	if err := json.Unmarshal(b, &v); err != nil {
		return Vae{}, fmt.Errorf("qwenimage: vae/config.json: %w", err)
	}
	if !v.Residual || v.Patch != nil || len(v.Attn) != 0 || len(v.Mult) != len(v.TDown)+1 || len(v.Mean) != v.Z || len(v.Std) != v.Z {
		return Vae{}, fmt.Errorf("qwenimage: the VAE (%v, residual %v) is not Qwen-Image-2.1's", v.Mult, v.Residual)
	}
	return v, nil
}

// dims are the decoder's widths: conv_in's, then each up block's output.
func (v Vae) dims() []int {
	m := append([]int{v.Mult[len(v.Mult)-1]}, reverse(v.Mult)...)
	out := make([]int, len(m))
	for i, u := range m {
		out[i] = v.Base * u
	}
	return out
}

func reverse(xs []int) []int {
	out := make([]int, len(xs))
	for i, x := range xs {
		out[len(xs)-1-i] = x
	}
	return out
}

// VaeOp is one module of the decoder a stage runs: "in" (the latents unnormalized and
// post_quant_conv, as one 1 x 1 convolution, then conv_in to Out), "res" (a residual block of
// Path, In to Out channels), "attn" (the middle attention), "up" (Path's upsampler: nearest 2x and
// a 3 x 3 convolution of Out channels, plus DupUp3D's shortcut of the up block's input, In
// channels, of time factor T) or "head" (norm_out, SiLU, conv_out to RGBA).
type VaeOp struct {
	Kind    string
	Path    string
	In, Out int
	T       int
}

// VaeStage is a stage of the decoder: its modules and its input's shape (channels, rows, columns).
type VaeStage struct {
	Ops     []VaeOp
	C, H, W int
	// Rows and Cols tile a stage too wide for the Neural Engine (wideChannels): its program runs on
	// windows of the input (Window), Rows.N x Cols.N of them, each giving a tile of the output.
	Rows, Cols Tiling
}

// Tiling splits an axis of a stage's input of Size into N tiles of at most Tile each, every one
// read with Halo more of the input around it (Windows): N <= 1 is the axis whole.
type Tiling struct {
	N, Size, Tile, Halo int
}

// TileWindow is a tile of an axis: its window of the input starts at Start, and the tile is Len of
// it from Off (in the input: Start+Off).
type TileWindow struct {
	Start, Off, Len int
}

// Windows are the axis's tiles. Every window is Tile+Halo long and inside the input, so that the
// zero padding at the input's edges is the stage's own; a tile is Halo/2 or more (Halo for two
// tiles) from a window's edge that is not one, farther than the stage reaches (VaeStage.Reach).
func (t Tiling) Windows() []TileWindow {
	if t.N <= 1 {
		return []TileWindow{{0, 0, t.Size}}
	}
	win := t.Tile + t.Halo
	out := make([]TileWindow, t.N)
	for i := range out {
		lo := i * t.Tile
		start := min(max(lo-t.Halo/2, 0), t.Size-win)
		out[i] = TileWindow{start, lo - start, min(t.Tile, t.Size-lo)}
	}
	return out
}

// Window is the length of a window: the axis whole when not tiled.
func (t Tiling) Window() int {
	if t.N <= 1 {
		return t.Size
	}
	return t.Tile + t.Halo
}

// wideChannels and wideWindow: a stage whose residual blocks give more than wideChannels channels
// runs on windows of at most wideWindow rows and columns. Through Core AI on M1 the Neural
// Engine's compiler fails on the up blocks of 1152 to 1152 channels at 64 x 64 and 128 x 128
// ("Allocation failure for L2-dep pair"; the GPU then), but takes them at 32 x 32, the one of 1152
// to 576 channels at 256 x 256, and the norm and a convolution of 1152 channels at 48 x 48 or of
// 768 at 64 x 64 (design.md 16.184).
const (
	wideChannels = 768
	wideWindow   = 48
)

// Reach is how far a pixel of the stage's output depends on the input around it, in the input's
// pixels: a 3 x 3 convolution a pixel, a residual block two, an up block's convolution at twice the
// size one; 0 for a stage that sees the whole input (the middle attention), which is not tiled.
func (s VaeStage) Reach() int {
	r := 0
	for _, op := range s.Ops {
		switch op.Kind {
		case "res":
			r += 2
		case "up", "in", "head":
			r++
		case "attn":
			return 0
		}
	}
	return r
}

// Scale is how many times larger the stage's output is than its input on each axis.
func (s VaeStage) Scale() int {
	for _, op := range s.Ops {
		if op.Kind == "up" {
			return 2
		}
	}
	return 1
}

// OutC is the channels of the stage's output.
func (s VaeStage) OutC() int { return s.Ops[len(s.Ops)-1].Out }

// width is the most channels the stage's residual blocks give.
func (s VaeStage) width() int {
	w := 0
	for _, op := range s.Ops {
		if op.Kind == "res" {
			w = max(w, op.Out)
		}
	}
	return w
}

// tile sets the stage's tiling when it is too wide for the Neural Engine (wideChannels): the fewest
// tiles of each axis whose windows are at most wideWindow long.
func (s *VaeStage) tile() {
	s.Rows, s.Cols = Tiling{N: 1, Size: s.H}, Tiling{N: 1, Size: s.W}
	r := s.Reach()
	if s.width() <= wideChannels || r == 0 {
		return
	}
	axis := func(size int) Tiling {
		if size <= wideWindow {
			return Tiling{N: 1, Size: size}
		}
		for n := 2; n <= size; n++ {
			t := Tiling{N: n, Size: size, Tile: (size + n - 1) / n, Halo: r}
			if n > 2 {
				t.Halo = 2 * r // a tile inside has the stage's reach on both sides
			}
			if t.Tile+t.Halo <= min(wideWindow, size) {
				return t
			}
		}
		return Tiling{N: 1, Size: size}
	}
	s.Rows, s.Cols = axis(s.H), axis(s.W)
}

// VaeStages is the decoder of a w x h picture in stages, each a program of its own: the middle,
// each up block, the last (at the full size) split after its first residual block.
func VaeStages(v Vae, w, h int) []VaeStage {
	dims := v.dims()
	tup := make([]bool, len(v.TDown))
	for i, t := range v.TDown {
		tup[len(v.TDown)-1-i] = t
	}
	gh, gw := h/VAEScale, w/VAEScale
	d := "vae.decoder."
	res := func(path string, in, out int) VaeOp { return VaeOp{Kind: "res", Path: d + path, In: in, Out: out} }
	s := []VaeStage{{Ops: []VaeOp{{Kind: "in", Out: dims[0]}, res("mid_block.resnets.0", dims[0], dims[0]),
		{Kind: "attn", Path: d + "mid_block.attentions.0"}, res("mid_block.resnets.1", dims[0], dims[0])}, C: v.Z, H: gh, W: gw}}
	for i := 0; i < len(dims)-1; i++ {
		cin, cout := dims[i], dims[i+1]
		p := fmt.Sprintf("up_blocks.%d", i)
		var ops []VaeOp
		for j := 0; j <= v.Res; j++ {
			in := cout
			if j == 0 {
				in = cin
			}
			ops = append(ops, res(fmt.Sprintf("%s.resnets.%d", p, j), in, cout))
		}
		if i < len(dims)-2 {
			t := 1
			if tup[i] {
				t = 2
			}
			ops = append(ops, VaeOp{Kind: "up", Path: d + p, In: cin, Out: cout, T: t})
			s = append(s, VaeStage{Ops: ops, C: cin, H: gh, W: gw})
			gh, gw = 2*gh, 2*gw
			continue
		}
		// the last block, at the full size
		s = append(s, VaeStage{Ops: ops[:1], C: cin, H: gh, W: gw})
		s = append(s, VaeStage{Ops: append(ops[1:], VaeOp{Kind: "head", In: cout, Out: v.Out}), C: cout, H: gh, W: gw})
	}
	for i := range s {
		s[i].tile()
	}
	return s
}

// VaePart is stage i of VaeStages(v, w, h): input x (1, C, H, W) (the DiT's latents for the first,
// unnormalized there); output the next stage's input, or the picture (1, 4, h, w), RGBA in [-1, 1]
// roughly (the host clamps). A tiled stage's input is a window (1, C, Rows.Window(), Cols.Window())
// and its output that window's, Scale() times larger. The middle attention runs qchunk pixels'
// queries at a time (0: all).
func VaePart(v Vae, w, h, i, qchunk int) *Part {
	st := VaeStages(v, w, h)[i]
	return part(fmt.Sprintf("vae_s%d", i), func(g *graph) {
		x := g.Input("x", 1, st.C, st.Rows.Window(), st.Cols.Window())
		in := x // an up block's input, for its shortcut
		for _, op := range st.Ops {
			switch op.Kind {
			case "in":
				x = conv(g, vaeIn(g, x, v), "vae.decoder.conv_in", op.Out)
			case "res":
				x = vaeRes(g, x, op.Path, op.Out)
			case "attn":
				x = vaeAttn(g, x, op.Path, qchunk)
			case "up":
				x = g.Add(vaeUp(g, x, op.Path+".upsampler.resample.1", op.Out), dupUp(g, in, op.Path, op.Out, op.T))
			case "head":
				x = g.Unary("silu", chanNorm(g, x, "vae.decoder.norm_out.gamma"))
				x = conv(g, x, "vae.decoder.conv_out", op.Out)
			}
		}
		g.Output(x)
	})
}

// conv is a 3 x 3 (or 1 x 1) convolution named path (.weight, .bias), zero padded to the same
// size: QwenImage21CausalConv3d of one frame is a 2 D convolution.
func conv(g *graph, x *mlpackage.Var, path string, cout int) *mlpackage.Var {
	raw, shape, err := g.W.Get(path + ".weight")
	if err != nil {
		g.Fail("%v", err)
	}
	cin := x.Shape[1]
	k := 1
	switch len(shape) {
	case 4:
		k = shape[2]
	case 2:
	default:
		g.Fail("%s: weight shape %v", path, shape)
	}
	if shape[0] != cout || shape[1] != cin {
		g.Fail("%s: weight %v for %d to %d channels", path, shape, cin, cout)
	}
	return g.Conv2D(x, path, raw, g.Get(path+".bias", cout), cout, k, 1)
}

// chanNorm is the VAE's RMS_norm over the channels (x / ||x|| sqrt(C) gamma, which is x / rms(x)
// gamma) of x divided first by its largest magnitude over the channels (the norm does not change).
func chanNorm(g *graph, x *mlpackage.Var, gamma string) *mlpackage.Var {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	gm := g.F16s(gamma, []int{1, C, 1, 1}, g.Get(gamma))
	x = g.MaxScaled(x, 1)
	ms := g.Op("reduce_mean", map[string]any{"x": g.Mul(x, x), "axes": g.B.Ints(1), "keep_dims": g.B.ScalarBool(true)}, 1, 1, H, W)
	r := g.Op("rsqrt", map[string]any{"x": ms, "epsilon": g.B.ScalarF16(1e-12)}, 1, 1, H, W)
	return g.Mul(g.Mul(x, r), gm)
}

// vaeIn is the latents (the DiT's, normalized) times std plus mean, then post_quant_conv: one 1 x
// 1 convolution.
func vaeIn(g *graph, z *mlpackage.Var, v Vae) *mlpackage.Var {
	Z := v.Z
	pq := g.Get("vae.post_quant_conv.weight", Z, Z)
	pb := g.Get("vae.post_quant_conv.bias", Z)
	wt := make([]float32, Z*Z)
	b := make([]float32, Z)
	for o := 0; o < Z; o++ {
		acc := float64(pb[o])
		for i := 0; i < Z; i++ {
			wt[o*Z+i] = pq[o*Z+i] * float32(v.Std[i])
			acc += float64(pq[o*Z+i]) * v.Mean[i]
		}
		b[o] = float32(acc)
	}
	return g.Conv2D(z, "vae.in", wt, b, Z, 1, 1)
}

func vaeRes(g *graph, x *mlpackage.Var, path string, cout int) *mlpackage.Var {
	h := x
	if x.Shape[1] != cout {
		h = conv(g, x, path+".conv_shortcut", cout)
	}
	// the Neural Engine's own silu, as Anima's VAE (no image changed with blocks.SiLU, design.md 16.95)
	a := g.Unary("silu", chanNorm(g, x, path+".norm1.gamma"))
	a = conv(g, a, path+".conv1", cout)
	a = g.Unary("silu", chanNorm(g, a, path+".norm2.gamma"))
	a = conv(g, a, path+".conv2", cout)
	return g.Add(a, h)
}

// vaeAttn is the middle attention: one head over the pixels, its q, k and v a 1 x 1 convolution,
// then proj, added to x.
func vaeAttn(g *graph, x *mlpackage.Var, path string, qchunk int) *mlpackage.Var {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	n := H * W
	h := chanNorm(g, x, path+".norm.gamma")
	qkvw := g.Get(path+".to_qkv.weight", 3*C, C)
	qkvb := g.Get(path+".to_qkv.bias", 3*C)
	head := func(j int) *mlpackage.Var {
		y := g.Conv2D(h, fmt.Sprintf("%s.to_qkv.%d", path, j), qkvw[j*C*C:(j+1)*C*C], qkvb[j*C:(j+1)*C], C, 1, 1)
		return g.Reshape(g.Transpose(g.Reshape(y, 1, C, n), 0, 2, 1), 1, 1, n, C)
	}
	o := g.QChunkAttention(head(0), head(1), head(2), 1/math.Sqrt(float64(C)), nil, qchunk)
	o = g.Reshape(g.Transpose(g.Reshape(o, 1, n, C), 0, 2, 1), 1, C, H, W)
	o = g.Conv2D(o, path+".proj", g.Get(path+".proj.weight", C, C), g.Get(path+".proj.bias", C), C, 1, 1)
	return g.Add(x, o)
}

// vaeUp is nearest 2x upsampling and the 3 x 3 convolution path.
func vaeUp(g *graph, x *mlpackage.Var, path string, cout int) *mlpackage.Var {
	C, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	u := g.Op("upsample_nearest_neighbor", map[string]any{"x": x, "scale_factor_height": g.B.ScalarI32(2),
		"scale_factor_width": g.B.ScalarI32(2)}, 1, C, 2*H, 2*W)
	return conv(g, u, path, cout)
}

// DupSource is DupUp3D of one frame (its first chunk): the input channel each (output channel, i,
// j) of a 2 x 2 pixel block takes, cout x 2 x 2 (iron_qwenimage.dup_source).
func DupSource(cin, cout, ft int) []int {
	rep := cout * ft * 4 / cin
	src := make([]int, cout*4)
	for co := 0; co < cout; co++ {
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				k := ((co*ft+(ft-1))*2+i)*2 + j
				src[co*4+i*2+j] = k / rep
			}
		}
	}
	return src
}

// dupUp is DupUp3D's shortcut of x (1, cin, H, W): (1, cout, 2 H, 2 W). When every pixel of a 2 x
// 2 block takes the same channel, it is a selection of the channels (none when it is the identity)
// then nearest 2x; otherwise each phase's selection, as one 1 x 1 convolution of 0s and 1s (exact
// in fp16) to 4 cout channels, put in place by pixel_shuffle.
func dupUp(g *graph, x *mlpackage.Var, path string, cout, ft int) *mlpackage.Var {
	cin, H, W := x.Shape[1], x.Shape[2], x.Shape[3]
	src := DupSource(cin, cout, ft)
	same, ident := true, cin == cout
	for co := 0; co < cout; co++ {
		for p := 1; p < 4; p++ {
			same = same && src[co*4+p] == src[co*4]
		}
		ident = ident && src[co*4] == co
	}
	if same {
		y := x
		if !ident {
			m := make([]float32, cout*cin)
			for co := 0; co < cout; co++ {
				m[co*cin+src[co*4]] = 1
			}
			y = g.Conv2D(x, path+".dup", m, nil, cout, 1, 1)
		}
		return g.Op("upsample_nearest_neighbor", map[string]any{"x": y, "scale_factor_height": g.B.ScalarI32(2),
			"scale_factor_width": g.B.ScalarI32(2)}, 1, cout, 2*H, 2*W)
	}
	m := make([]float32, 4*cout*cin)
	for r, s := range src { // row co * 4 + i * 2 + j: pixel_shuffle's channel order
		m[r*cin+s] = 1
	}
	y := g.Conv2D(x, path+".dup", m, nil, 4*cout, 1, 1)
	return g.Op("pixel_shuffle", map[string]any{"x": y, "upscale_factor": g.B.ScalarI32(2)}, 1, cout, 2*H, 2*W)
}
