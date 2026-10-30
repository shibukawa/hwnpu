package lfm3vl

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// tiny is a small LFM3-VL config: both layer kinds, GQA, two temporal slices.
const tiny = `{"architectures": ["Lfm3VlForConditionalGeneration"], "model_type": "lfm3_vl",
	"projector_hidden_act": "gelu", "projector_bias": true, "projector_layer_norm_eps": 1e-6, "projector_hidden_size": 64,
	"downsample_factor": 2,
	"text_config": {"model_type": "lfm3", "hidden_size": 48},
	"vision_config": {"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 3, "num_attention_heads": 4,
		"num_key_value_heads": 2, "head_dim": 8, "patch_size": 4, "temporal_patch_size": 2, "in_channels": 3,
		"conv_kernel_size": 3, "norm_eps": 1e-5, "layer_types": ["conv", "full_attention", "conv"],
		"rope_parameters": {"rope_theta": 100}}}`

func round16(v float64) float64 {
	return float64(mlpackage.Float16Value(mlpackage.Float16Bits(float32(v))))
}

func f16(t *testing.T, w llm.Weights, name string) []float64 {
	t.Helper()
	v, _, err := w.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = round16(float64(x))
	}
	return out
}

// reference is the vision encoder written out directly in float64 from llama.cpp's graph
// (tools/mtmd/models/lfm3vl.cpp): the patchifier over a still image given to every temporal
// slice, the embedding RMS norm, pre-norm layers (a 3x3 depthwise convolution of B*x gated by C,
// or attention with QK RMS norm and ggml's vision RoPE), SwiGLU, then the projector: LayerNorm
// per patch, 2x2 blocks merged (top row then bottom row), linear, exact GELU, linear. Constants
// that the program stores in fp16 (weights, rotary tables, epsilons, the attention scale) are
// rounded to fp16. patches are (n, 3*p*p) in raster order of a gw x gh grid.
func reference(t *testing.T, c *Config, w llm.Weights, patches []float64, gw, gh int) [][]float64 {
	v := c.Vision
	d, n, pd := v.HiddenSize, gw*gh, c.PatchDim()
	pp := v.PatchSize * v.PatchSize
	linear := func(name string, rows int, xs [][]float64, bias string) [][]float64 {
		m := f16(t, w, name)
		var b []float64
		if bias != "" {
			b = f16(t, w, bias)
		}
		out := make([][]float64, len(xs))
		for i, x := range xs {
			out[i] = make([]float64, rows)
			for r := 0; r < rows; r++ {
				for k, xv := range x {
					out[i][r] += m[r*len(x)+k] * xv
				}
				if b != nil {
					out[i][r] += b[r]
				}
			}
		}
		return out
	}
	rms := func(name string, xs [][]float64) [][]float64 {
		g := f16(t, w, name)
		out := make([][]float64, len(xs))
		for i, x := range xs {
			var ss float64
			for _, e := range x {
				ss += e * e
			}
			r := 1 / math.Sqrt(ss/float64(len(x))+round16(v.NormEps))
			out[i] = make([]float64, len(x))
			for j, e := range x {
				out[i][j] = e * r * g[j]
			}
		}
		return out
	}
	add := func(a, b [][]float64) [][]float64 {
		out := make([][]float64, len(a))
		for i := range a {
			out[i] = make([]float64, len(a[i]))
			for j := range a[i] {
				out[i][j] = a[i][j] + b[i][j]
			}
		}
		return out
	}
	silu := func(x float64) float64 { return x / (1 + math.Exp(-x)) }

	// patchifier: weight (d, 3, t, p, p), the same pixels in each temporal slice. The program
	// stores the slices' sum in fp16, so the sum is rounded here.
	pw, _, err := w.Get(tower + "patchifier.weight")
	if err != nil {
		t.Fatal(err)
	}
	h := make([][]float64, n)
	for i := range h {
		h[i] = make([]float64, d)
		for o := 0; o < d; o++ {
			for ch := 0; ch < 3; ch++ {
				for k := 0; k < pp; k++ {
					var wsum float32
					for ts := 0; ts < v.TemporalPatchSize; ts++ {
						wsum += pw[((o*3+ch)*v.TemporalPatchSize+ts)*pp+k]
					}
					h[i][o] += round16(float64(wsum)) * patches[i*pd+ch*pp+k]
				}
			}
		}
	}
	h = rms(tower+"embedding_norm.weight", h)

	for l, typ := range v.LayerTypes {
		p := fmt.Sprintf("%slayers.%d.", tower, l)
		x := rms(p+"operator_norm.weight", h)
		if typ == "conv" {
			bcx := linear(p+"conv.in_proj.weight", 3*d, x, "")
			cw := f16(t, w, p+"conv.spatial_conv.weight") // (d, 1, 3, 3)
			y := make([][]float64, n)
			for i := range y {
				py, px := i/gw, i%gw
				y[i] = make([]float64, d)
				for ch := 0; ch < d; ch++ {
					var acc float64
					for dy := -1; dy <= 1; dy++ {
						for dx := -1; dx <= 1; dx++ {
							qy, qx := py+dy, px+dx
							if qy < 0 || qy >= gh || qx < 0 || qx >= gw {
								continue
							}
							j := qy*gw + qx
							acc += cw[ch*9+(dy+1)*3+dx+1] * bcx[j][ch] * bcx[j][2*d+ch]
						}
					}
					y[i][ch] = bcx[i][d+ch] * acc
				}
			}
			x = linear(p+"conv.out_proj.weight", d, y, "")
		} else {
			a := p + "self_attn."
			nh, nkv, hd := v.NumHeads, v.NumKVHeads, v.HeadDim
			proj := func(name string, k int, norm string) [][][]float64 {
				flat := linear(a+name+"_proj.weight", k*hd, x, "")
				out := make([][][]float64, n)
				for i := range flat {
					out[i] = make([][]float64, k)
					for j := 0; j < k; j++ {
						out[i][j] = flat[i][j*hd : (j+1)*hd]
					}
					if norm == "" {
						continue
					}
					out[i] = rms(a+norm+".weight", out[i])
					// ggml vision RoPE: n_dims = hd/2, pairs (k, k+hd/2), theta_scale = base^(-2/n_dims),
					// the first hd/4 pairs by the row, the next by the column, each from its own start
					py, px := i/gw, i%gw
					for j := range out[i] {
						r := append([]float64(nil), out[i][j]...)
						for kk := 0; kk < hd/2; kk++ {
							pos, e := py, kk
							if kk >= hd/4 {
								pos, e = px, kk-hd/4
							}
							ang := float64(pos) * math.Pow(math.Pow(v.RopeParameters.Theta, -2.0/float64(hd/2)), float64(e))
							co, si := round16(math.Cos(ang)), round16(math.Sin(ang))
							x0, x1 := out[i][j][kk], out[i][j][kk+hd/2]
							r[kk], r[kk+hd/2] = x0*co-x1*si, x0*si+x1*co
						}
						out[i][j] = r
					}
				}
				return out
			}
			q, k, vv := proj("q", nh, "q_layernorm"), proj("k", nkv, "k_layernorm"), proj("v", nkv, "")
			o := make([][]float64, n)
			for i := range o {
				o[i] = make([]float64, nh*hd)
				for hh := 0; hh < nh; hh++ {
					kv := hh / (nh / nkv)
					s := make([]float64, n)
					mx := math.Inf(-1)
					for j := 0; j < n; j++ {
						for e := 0; e < hd; e++ {
							s[j] += q[i][hh][e] * k[j][kv][e]
						}
						s[j] *= round16(1 / math.Sqrt(float64(hd)))
						mx = math.Max(mx, s[j])
					}
					var sum float64
					for j := range s {
						s[j] = math.Exp(s[j] - mx)
						sum += s[j]
					}
					for j := range s {
						for e := 0; e < hd; e++ {
							o[i][hh*hd+e] += s[j] / sum * vv[j][kv][e]
						}
					}
				}
			}
			x = linear(a+"out_proj.weight", d, o, "")
		}
		h = add(h, x)
		f := rms(p+"ffn_norm.weight", h)
		g := linear(p+"feed_forward.w1.weight", v.IntermediateSize, f, "")
		u := linear(p+"feed_forward.w3.weight", v.IntermediateSize, f, "")
		for r := range g {
			for j := range g[r] {
				g[r][j] = silu(g[r][j]) * u[r][j]
			}
		}
		h = add(h, linear(p+"feed_forward.w2.weight", d, g, ""))
	}

	// projector
	lg, lb := f16(t, w, proj+"layer_norm.weight"), f16(t, w, proj+"layer_norm.bias")
	for i, x := range h {
		var mean, varr float64
		for _, e := range x {
			mean += e
		}
		mean /= float64(d)
		for _, e := range x {
			varr += (e - mean) * (e - mean)
		}
		varr /= float64(d)
		r := 1 / math.Sqrt(varr+round16(*c.ProjectorLNEps))
		for j, e := range x {
			h[i][j] = (e-mean)*r*lg[j] + lb[j]
		}
	}
	var merged [][]float64
	for by := 0; by < gh/2; by++ {
		for bx := 0; bx < gw/2; bx++ {
			var row []float64
			for dy := 0; dy < 2; dy++ {
				for dx := 0; dx < 2; dx++ {
					row = append(row, h[(2*by+dy)*gw+2*bx+dx]...)
				}
			}
			merged = append(merged, row)
		}
	}
	m := linear(proj+"linear_1.weight", c.ProjectorHiddenSize, merged, proj+"linear_1.bias")
	for i := range m {
		for j, e := range m[i] {
			m[i][j] = 0.5 * e * (1 + math.Erf(e/math.Sqrt2))
		}
	}
	return linear(proj+"linear_2.weight", c.Text.HiddenSize, m, proj+"linear_2.bias")
}

// scaled makes the synthetic weights act like trained ones: the projections 5 times larger and
// the norms' weights around 1 (3 for the QK norms, so that attention is sharp and depends on the
// positions). With weights in [-0.1, 0.1] and no norms after the layers, the layers' outputs
// would be a few percent of the patches' embeddings and attention nearly uniform, and a wrong
// layer or a wrong rotation would show as an error of 1e-4 or less.
type scaled struct{ llm.Synthetic }

func (w scaled) Get(name string) ([]float32, []int, error) {
	v, shape, err := w.Synthetic.Get(name)
	for i := range v {
		switch {
		case strings.HasSuffix(name, "layernorm.weight"):
			v[i] = 3 * (1 + v[i])
		case strings.HasSuffix(name, "norm.weight"):
			v[i] = 1 + v[i]
		case len(shape) >= 2 && !strings.Contains(name, "patchifier"):
			v[i] *= 5
		}
	}
	return v, shape, err
}

func run(t *testing.T, c *Config, w llm.Weights, opts llm.Options, x []float32) []float32 {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	ins, out, err := c.Build(b, w, opts)
	if err != nil {
		t.Fatal(err)
	}
	prog := b.Finish(&mlpackage.Var{Name: out.Name})
	if err := blobs.Close(); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	p, err := milcpu.Load(prog, f)
	if err != nil {
		t.Fatal(err)
	}
	res, err := p.Run(map[string]milcpu.Tensor{"x": {Shape: ins[0].Shape, Data: x}}, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return res[out.Name].Data
}

func relErr(got []float32, ref [][]float64) float64 {
	var se, sr float64
	w := len(ref[0])
	for i, row := range ref {
		for j, r := range row {
			e := float64(got[i*w+j]) - r
			se += e * e
			sr += r * r
		}
	}
	return math.Sqrt(se / sr)
}

// The program computes the reference's image embeddings for a non-square image, whole and one
// layer at a time.
func TestAgainstFP64(t *testing.T) {
	cfg, err := ParseConfig([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	w := scaled{llm.NewSynthetic(cfg)}
	const iw, ih = 24, 16 // 6 x 4 patches, 3 x 2 tokens
	gw, gh := iw/cfg.Vision.PatchSize, ih/cfg.Vision.PatchSize
	x := make([]float32, gw*gh*cfg.PatchDim())
	px := make([]float64, len(x))
	for i := range x {
		x[i] = float32(round16(math.Sin(float64(i)*0.37) * 1.5))
		px[i] = float64(x[i])
	}
	want := reference(t, cfg, w, px, gw, gh)
	if len(want) != cfg.Tokens(iw, ih) {
		t.Fatalf("%d tokens, Tokens says %d", len(want), cfg.Tokens(iw, ih))
	}
	opts := llm.Options{Format: "fp16", Params: map[string]string{"image": fmt.Sprintf("%dx%d", iw, ih)}}
	got := run(t, cfg, w, opts, x)
	if len(got) != len(want)*cfg.Text.HiddenSize {
		t.Fatalf("%d outputs", len(got))
	}
	e := relErr(got, want)
	if e > 1e-4 {
		t.Errorf("relative error %g against the reference", e)
	}
	t.Logf("relative error %.2g", e)
	h := x
	for i := 0; i < cfg.Vision.NumLayers; i++ {
		o := opts
		o.Start, o.Layers, o.NoHead = i, 1, true
		h = run(t, cfg, w, o, h)
	}
	o := opts
	o.Start = cfg.Vision.NumLayers
	if e := relErr(run(t, cfg, w, o, h), want); e > 1e-4 {
		t.Errorf("layer by layer: relative error %g against the reference", e)
	}
}

func TestConfig(t *testing.T) {
	cfg, err := ParseConfig([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	if w, h, err := cfg.ImageSize("512x256"); err != nil || w != 512 || h != 256 {
		t.Errorf("image size %d x %d, %v", w, h, err)
	}
	for _, s := range []string{"512", "500x500", "0x8", "8x-8"} {
		if _, _, err := cfg.ImageSize(s); err == nil {
			t.Errorf("image size %q accepted", s)
		}
	}
	for _, c := range []struct{ name, from, to string }{
		{"layer type", `"conv", "full_attention", "conv"]`, `"conv", "sliding_attention", "conv"]`},
		{"layer count", `"conv", "full_attention", "conv"]`, `"conv", "full_attention"]`},
		{"conv kernel", `"conv_kernel_size": 3`, `"conv_kernel_size": 5`},
		{"channels", `"in_channels": 3`, `"in_channels": 4`},
		{"projector act", `"projector_hidden_act": "gelu"`, `"projector_hidden_act": "relu"`},
		{"projector bias", `"projector_bias": true`, `"projector_bias": false`},
		{"downsample", `"downsample_factor": 2`, `"downsample_factor": 4`},
		{"model type", `"model_type": "lfm3_vl"`, `"model_type": "lfm3"`},
		{"temporal", `"temporal_patch_size": 2`, `"temporal_patch_size": 3`},
	} {
		if _, err := ParseConfig([]byte(strings.Replace(tiny, c.from, c.to, 1))); err == nil {
			t.Errorf("%s: unsupported config accepted", c.name)
		}
	}
}

// Resize agrees with PIL's antialiased bilinear and bicubic resizes (testdata from
// tools/ref/resize_golden.py) to within one level, larger and smaller.
func TestResizeMatchesPIL(t *testing.T) {
	f, err := os.Open("testdata/resize.png")
	if err != nil {
		t.Fatal(err)
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("testdata/resize.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Cases []struct {
			Filter        string `json:"filter"`
			Width, Height int
			RGB           string `json:"rgb"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	for _, c := range golden.Cases {
		want, _ := hex.DecodeString(c.RGB)
		got := Resize(img, c.Width, c.Height, c.Filter)
		off := 0
		for i := range want {
			d := int(got[i]) - int(want[i])
			if d > 1 || d < -1 {
				t.Errorf("%s %dx%d: byte %d is %d, PIL %d", c.Filter, c.Width, c.Height, i, got[i], want[i])
				break
			}
			if d != 0 {
				off++
			}
		}
		if off > len(want)/20 {
			t.Errorf("%s %dx%d: %d of %d bytes off by one", c.Filter, c.Width, c.Height, off, len(want))
		}
	}
}

func TestProcessor(t *testing.T) {
	p := &Processor{Mean: [3]float64{0.5, 0.4, 0.3}, Std: [3]float64{0.2, 0.25, 0.5}, Rescale: 1.0 / 255, Resample: "bilinear",
		MinPixels: 32 * 32 * 4, MaxPixels: 512 * 512, PatchSize: 16, DoNorm: true, DoRescaled: true}
	for _, c := range []struct{ w, h, wantW, wantH int }{
		{1000, 500, 704, 352}, // over the maximum: scaled down to fit
		{500, 300, 512, 288},  // rounded to multiples of 32
		{40, 40, 64, 64},      // under the minimum: scaled up
		{10, 300, 32, 352},    // an edge rounded to 0, then scaled up
	} {
		if w, h := p.SmartResize(c.w, c.h); w != c.wantW || h != c.wantH {
			t.Errorf("SmartResize(%d, %d) = %d x %d, want %d x %d", c.w, c.h, w, h, c.wantW, c.wantH)
		}
	}
	// an image already at the size: each patch holds its pixels channel by channel, normalized
	img := image.NewNRGBA(image.Rect(0, 0, 32, 16))
	for y := 0; y < 16; y++ {
		for x := 0; x < 32; x++ {
			img.Set(x, y, color.NRGBA{uint8(x * 7), uint8(y * 13), uint8(x + y), 255})
		}
	}
	out := p.Patches(img, 32, 16)
	dim := 3 * 16 * 16
	if len(out) != 2*dim {
		t.Fatalf("%d values", len(out))
	}
	// patch 1 (x 16..31), channel 1 (green), row 5, column 3: pixel (19, 5)
	got := out[dim+(1*16+5)*16+3]
	want := (float64(5*13)/255 - 0.4) / 0.25
	if math.Abs(float64(got)-want) > 1e-6 {
		t.Errorf("patch value %g, want %g", got, want)
	}
}
