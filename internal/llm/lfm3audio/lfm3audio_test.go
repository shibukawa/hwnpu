package lfm3audio

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// tiny is a small LFM3-Audio config: 16 mel bins, 4 downsampler channels, both layer kinds, GQA.
const tiny = `{"model_type": "lfm3_audio", "lfm": {"model_type": "lfm3", "hidden_size": 48}, "adapter_hidden_size": 40,
	"encoder": {"dim_in": 16, "sliding_window": 3, "lfm": {"hidden_size": 32, "num_hidden_layers": 3, "num_attention_heads": 4,
		"num_key_value_heads": 2, "intermediate_size": 48, "block_auto_adjust_ff_dim": true, "block_ffn_dim_multiplier": 1.0,
		"block_multiple_of": 16, "norm_eps": 1e-5, "conv_L_cache": 3, "conv_bias": false,
		"layer_types": ["conv", "full_attention", "conv"], "rope_theta": 1000000.0, "subsampling_conv_channels": 4}}}`

// tinyOmni is the same encoder in an LFM3-Omni config.
const tinyOmni = `{"model_type": "lfm3_omni", "text_config": {"model_type": "lfm3", "hidden_size": 48}, "adapter_hidden_size": 40,
	"audio_encoder_config": {"num_mel_bins": 16, "sliding_window": 3, "text_config": {"hidden_size": 32, "num_hidden_layers": 3,
		"num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 48, "block_auto_adjust_ff_dim": true,
		"block_ffn_dim_multiplier": 1.0, "block_multiple_of": 16, "norm_eps": 1e-5, "conv_L_cache": 3, "conv_bias": false,
		"layer_types": ["conv", "full_attention", "conv"], "rope_parameters": {"rope_theta": 1000000.0}, "subsampling_conv_channels": 4}}}`

func round16(v float64) float64 {
	return float64(mlpackage.Float16Value(mlpackage.Float16Bits(float32(v))))
}

// scaled makes the synthetic weights act like trained ones (see lfm3vl's test): matrices and
// kernels 5 times larger, norms around 1 (3 for the QK norms).
type scaled struct{ llm.Synthetic }

func (w scaled) Get(name string) ([]float32, []int, error) {
	v, shape, err := w.Synthetic.Get(name)
	for i := range v {
		switch {
		case strings.HasSuffix(name, "layernorm.weight"):
			v[i] = 3 * (1 + v[i])
		case strings.HasSuffix(name, "norm.weight") || strings.HasSuffix(name, "model.0.weight") || strings.HasSuffix(name, "layer_norm.weight"):
			v[i] = 1 + v[i]
		case len(shape) >= 2:
			v[i] *= 5
		}
	}
	return v, shape, err
}

func (w scaled) Rows(name string, begin, end int) ([]float32, error) {
	v, shape, err := w.Get(name)
	if err != nil {
		return nil, err
	}
	return v[begin*shape[1] : end*shape[1]], nil
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

// reference is the audio encoder written out directly in float64 from llama.cpp's graph
// (tools/mtmd/models/lfm3a.cpp with a zero state): mel is (frames, mel bins) of the clip alone.
// Each downsampler convolution is a PyTorch style stride 2, padding 1 convolution of the
// (channels, mel, time) array with the kernel indexed [out][in][mel][time]; the flattening is
// channel major; the stack is causal with attention over the last Window positions.
func reference(t *testing.T, c *Config, w llm.Weights, mel [][]float64) [][]float64 {
	e := c.Enc
	ch, d := e.SubsamplingChannels, e.HiddenSize
	// x[c][f][t]
	x := [][][]float64{make([][]float64, c.MelBins)}
	for f := range x[0] {
		x[0][f] = make([]float64, len(mel))
		for tt := range mel {
			x[0][f][tt] = mel[tt][f]
		}
	}
	// conv_in: one input channel, kernel (ch, 1, 3, 3)
	conv := func(x [][][]float64, name string) [][][]float64 {
		k, b := f16(t, w, name+".weight"), f16(t, w, name+".bias")
		F, T := len(x[0]), len(x[0][0])
		F2, T2 := (F-1)/2+1, (T-1)/2+1
		out := make([][][]float64, ch)
		for o := range out {
			out[o] = make([][]float64, F2)
			for f := range out[o] {
				out[o][f] = make([]float64, T2)
				for tt := range out[o][f] {
					acc := b[o]
					for kf := 0; kf < 3; kf++ {
						for kt := 0; kt < 3; kt++ {
							sf, st := 2*f+kf-1, 2*tt+kt-1
							if sf >= 0 && sf < F && st >= 0 && st < T {
								acc += k[(o*3+kf)*3+kt] * x[0][sf][st]
							}
						}
					}
					out[o][f][tt] = acc
				}
			}
		}
		return out
	}
	// conv_mid and conv_end: depthwise, kernel (ch, 1, 3, 3)
	dw := func(x [][][]float64, name string) [][][]float64 {
		k, b := f16(t, w, name+".weight"), f16(t, w, name+".bias")
		F, T := len(x[0]), len(x[0][0])
		F2, T2 := (F-1)/2+1, (T-1)/2+1
		out := make([][][]float64, ch)
		for o := range out {
			out[o] = make([][]float64, F2)
			for f := range out[o] {
				out[o][f] = make([]float64, T2)
				for tt := range out[o][f] {
					acc := b[o]
					for kf := 0; kf < 3; kf++ {
						for kt := 0; kt < 3; kt++ {
							sf, st := 2*f+kf-1, 2*tt+kt-1
							if sf >= 0 && sf < F && st >= 0 && st < T {
								acc += k[(o*3+kf)*3+kt] * x[o][sf][st]
							}
						}
					}
					out[o][f][tt] = acc
				}
			}
		}
		return out
	}
	pw := func(x [][][]float64, name string) [][][]float64 {
		k, b := f16(t, w, name+".weight"), f16(t, w, name+".bias")
		out := make([][][]float64, ch)
		for o := range out {
			out[o] = make([][]float64, len(x[0]))
			for f := range out[o] {
				out[o][f] = make([]float64, len(x[0][0]))
				for tt := range out[o][f] {
					acc := b[o]
					for i := range x {
						acc += k[o*ch+i] * x[i][f][tt]
					}
					out[o][f][tt] = acc
				}
			}
		}
		return out
	}
	silu := func(x [][][]float64) [][][]float64 {
		for _, a := range x {
			for _, r := range a {
				for i, v := range r {
					r[i] = v / (1 + math.Exp(-v))
				}
			}
		}
		return x
	}
	p := c.down
	h3 := silu(conv(x, p+"conv_in"))
	h3 = silu(pw(dw(h3, p+"conv_mid"), p+"lin_mid"))
	h3 = silu(pw(dw(h3, p+"conv_end"), p+"lin_end"))
	n, F3 := len(h3[0][0]), len(h3[0])
	lw, lb := f16(t, w, p+"lin_out.weight"), f16(t, w, p+"lin_out.bias")
	h := make([][]float64, n)
	for tt := range h {
		h[tt] = make([]float64, d)
		for o := range h[tt] {
			acc := lb[o]
			for cc := 0; cc < ch; cc++ {
				for f := 0; f < F3; f++ {
					acc += lw[o*ch*F3+cc*F3+f] * h3[cc][f][tt]
				}
			}
			h[tt][o] = acc
		}
	}

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
			for _, v := range x {
				ss += v * v
			}
			r := 1 / math.Sqrt(ss/float64(len(x))+round16(e.NormEps))
			out[i] = make([]float64, len(x))
			for j, v := range x {
				out[i][j] = v * r * g[j]
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
	for l, typ := range e.LayerTypes {
		lp := fmt.Sprintf("%slayers.%d.", c.stack, l)
		x := rms(lp+"operator_norm.weight", h)
		if typ == "conv" {
			bcx := linear(lp+"conv.in_proj.weight", 3*d, x, "")
			cw := f16(t, w, lp+"conv.conv.weight")
			k := e.ConvKernel
			y := make([][]float64, n)
			for i := range y {
				y[i] = make([]float64, d)
				for cc := 0; cc < d; cc++ {
					var acc float64
					for j := 0; j < k; j++ {
						if src := i + j - (k - 1); src >= 0 {
							acc += cw[cc*k+j] * bcx[src][cc] * bcx[src][2*d+cc]
						}
					}
					y[i][cc] = bcx[i][d+cc] * acc
				}
			}
			x = linear(lp+"conv.out_proj.weight", d, y, "")
		} else {
			a := lp + "self_attn."
			nh, nkv, hd := e.NumHeads, e.NumKVHeads, e.HeadDim
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
					for j := range out[i] {
						r := append([]float64(nil), out[i][j]...)
						for f := 0; f < hd/2; f++ {
							ang := float64(i) * math.Pow(e.RopeTheta, -float64(2*f)/float64(hd))
							co, si := round16(math.Cos(ang)), round16(math.Sin(ang))
							x0, x1 := out[i][j][f], out[i][j][f+hd/2]
							r[f], r[f+hd/2] = x0*co-x1*si, x1*co+x0*si
						}
						out[i][j] = r
					}
				}
				return out
			}
			q, k, v := proj("q", nh, "q_layernorm"), proj("k", nkv, "k_layernorm"), proj("v", nkv, "")
			o := make([][]float64, n)
			for i := range o {
				o[i] = make([]float64, nh*hd)
				for hh := 0; hh < nh; hh++ {
					kv := hh / (nh / nkv)
					lo := max(0, i-c.Window+1)
					s := make([]float64, i+1)
					mx := math.Inf(-1)
					for j := lo; j <= i; j++ {
						for f := 0; f < hd; f++ {
							s[j] += q[i][hh][f] * k[j][kv][f]
						}
						s[j] *= round16(1 / math.Sqrt(float64(hd)))
						mx = math.Max(mx, s[j])
					}
					var sum float64
					for j := lo; j <= i; j++ {
						s[j] = math.Exp(s[j] - mx)
						sum += s[j]
					}
					for j := lo; j <= i; j++ {
						for f := 0; f < hd; f++ {
							o[i][hh*hd+f] += s[j] / sum * v[j][kv][f]
						}
					}
				}
			}
			x = linear(a+"out_proj.weight", d, o, "")
		}
		h = add(h, x)
		f := rms(lp+"ffn_norm.weight", h)
		g := linear(lp+"feed_forward.w1.weight", e.FFDim(), f, "")
		u := linear(lp+"feed_forward.w3.weight", e.FFDim(), f, "")
		for r := range g {
			for j := range g[r] {
				g[r][j] = g[r][j] / (1 + math.Exp(-g[r][j])) * u[r][j]
			}
		}
		h = add(h, linear(lp+"feed_forward.w2.weight", d, g, ""))
	}
	h = rms(c.stack+"embedding_norm.weight", h)
	lg, lbb := f16(t, w, c.norm+"weight"), f16(t, w, c.norm+"bias")
	for i, x := range h {
		var mean, varr float64
		for _, v := range x {
			mean += v
		}
		mean /= float64(d)
		for _, v := range x {
			varr += (v - mean) * (v - mean)
		}
		varr /= float64(d)
		r := 1 / math.Sqrt(varr+round16(1e-5))
		for j, v := range x {
			h[i][j] = (v-mean)*r*lg[j] + lbb[j]
		}
	}
	m := linear(c.lin1+"weight", c.ProjectorHidden, h, c.lin1+"bias")
	for i := range m {
		for j, v := range m[i] {
			m[i][j] = 0.5 * v * (1 + math.Erf(v/math.Sqrt2))
		}
	}
	return linear(c.lin2+"weight", c.TextHidden, m, c.lin2+"bias")
}

func run(t *testing.T, c *Config, w llm.Weights, opts llm.Options, in map[string][]float32) []float32 {
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
	feed := map[string]milcpu.Tensor{}
	for _, i := range ins {
		feed[i.Name] = milcpu.Tensor{Shape: i.Shape, Data: in[i.Name]}
	}
	res, err := p.Run(feed, milcpu.Options{})
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

// The program computes the reference's embeddings of a clip shorter than the program (its frames
// followed by zeros, masked), whole and one layer at a time, from both checkpoint layouts.
func TestAgainstFP64(t *testing.T) {
	for _, js := range []string{tiny, tinyOmni} {
		cfg, err := ParseConfig([]byte(js))
		if err != nil {
			t.Fatal(err)
		}
		w := scaled{llm.NewSynthetic(cfg)}
		const clip, frames = 37, 50
		mel := make([][]float64, clip)
		x := make([]float32, frames*cfg.MelBins)
		mask := make([]float32, frames)
		for i := range mel {
			mask[i] = 1
			mel[i] = make([]float64, cfg.MelBins)
			for j := range mel[i] {
				v := float32(round16(math.Sin(float64(i*cfg.MelBins+j)*0.61) * 1.2))
				x[i*cfg.MelBins+j] = v
				mel[i][j] = float64(v)
			}
		}
		want := reference(t, cfg, w, mel)
		if len(want) != Tokens(clip) {
			t.Fatalf("%d reference rows for %d tokens", len(want), Tokens(clip))
		}
		opts := llm.Options{Format: "fp16", Params: map[string]string{"frames": fmt.Sprint(frames)}}
		got := run(t, cfg, w, opts, map[string][]float32{"x": x, "mask": mask})
		if len(got) != Tokens(frames)*cfg.TextHidden {
			t.Fatalf("%d outputs", len(got))
		}
		e := relErr(got, want)
		if e > 1e-4 {
			t.Errorf("omni %v: relative error %g against the reference", cfg.Omni, e)
		}
		t.Logf("omni %v: relative error %.2g", cfg.Omni, e)
		h := run(t, cfg, w, llm.Options{Format: "fp16", Params: opts.Params, Layers: 1, NoHead: true}, map[string][]float32{"x": x, "mask": mask})
		for i := 1; i < cfg.Enc.NumLayers; i++ {
			o := opts
			o.Start, o.Layers, o.NoHead = i, 1, true
			h = run(t, cfg, w, o, map[string][]float32{"x": h})
		}
		o := opts
		o.Start = cfg.Enc.NumLayers
		if e := relErr(run(t, cfg, w, o, map[string][]float32{"x": h}), want); e > 1e-4 {
			t.Errorf("omni %v: layer by layer: relative error %g against the reference", cfg.Omni, e)
		}
	}
}

func TestConfig(t *testing.T) {
	a, err := ParseConfig([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	o, err := ParseConfig([]byte(tinyOmni))
	if err != nil {
		t.Fatal(err)
	}
	if a.Enc.FFDim() != 32 || a.Omni || !o.Omni || a.ProjectorHidden != 40 {
		t.Errorf("ff %d, omni %v %v, projector %d", a.Enc.FFDim(), a.Omni, o.Omni, a.ProjectorHidden)
	}
	if _, ok := o.Tensors()["model.audio_encoder.lfm.layers.1.self_attn.q_proj.weight"]; !ok {
		t.Error("LFM3-Omni's encoder is not under model.audio_encoder.lfm.")
	}
	if _, ok := a.Tensors()["encoder.lfm.lfm.layers.1.self_attn.q_proj.weight"]; !ok {
		t.Error("LFM3-Audio's encoder is not under encoder.lfm.lfm.")
	}
	if Tokens(3000) != 375 || Tokens(37) != 5 || Tokens(1) != 1 {
		t.Errorf("tokens %d %d %d", Tokens(3000), Tokens(37), Tokens(1))
	}
	for _, c := range []struct{ name, from, to string }{
		{"rope theta", `"rope_theta": 1000000.0`, `"rope_theta": 10000.0`},
		{"layer type", `["conv", "full_attention", "conv"]`, `["conv", "mamba", "conv"]`},
		{"layer count", `["conv", "full_attention", "conv"]`, `["conv", "full_attention"]`},
		{"conv bias", `"conv_bias": false`, `"conv_bias": true`},
		{"mel bins", `"dim_in": 16`, `"dim_in": 20`},
		{"window", `"sliding_window": 3`, `"sliding_window": 0`},
		{"no encoder", `"encoder"`, `"decoder"`},
	} {
		if _, err := ParseConfig([]byte(strings.Replace(tiny, c.from, c.to, 1))); err == nil {
			t.Errorf("%s: unsupported config accepted", c.name)
		}
	}
}

// Features agrees with librosa's STFT and Slaney filter bank (testdata from
// tools/ref/mel_golden.py), for a clip of several frames and one shorter than a window.
func TestFeaturesMatchLibrosa(t *testing.T) {
	raw, err := os.ReadFile("testdata/mel.json")
	if err != nil {
		t.Fatal(err)
	}
	var golden struct {
		Cases []struct {
			Name    string      `json:"name"`
			Samples []float32   `json:"samples"`
			Frames  [][]float32 `json:"frames"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	for _, c := range golden.Cases {
		got, n := Features(c.Samples, 128)
		if n != len(c.Frames) {
			t.Errorf("%s: %d frames, librosa %d", c.Name, n, len(c.Frames))
			continue
		}
		var worst float64
		for i, row := range c.Frames {
			for j, v := range row {
				worst = math.Max(worst, math.Abs(float64(got[i*128+j]-v)))
			}
		}
		if worst > 1e-4 {
			t.Errorf("%s: largest difference %g from librosa", c.Name, worst)
		}
	}
}

func wav(rate, channels, bits int, frames [][]float64) []byte {
	var data bytes.Buffer
	for _, fr := range frames {
		for _, v := range fr {
			switch bits {
			case 16:
				binary.Write(&data, binary.LittleEndian, int16(math.Round(v*32767)))
			case 32:
				binary.Write(&data, binary.LittleEndian, float32(v))
			}
		}
	}
	format := uint16(1)
	if bits == 32 {
		format = 3
	}
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+data.Len()))
	b.WriteString("WAVEfmt ")
	for _, v := range []any{uint32(16), format, uint16(channels), uint32(rate), uint32(rate * channels * bits / 8), uint16(channels * bits / 8), uint16(bits)} {
		binary.Write(&b, binary.LittleEndian, v)
	}
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(data.Len()))
	b.Write(data.Bytes())
	return b.Bytes()
}

func TestReadWAV(t *testing.T) {
	// stereo 16 bit at 16 kHz: the channels are averaged
	got, err := ReadWAV(wav(16000, 2, 16, [][]float64{{0.5, 0.1}, {-0.25, -0.25}, {1, 0}}))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 3 || math.Abs(float64(got[0])-0.3) > 1e-3 || math.Abs(float64(got[1])+0.25) > 1e-3 {
		t.Errorf("samples %v", got)
	}
	// a 1 kHz tone at 48 kHz in float: a third as many samples, the same tone
	var frames [][]float64
	for i := 0; i < 4800; i++ {
		frames = append(frames, []float64{0.5 * math.Sin(2*math.Pi*1000*float64(i)/48000)})
	}
	got, err = ReadWAV(wav(48000, 1, 32, frames))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1600 {
		t.Fatalf("%d samples at 16 kHz", len(got))
	}
	var worst float64
	for i := 200; i < 1400; i++ {
		worst = math.Max(worst, math.Abs(float64(got[i])-0.5*math.Sin(2*math.Pi*1000*float64(i)/16000)))
	}
	if worst > 0.01 {
		t.Errorf("resampled tone off by %g", worst)
	}
	if _, err := ReadWAV([]byte("RIFF....WAVE")); err == nil {
		t.Error("a WAV without chunks read")
	}
}
