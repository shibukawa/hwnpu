package lfm3gen

import (
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

const tinyAudio = `{"model_type": "lfm3_audio", "lfm": {"hidden_size": 48}, "codebooks": 4, "audio_vocab_size": 17, "maskgit_steps": 2,
	"depthformer": {"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
		"norm_eps": 1e-5, "layer_types": ["full_attention", "full_attention"], "use_sandwich_norm": false, "is_causal": false,
		"rope_parameters": {"full_attention": {"rope_theta": 1000000.0}}}}`

const tinyOmni = `{"model_type": "lfm3_omni", "text_config": {"hidden_size": 48}, "num_codebooks": 4, "codebook_size": 17, "maskgit_schedule": [1, 3],
	"depth_config": {"hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
		"norm_eps": 1e-5, "layer_types": ["full_attention", "full_attention"], "use_sandwich_norm": false, "is_causal": false,
		"rope_parameters": {"full_attention": {"rope_theta": 1000000.0}}}}`

const tinyCodec = `{"model_type": "liquid_codec", "fsq_levels": [4, 3], "n_q": 3, "quantizer_dim": 24, "encoder_dim": 512,
	"stft_n_fft": 16, "stft_hop_length": 4, "stft_win_length": 16, "stft_frame_stack": 2, "decoder_window": 3, "sampling_rate": 24000}`

func round16(v float64) float64 {
	return float64(mlpackage.Float16Value(mlpackage.Float16Bits(float32(v))))
}

// scaled makes the synthetic weights act like trained ones: matrices 5 times larger, norms
// around 1, the QK norms around 3 (see lfm3vl's test).
type scaled struct{ llm.Synthetic }

func (w scaled) Get(name string) ([]float32, []int, error) {
	v, shape, err := w.Synthetic.Get(name)
	for i := range v {
		switch {
		case strings.HasSuffix(name, "q_norm.weight") || strings.HasSuffix(name, "k_norm.weight"):
			v[i] = 3 * (1 + v[i])
		case strings.HasSuffix(name, "norm.weight"):
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

func linear(t *testing.T, w llm.Weights, name string, rows int, xs [][]float64, bias string) [][]float64 {
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

func rms(t *testing.T, w llm.Weights, name string, eps float64, xs [][]float64) [][]float64 {
	var g []float64
	if name != "" {
		g = f16(t, w, name)
	}
	out := make([][]float64, len(xs))
	for i, x := range xs {
		var ss float64
		for _, v := range x {
			ss += v * v
		}
		r := 1 / math.Sqrt(ss/float64(len(x))+round16(eps))
		out[i] = make([]float64, len(x))
		for j, v := range x {
			out[i][j] = v * r
			if g != nil {
				out[i][j] *= g[j]
			}
		}
	}
	return out
}

// lfmStack is an LFM3 stack written out in float64 from llama.cpp's lfm3a-gen.cpp
// (build_lfm_layer, build_lfm_attn, build_lfm_conv): pre-norm layers, with sandwich norms if the
// stack has them; attention with QK RMS norm, rotate_half RoPE over the whole head at positions
// 0.., causal over the last window positions (or bidirectional) and a sigmoid gate per head;
// the short convolution causal from zeros; SwiGLU.
func lfmStack(t *testing.T, w llm.Weights, st stack, h [][]float64) [][]float64 {
	n, d := len(h), st.d
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
	for l, typ := range st.layerTypes {
		p := fmt.Sprintf("%slayers.%d.", st.prefix, l)
		x := rms(t, w, p+"operator_norm.weight", st.eps, h)
		if typ == "conv" {
			bcx := linear(t, w, p+"conv.in_proj.weight", 3*d, x, "")
			cw := f16(t, w, p+"conv.conv.weight")
			k := st.convKernel
			y := make([][]float64, n)
			for i := range y {
				y[i] = make([]float64, d)
				for c := 0; c < d; c++ {
					var acc float64
					for j := 0; j < k; j++ {
						if src := i + j - (k - 1); src >= 0 {
							acc += cw[c*k+j] * bcx[src][c] * bcx[src][2*d+c]
						}
					}
					y[i][c] = bcx[i][d+c] * acc
				}
			}
			x = linear(t, w, p+"conv.out_proj.weight", d, y, "")
		} else {
			a := p + "self_attn."
			nh, nkv, hd := st.heads, st.kvHeads, st.headDim
			heads := func(name string, k int, norm string) [][][]float64 {
				flat := linear(t, w, a+name+"_proj.weight", k*hd, x, "")
				out := make([][][]float64, n)
				for i := range flat {
					out[i] = make([][]float64, k)
					for j := 0; j < k; j++ {
						out[i][j] = flat[i][j*hd : (j+1)*hd]
					}
					if norm == "" {
						continue
					}
					out[i] = rms(t, w, a+norm+".weight", st.eps, out[i])
					for j := range out[i] {
						r := append([]float64(nil), out[i][j]...)
						for f := 0; f < hd/2; f++ {
							ang := float64(i) * math.Pow(st.ropeTheta, -float64(2*f)/float64(hd))
							co, si := round16(math.Cos(ang)), round16(math.Sin(ang))
							x0, x1 := out[i][j][f], out[i][j][f+hd/2]
							r[f], r[f+hd/2] = x0*co-x1*si, x1*co+x0*si
						}
						out[i][j] = r
					}
				}
				return out
			}
			q, kk, v := heads("q", nh, "q_norm"), heads("k", nkv, "k_norm"), heads("v", nkv, "")
			gate := linear(t, w, a+"gate_proj.weight", nh, x, "")
			o := make([][]float64, n)
			for i := range o {
				o[i] = make([]float64, nh*hd)
				lo, hi := 0, n-1
				if st.window > 0 {
					lo, hi = max(0, i-st.window+1), i
				}
				for hh := 0; hh < nh; hh++ {
					kv := hh / (nh / nkv)
					s := make([]float64, n)
					mx := math.Inf(-1)
					for j := lo; j <= hi; j++ {
						for f := 0; f < hd; f++ {
							s[j] += q[i][hh][f] * kk[j][kv][f]
						}
						s[j] *= round16(1 / math.Sqrt(float64(hd)))
						mx = math.Max(mx, s[j])
					}
					var sum float64
					for j := lo; j <= hi; j++ {
						s[j] = math.Exp(s[j] - mx)
						sum += s[j]
					}
					g := 1 / (1 + math.Exp(-gate[i][hh]))
					for j := lo; j <= hi; j++ {
						for f := 0; f < hd; f++ {
							o[i][hh*hd+f] += s[j] / sum * v[j][kv][f] * g
						}
					}
				}
			}
			x = linear(t, w, a+"o_proj.weight", d, o, "")
		}
		if st.sandwich {
			x = rms(t, w, p+"post_operator_norm.weight", st.eps, x)
		}
		h = add(h, x)
		f := rms(t, w, p+"ffn_norm.weight", st.eps, h)
		g := linear(t, w, p+"feed_forward.w1.weight", st.ff, f, "")
		u := linear(t, w, p+"feed_forward.w3.weight", st.ff, f, "")
		for r := range g {
			for j := range g[r] {
				g[r][j] = g[r][j] / (1 + math.Exp(-g[r][j])) * u[r][j]
			}
		}
		f = linear(t, w, p+"feed_forward.w2.weight", d, g, "")
		if st.sandwich {
			f = rms(t, w, p+"post_ffn_norm.weight", st.eps, f)
		}
		h = add(h, f)
	}
	return h
}

func run(t *testing.T, m llm.Model, w llm.Weights, opts llm.Options, in map[string][]float32) []float32 {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	ins, out, err := m.Build(b, w, opts)
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

func wave(n int, f float64) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(round16(math.Sin(float64(i)*f) * 0.8))
	}
	return v
}

// A MaskGIT pass computes the reference's logits (llama.cpp's build_gen_code): the projected
// hidden state plus each codebook's input embedding, the bidirectional stack, the final norm,
// each codebook's norm, its code table.
func TestDepthAgainstFP64(t *testing.T) {
	for _, js := range []string{tinyAudio, tinyOmni} {
		c, err := ParseDepth([]byte(js))
		if err != nil {
			t.Fatal(err)
		}
		w := scaled{llm.NewSynthetic(c)}
		n, d := c.Codebooks, c.st.d
		x, h := wave(n*d, 0.37), wave(c.TextHidden, 0.11)
		hv := make([]float64, len(h))
		for i, v := range h {
			hv[i] = float64(v)
		}
		p := linear(t, w, c.linear+"weight", n*d, [][]float64{hv}, c.linear+"bias")[0]
		cur := make([][]float64, n)
		for cb := range cur {
			cur[cb] = make([]float64, d)
			for j := range cur[cb] {
				cur[cb][j] = p[cb*d+j] + float64(x[cb*d+j])
			}
		}
		cur = lfmStack(t, w, c.st, cur)
		cur = rms(t, w, c.st.prefix+"embedding_norm.weight", c.st.eps, cur)
		want := make([][]float64, n)
		for cb := range want {
			row := rms(t, w, fmt.Sprintf("%s%d.embedding_norm.weight", c.embeddings, cb), c.st.eps, cur[cb:cb+1])[0]
			e := f16(t, w, c.Table(cb))
			want[cb] = make([]float64, c.Vocab)
			for v := range want[cb] {
				for j := range d {
					want[cb][v] += e[v*d+j] * row[j]
				}
			}
		}
		got := run(t, c, w, llm.Options{Format: "fp16"}, map[string][]float32{"x": x, "h": h})
		e := relErr(got, want)
		if e > 1e-4 {
			t.Errorf("omni %v: relative error %g against the reference", c.Omni, e)
		}
		t.Logf("omni %v: relative error %.2g, schedule %v", c.Omni, e, c.Schedule)
	}
}

// The codec decoder computes the reference's STFT frames (llama.cpp's build_gen_wav from a zero
// state): the code vectors' two projections, each frame repeated FrameStack times, the causal
// stack with sandwich norms and a window, the output norm and projection.
func TestCodecAgainstFP64(t *testing.T) {
	c, err := ParseCodec([]byte(tinyCodec))
	if err != nil {
		t.Fatal(err)
	}
	c.SetLayers([]string{"conv", "full_attention", "conv", "full_attention", "conv", "conv", "full_attention", "conv"}, 8, 4, 16, 64, 3)
	w := scaled{llm.NewSynthetic(c)}
	const frames = 7
	codes := make([]int, frames*c.Codebooks)
	x := make([]float32, 0, frames*c.CodeDim())
	for i := range codes {
		codes[i] = (i*5 + 3) % 12
		x = append(x, c.Codes(codes[i])...)
	}
	q := make([][]float64, frames)
	for f := range q {
		for _, v := range x[f*c.CodeDim() : (f+1)*c.CodeDim()] {
			q[f] = append(q[f], float64(v))
		}
	}
	h := linear(t, w, "quantizer.fsq.project_out.weight", c.QuantDim, q, "quantizer.fsq.project_out.bias")
	h = linear(t, w, "from_quantizer.weight", codecWidth, h, "from_quantizer.bias")
	var up [][]float64
	for _, row := range h {
		for range c.FrameStack {
			up = append(up, row)
		}
	}
	up = lfmStack(t, w, c.st, up)
	up = rms(t, w, "decoder.lfm.embedding_norm.weight", codecEps, up)
	want := linear(t, w, "decoder.lin.weight", 2*c.Bins(), up, "decoder.lin.bias")
	got := run(t, c, w, llm.Options{Format: "fp16", Params: map[string]string{"codec-frames": fmt.Sprint(frames)}}, map[string][]float32{"x": x})
	e := relErr(got, want)
	if e > 1e-4 {
		t.Errorf("relative error %g against the reference", e)
	}
	t.Logf("relative error %.2g", e)
	if lb := c.Lookback(); lb != 5*2+3*2 {
		t.Errorf("lookback %d", lb)
	}
}

func TestConfigs(t *testing.T) {
	c, err := ParseCodec([]byte(tinyCodec))
	if err != nil {
		t.Fatal(err)
	}
	// levels 4 and 3: code = l0 + 4*l1, values (l - half) / half
	if got := c.Codes(7); got[0] != 0.5 || got[1] != 0 {
		t.Errorf("code 7: %v", got)
	}
	a, err := ParseDepth([]byte(tinyAudio))
	if err != nil {
		t.Fatal(err)
	}
	if len(a.Schedule) != 2 || a.Schedule[0] != 2 || a.TopK != 128 || a.Feedback() != "codebook_embedding.weight" {
		t.Errorf("audio: schedule %v, top-k %d, feedback %s", a.Schedule, a.TopK, a.Feedback())
	}
	o, err := ParseDepth([]byte(tinyOmni))
	if err != nil {
		t.Fatal(err)
	}
	if !o.Omni || o.Table(1) != "model.depth_decoder.depth_embeddings.1.embedding.weight" {
		t.Errorf("omni table %s", o.Table(1))
	}
	for _, c := range []struct{ name, from, to string }{
		{"causal", `"is_causal": false`, `"is_causal": true`},
		{"sandwich", `"use_sandwich_norm": false`, `"use_sandwich_norm": true`},
		{"rope", `"rope_theta": 1000000.0`, `"rope_theta": 10000.0`},
		{"schedule", `"maskgit_steps": 2`, `"maskgit_steps": 3`},
		{"conv layer", `["full_attention", "full_attention"]`, `["conv", "full_attention"]`},
	} {
		if _, err := ParseDepth([]byte(strings.Replace(tinyAudio, c.from, c.to, 1))); err == nil {
			t.Errorf("%s: unsupported depthformer accepted", c.name)
		}
	}
	if _, err := ParseCodec([]byte(strings.Replace(tinyCodec, `"encoder_dim": 512`, `"encoder_dim": 256`, 1))); err == nil {
		t.Error("a codec of another width accepted")
	}
}
