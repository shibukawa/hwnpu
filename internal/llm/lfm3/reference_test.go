package lfm3

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

// tinyD1 is a small d1 model as its checkpoint describes it: LFM3 under the text_config of an
// lfm3_vl config, the output layer tied at the top level, two full attention layers.
const tinyD1 = `{"architectures": ["D1Model"], "model_type": "lfm3_vl", "tie_word_embeddings": true,
	"d1_codes": 2, "d1_attention": "set",
	"text_config": {"model_type": "lfm3", "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 4,
		"num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 8, "vocab_size": 48, "norm_eps": 1e-5,
		"conv_L_cache": 3, "conv_bias": false, "sliding_window": 3, "use_sandwich_norm": true, "tie_word_embeddings": false,
		"layer_types": ["full_attention", "conv", "sliding_attention", "full_attention"],
		"rope_parameters": {"full_attention": {"rope_theta": 10000, "partial_rotary_factor": 0.5, "rope_type": "default"},
			"sliding_attention": {"rope_theta": 1000, "rope_type": "default"}}}}`

// round16 rounds to fp16: the program's constants (rotary tables, epsilons, the attention
// scale) are stored in it.
func round16(v float64) float64 {
	return float64(mlpackage.Float16Value(mlpackage.Float16Bits(float32(v))))
}

// f16 reads a tensor rounded to fp16, as the program stores it.
func f16(t *testing.T, w llm.Weights, name string) []float64 {
	t.Helper()
	v, _, err := w.Get(name)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(mlpackage.Float16Value(mlpackage.Float16Bits(x)))
	}
	return out
}

// reference is the LFM3 forward pass written out directly, in float64, from modeling_lfm3.py as
// hwnpu-check confirmed it against PyTorch (sandwich RMS norms, the gated short convolution of
// Lfm2ShortConv, attention with QK norm, rotate_half RoPE over the first partial_rotary_factor
// of each head, repeat_kv and a sigmoid gate per head, SwiGLU), and for a d1 model from
// llama.cpp's graph of it (src/models/lfm3.cpp: the full attention layers bidirectional, the rest
// causal). It is an oracle independent of the program builder and the interpreter. ids are the
// prompt alone, without padding. The result is the final hidden state (after the last norm).
func reference(t *testing.T, c *Config, w llm.Weights, ids []int32) [][]float64 {
	d, s := c.HiddenSize, len(ids)
	p := c.prefix()
	matvec := func(m []float64, rows int, x []float64) []float64 {
		cols := len(x)
		y := make([]float64, rows)
		for r := 0; r < rows; r++ {
			for k := 0; k < cols; k++ {
				y[r] += m[r*cols+k] * x[k]
			}
		}
		return y
	}
	linear := func(name string, rows int, xs [][]float64) [][]float64 {
		m := f16(t, w, name)
		out := make([][]float64, len(xs))
		for i, x := range xs {
			out[i] = matvec(m, rows, x)
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
			r := 1 / math.Sqrt(ss/float64(len(x))+round16(c.NormEps))
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
	sigmoid := func(v float64) float64 { return 1 / (1 + math.Exp(-v)) }

	emb := f16(t, w, c.Embedding())
	h := make([][]float64, s)
	for i, id := range ids {
		h[i] = append([]float64(nil), emb[int(id)*d:int(id+1)*d]...)
	}

	conv := func(l string, x [][]float64) [][]float64 {
		k := c.ConvKernel
		bcx := linear(l+"conv.in_proj.weight", 3*d, x)
		cw := f16(t, w, l+"conv.conv.weight") // (d, 1, k)
		var cb []float64
		if c.ConvBias {
			cb = f16(t, w, l+"conv.conv.bias")
		}
		y := make([][]float64, s)
		for i := range y {
			y[i] = make([]float64, d)
			for ch := 0; ch < d; ch++ {
				// nn.Conv1d with padding k-1, the first seq outputs: causal from zeros
				var acc float64
				for j := 0; j < k; j++ {
					if src := i + j - (k - 1); src >= 0 {
						acc += cw[ch*k+j] * bcx[src][ch] * bcx[src][2*d+ch] // B * x
					}
				}
				if cb != nil {
					acc += cb[ch]
				}
				y[i][ch] = bcx[i][d+ch] * acc // C gates
			}
		}
		return linear(l+"conv.out_proj.weight", d, y)
	}

	attn := func(l, layerType string, x [][]float64) [][]float64 {
		nh, nkv, hd := c.NumHeads, c.NumKVHeads, c.HeadDim
		a := l + "self_attn."
		rp := c.RopeParameters[layerType]
		factor := 1.0
		if rp.PartialFactor != nil {
			factor = *rp.PartialFactor
		}
		rotated := int(factor*float64(hd)) / 2 // frequency pairs that rotate
		heads := func(name string, n int) [][][]float64 {
			flat := linear(a+name+"_proj.weight", n*hd, x)
			out := make([][][]float64, s)
			for i := range flat {
				out[i] = make([][]float64, n)
				for j := 0; j < n; j++ {
					out[i][j] = flat[i][j*hd : (j+1)*hd]
				}
			}
			return out
		}
		normRope := func(v [][][]float64, norm string) {
			for i := range v {
				v[i] = rms(a+norm, v[i])
				for j := range v[i] {
					x := v[i][j]
					r := make([]float64, hd)
					for f := 0; f < hd/2; f++ {
						var ang float64
						if f < rotated {
							ang = float64(i) * math.Pow(rp.Theta, -float64(2*f)/float64(hd))
						}
						co, si := round16(math.Cos(ang)), round16(math.Sin(ang)) // the program's tables are fp16
						r[f] = x[f]*co - x[f+hd/2]*si
						r[f+hd/2] = x[f+hd/2]*co + x[f]*si
					}
					v[i][j] = r
				}
			}
		}
		q, k, v := heads("q", nh), heads("k", nkv), heads("v", nkv)
		normRope(q, "q_norm.weight")
		normRope(k, "k_norm.weight")
		gate := linear(a+"gate_proj.weight", nh, x)
		o := make([][]float64, s)
		for i := range o {
			o[i] = make([]float64, nh*hd)
			for hh := 0; hh < nh; hh++ {
				kv := hh / (nh / nkv)
				visible := func(j int) bool {
					switch {
					case c.D1 && layerType == "full_attention":
						return true
					case layerType == "sliding_attention":
						return j <= i && i-j < c.SlidingWindow
					}
					return j <= i
				}
				scores := make([]float64, s)
				mx := math.Inf(-1)
				for j := 0; j < s; j++ {
					if !visible(j) {
						continue
					}
					for e := 0; e < hd; e++ {
						scores[j] += q[i][hh][e] * k[j][kv][e]
					}
					scores[j] *= round16(1 / math.Sqrt(float64(hd)))
					mx = math.Max(mx, scores[j])
				}
				var sum float64
				for j := 0; j < s; j++ {
					if visible(j) {
						scores[j] = math.Exp(scores[j] - mx)
						sum += scores[j]
					} else {
						scores[j] = 0
					}
				}
				g := sigmoid(gate[i][hh])
				for j := 0; j < s; j++ {
					for e := 0; e < hd; e++ {
						o[i][hh*hd+e] += scores[j] / sum * v[j][kv][e] * g
					}
				}
			}
		}
		return linear(a+"o_proj.weight", d, o)
	}

	for i := 0; i < c.NumLayers; i++ {
		l := fmt.Sprintf("%slayers.%d.", p, i)
		x := rms(l+"operator_norm.weight", h)
		if c.IsAttention(i) {
			x = attn(l, c.LayerTypes[i], x)
		} else {
			x = conv(l, x)
		}
		if c.UseSandwichNorm {
			x = rms(l+"post_operator_norm.weight", x)
		}
		h = add(h, x)
		f := rms(l+"ffn_norm.weight", h)
		g := linear(l+"feed_forward.w1.weight", c.IntermediateSize, f)
		u := linear(l+"feed_forward.w3.weight", c.IntermediateSize, f)
		for r := range g {
			for j := range g[r] {
				g[r][j] = g[r][j] * sigmoid(g[r][j]) * u[r][j]
			}
		}
		f = linear(l+"feed_forward.w2.weight", d, g)
		if c.UseSandwichNorm {
			f = rms(l+"post_ffn_norm.weight", f)
		}
		h = add(h, f)
	}
	return rms(p+"embedding_norm.weight", h)
}

// embed16 is llm.Embed rounded to fp16, the program's input type.
func embed16(t *testing.T, c *Config, w llm.Weights, ids []int32) []float32 {
	t.Helper()
	x, err := llm.Embed(w, c, ids)
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range x {
		x[i] = mlpackage.Float16Value(mlpackage.Float16Bits(v))
	}
	return x
}

// logits multiplies hidden states by the output layer (lm_head, or the tied embedding table).
func logits(t *testing.T, c *Config, w llm.Weights, h [][]float64) [][]float64 {
	head := f16(t, w, c.Head())
	out := make([][]float64, len(h))
	for i, x := range h {
		out[i] = make([]float64, c.VocabSize)
		for v := range out[i] {
			for j, xv := range x {
				out[i][v] += head[v*c.HiddenSize+j] * xv
			}
		}
	}
	return out
}

// runInputs builds a program and runs it with the CPU interpreter on the given inputs.
func runInputs(t *testing.T, cfg *Config, w llm.Weights, opts Options, in map[string][]float32) []float32 {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	ins, outs, err := BuildCached(b, cfg, w, opts)
	if err != nil {
		t.Fatal(err)
	}
	prog := b.Finish(&mlpackage.Var{Name: outs[0].Name})
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
		v, ok := in[i.Name]
		if !ok {
			t.Fatalf("no value for input %s", i.Name)
		}
		feed[i.Name] = milcpu.Tensor{Shape: i.Shape, Data: v}
	}
	res, err := p.Run(feed, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return res[outs[0].Name].Data
}

// relErr64 is the relative RMS error of the first len(ref) rows of got (rows of width w).
func relErr64(got []float32, ref [][]float64, w int) float64 {
	var se, sr float64
	for i, row := range ref {
		for j, r := range row {
			e := float64(got[i*w+j]) - r
			se += e * e
			sr += r * r
		}
	}
	return math.Sqrt(se / sr)
}

// The causal LFM3 program computes the reference's logits.
func TestAgainstFP64(t *testing.T) {
	cfg := tinyConfig()
	w := llm.NewSynthetic(cfg)
	ids := []int32{3, 1, 4, 1, 5, 9}
	x := embed16(t, cfg, w, ids)
	want := logits(t, cfg, w, reference(t, cfg, w, ids))
	got := runInputs(t, cfg, w, Options{Options: llm.Options{Seq: len(ids), Format: "fp16"}}, map[string][]float32{"x": x})
	e := relErr64(got, want, cfg.VocabSize)
	if e > 1e-4 {
		t.Errorf("relative error %g against the reference", e)
	}
	t.Logf("relative error %.2g", e)
}

// A d1 program computes the reference's final hidden state for the prompt's tokens whatever the
// padding after them, whole or one layer at a time, and with post-scale.
func TestD1AgainstFP64(t *testing.T) {
	cfg, err := ParseConfig([]byte(tinyD1))
	if err != nil {
		t.Fatal(err)
	}
	if !cfg.D1 || cfg.Prefix != "model.language_model." || !cfg.TieWordEmbed {
		t.Fatalf("parsed D1 %v, prefix %q, tied %v", cfg.D1, cfg.Prefix, cfg.TieWordEmbed)
	}
	if n, logits := cfg.Output(); n != cfg.HiddenSize || logits {
		t.Fatalf("output %d, logits %v", n, logits)
	}
	w := llm.NewSynthetic(cfg)
	ids := []int32{7, 3, 40, 1, 5, 22, 9}
	const seq = 10
	want := reference(t, cfg, w, ids)
	x := embed16(t, cfg, w, append(append([]int32{}, ids...), 0, 0, 0))
	mask := make([]float32, seq)
	for i := range ids {
		mask[i] = 1
	}
	fp16 := llm.Options{Seq: seq, Format: "fp16"}
	got := runInputs(t, cfg, w, Options{Options: fp16}, map[string][]float32{"x": x, "mask": mask})
	if len(got) != seq*cfg.HiddenSize {
		t.Fatalf("%d outputs", len(got))
	}
	e := relErr64(got, want, cfg.HiddenSize)
	if e > 1e-4 {
		t.Errorf("relative error %g against the reference", e)
	}
	t.Logf("relative error %.2g with %d tokens of padding", e, seq-len(ids))
	scaled := runInputs(t, cfg, w, Options{Options: fp16, PostScale: 64}, map[string][]float32{"x": x, "mask": mask})
	if e := relErr64(scaled, want, cfg.HiddenSize); e > 1e-4 {
		t.Errorf("post-scale 64: relative error %g against the reference", e)
	}
	h := x
	for i := 0; i < cfg.NumLayers; i++ {
		o := fp16
		o.Start, o.Layers, o.NoHead = i, 1, true
		h = runInputs(t, cfg, w, Options{Options: o}, map[string][]float32{"x": h, "mask": mask})
	}
	o := fp16
	o.Start = cfg.NumLayers
	h = runInputs(t, cfg, w, Options{Options: o}, map[string][]float32{"x": h})
	if e := relErr64(h, want, cfg.HiddenSize); e > 1e-4 {
		t.Errorf("layer by layer: relative error %g against the reference", e)
	}
	if _, _, err := BuildCached(mlpackage.NewBuilder("CoreML8", nil), cfg, w, Options{Options: llm.Options{Seq: 1, Cache: 8}}); err == nil {
		t.Error("d1 model built with a KV cache")
	}
}

func TestMultimodalConfig(t *testing.T) {
	cfg, err := ParseConfig([]byte(strings.Replace(tinyD1, `"D1Model"`, `"Lfm3VlForConditionalGeneration"`, 1)))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.D1 || cfg.Embedding() != "model.language_model.embed_tokens.weight" || cfg.Head() != cfg.Embedding() {
		t.Errorf("LFM3-VL: d1 %v, embedding %s, head %s", cfg.D1, cfg.Embedding(), cfg.Head())
	}
	if _, ok := cfg.Tensors()["model.language_model.layers.3.self_attn.q_proj.weight"]; !ok {
		t.Error("tensors are not under model.language_model.")
	}
	if a, err := llm.Detect([]byte(tinyD1)); err != nil || a != Arch {
		t.Errorf("d1 detected as %v, %v", a, err)
	}
	if g := ggufName("model.language_model.layers.2.feed_forward.w1.weight"); g != "blk.2.ffn_gate.weight" {
		t.Errorf("gguf name %q", g)
	}
	for _, c := range []struct{ name, from, to string }{
		{"d1_codes", `"d1_codes": 2`, `"d1_codes": 1`},
		{"d1_attention", `"d1_attention": "set"`, `"d1_attention": "causal"`},
		{"no d1 settings", `"d1_codes": 2, "d1_attention": "set",`, ``},
		{"text model type", `"model_type": "lfm3", "hidden_size"`, `"model_type": "lfm2", "hidden_size"`},
		{"no text_config", `"text_config"`, `"vision_config"`},
		{"model type", `"lfm3_vl"`, `"lfm3_audio"`},
	} {
		if _, err := ParseConfig([]byte(strings.Replace(tinyD1, c.from, c.to, 1))); err == nil {
			t.Errorf("%s: unsupported config accepted", c.name)
		}
	}
}
