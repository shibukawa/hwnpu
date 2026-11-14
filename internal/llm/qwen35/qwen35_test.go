package qwen35

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// tinyJSON is a small Qwen3.5 as its checkpoint describes it: the text model under text_config
// of a qwen3_5 config, every matrix's rows a multiple of the lut4 group of 16. It has both layer
// types, GQA (4 heads on 2), partial RoPE (8 of 16 dimensions), and key heads shared by two value
// heads each in the Gated DeltaNet.
const tinyJSON = `{"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "tie_word_embeddings": true,
	"text_config": {"model_type": "qwen3_5_text", "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 4,
		"num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 48, "rms_norm_eps": 1e-6,
		"hidden_act": "silu", "attn_output_gate": true, "attention_bias": false, "max_position_embeddings": 4096,
		"layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
		"linear_conv_kernel_dim": 4, "linear_key_head_dim": 4, "linear_num_key_heads": 8,
		"linear_num_value_heads": 16, "linear_value_head_dim": 4,
		"rope_parameters": {"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5,
			"mrope_interleaved": true, "mrope_section": [2, 1, 1]}}}`

func tinyConfig(t *testing.T) *Config {
	t.Helper()
	c, err := ParseConfig([]byte(tinyJSON))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// round16 rounds to fp16: the program's constants (rotary tables, 1 + w of the norms, -exp(A_log),
// dt_bias, the L2 norms' scales) are stored in it.
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

func sigmoid(x float64) float64  { return 1 / (1 + math.Exp(-x)) }
func silu(x float64) float64     { return x * sigmoid(x) }
func softplus(x float64) float64 { return math.Log1p(math.Exp(x)) }

// reference is the Qwen3.5 text model's forward pass written out directly in float64 from
// modeling_qwen3_5.py (transformers 5.22): Qwen3_5DecoderLayer with zero centered RMS norms,
// Qwen3_5GatedDeltaNet with causal_conv1d_fn and torch_recurrent_gated_delta_rule from a zero
// state, Qwen3_5Attention with the gate in q_proj, rotate_half over the rotary dimensions and
// repeat_kv, Qwen3_5MLP. It is an oracle independent of the program builder and the interpreter.
// The result is the final hidden state (after the last norm), or with stop >= 0 the hidden state
// after layer stop.
func reference(t *testing.T, c *Config, w llm.Weights, x []float32, stop int) [][]float64 {
	d := c.HiddenSize
	h := make([][]float64, len(x)/d)
	for i := range h {
		h[i] = make([]float64, d)
		for j := range h[i] {
			h[i][j] = float64(x[i*d+j])
		}
	}
	var layers []refLayer
	for l := 0; l < c.NumLayers; l++ {
		layers = append(layers, refLayer{fmt.Sprintf("%slayers.%d.", c.prefix(), l), c.IsLinear(l)})
	}
	return referenceLayers(t, c, w, h, layers, c.prefix()+"norm.weight", stop)
}

// refLayer is a decoder layer of the reference: its tensors' prefix and its kind.
type refLayer struct {
	prefix string
	linear bool
}

// referenceLayers runs the layers over the hidden states h (rows at positions 0, 1, ...), and the
// final norm, or with stop >= 0 stops after layer stop. It changes h.
func referenceLayers(t *testing.T, c *Config, w llm.Weights, h [][]float64, layers []refLayer, final string, stop int) [][]float64 {
	d := c.HiddenSize
	s := len(h)
	matvec := func(m []float64, rows int, v []float64) []float64 {
		cols := len(v)
		y := make([]float64, rows)
		for r := range y {
			for j := 0; j < cols; j++ {
				y[r] += m[r*cols+j] * v[j]
			}
		}
		return y
	}
	rms := func(v, gamma []float64, eps float64) []float64 {
		var ms float64
		for _, a := range v {
			ms += a * a
		}
		r := 1 / math.Sqrt(ms/float64(len(v))+round16(eps)) // the epsilon is an fp16 constant
		out := make([]float64, len(v))
		for i, a := range v {
			out[i] = a * r * gamma[i]
		}
		return out
	}
	// Qwen3_5RMSNorm: x normalized times (1 + w), 1 + w stored in fp16 by the program (rounded
	// once: rounding w first moves some to the neighbouring fp16 value of 1 + w)
	onePlus := func(name string) []float64 {
		v, _, err := w.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		g := make([]float64, len(v))
		for i, x := range v {
			g[i] = round16(1 + float64(x))
		}
		return g
	}
	inv := c.RopeInvFreq()
	rot := c.RotaryDim()
	ropeAt := func(v []float64, pos int) {
		half := rot / 2
		r := make([]float64, rot)
		for j := 0; j < rot; j++ {
			f := inv[j%half]
			cos, sin := round16(math.Cos(float64(pos)*f)), round16(math.Sin(float64(pos)*f))
			var rh float64 // rotate_half: cat(-x2, x1)
			if j < half {
				rh = -v[j+half]
			} else {
				rh = v[j-half]
			}
			r[j] = v[j]*cos + rh*sin
		}
		copy(v, r)
	}
	for l, layer := range layers {
		lp := layer.prefix
		in := onePlus(lp + "input_layernorm.weight")
		hn := make([][]float64, s)
		for i := range h {
			hn[i] = rms(h[i], in, c.NormEps)
		}
		mix := make([][]float64, s)
		if layer.linear {
			a := lp + "linear_attn."
			hk, hv, dk, dv, K := c.LinearKeyHeads, c.LinearValueHeads, c.LinearKeyDim, c.LinearValueDim, c.ConvKernel
			kd, vd, cd := c.keyDim(), c.valueDim(), c.convDim()
			wqkv, wz, wb, wa := f16(t, w, a+"in_proj_qkv.weight"), f16(t, w, a+"in_proj_z.weight"), f16(t, w, a+"in_proj_b.weight"), f16(t, w, a+"in_proj_a.weight")
			cw, wo, gn := f16(t, w, a+"conv1d.weight"), f16(t, w, a+"out_proj.weight"), f16(t, w, a+"norm.weight")
			alog, dt := f16(t, w, a+"A_log"), f16(t, w, a+"dt_bias")
			proj := make([][]float64, s)
			for i := range hn {
				proj[i] = matvec(wqkv, cd, hn[i])
			}
			// causal_conv1d_fn: depthwise over the sequence, zero padded on the left, then silu
			conv := make([][]float64, s)
			for i := range conv {
				conv[i] = make([]float64, cd)
				for ch := 0; ch < cd; ch++ {
					var acc float64
					for j := 0; j < K; j++ {
						if src := i + j - (K - 1); src >= 0 {
							acc += cw[ch*K+j] * proj[src][ch]
						}
					}
					conv[i][ch] = silu(acc)
				}
			}
			l2 := func(v []float64) []float64 {
				var ss float64
				for _, a := range v {
					ss += a * a
				}
				out := make([]float64, len(v))
				for i, a := range v {
					// the program's RMS norm with epsilon 1e-6 / n, stored in fp16
					n := float64(len(v))
					out[i] = a / math.Sqrt(ss+n*round16(1e-6/n))
				}
				return out
			}
			state := make([]float64, hv*dk*dv) // (hv, dk, dv)
			for i := 0; i < s; i++ {
				z, b, av := matvec(wz, vd, hn[i]), matvec(wb, hv, hn[i]), matvec(wa, hv, hn[i])
				o := make([]float64, vd)
				for j := 0; j < hv; j++ {
					g := j / (hv / hk) // repeat_interleave
					q := l2(conv[i][g*dk : (g+1)*dk])
					k := l2(conv[i][kd+g*dk : kd+(g+1)*dk])
					v := conv[i][2*kd+j*dv : 2*kd+(j+1)*dv]
					for e := range q {
						q[e] /= math.Sqrt(float64(dk))
					}
					beta := sigmoid(b[j])
					decay := math.Exp(round16(-math.Exp(alog[j])) * softplus(av[j]+dt[j]))
					S := state[j*dk*dv : (j+1)*dk*dv]
					for e := range S {
						S[e] *= decay
					}
					for e := 0; e < dv; e++ {
						var mem float64
						for r := 0; r < dk; r++ {
							mem += S[r*dv+e] * k[r]
						}
						delta := (v[e] - mem) * beta
						for r := 0; r < dk; r++ {
							S[r*dv+e] += k[r] * delta
						}
					}
					// Qwen3_5RMSNormGated: norm(o) * w * silu(z)
					oh := make([]float64, dv)
					for e := 0; e < dv; e++ {
						for r := 0; r < dk; r++ {
							oh[e] += S[r*dv+e] * q[r]
						}
					}
					// the program normalizes o scaled by r = oScale / oNormIn with eps r^2 in fp16
					var ms float64
					for _, a := range oh {
						ms += a * a
					}
					r2 := float64(oScale/oNormIn) * float64(oScale/oNormIn)
					inv := 1 / math.Sqrt(ms/float64(dv)+round16(c.NormEps*r2)/r2)
					for e := range oh {
						oh[e] *= inv * gn[e]
					}
					for e := range oh {
						o[j*dv+e] = oh[e] * silu(z[j*dv+e])
					}
				}
				mix[i] = matvec(wo, d, o)
			}
		} else {
			a := lp + "self_attn."
			nh, nkv, hd := c.NumHeads, c.NumKVHeads, c.HeadDim
			wq, wk, wv, wo := f16(t, w, a+"q_proj.weight"), f16(t, w, a+"k_proj.weight"), f16(t, w, a+"v_proj.weight"), f16(t, w, a+"o_proj.weight")
			qn, kn := onePlus(a+"q_norm.weight"), onePlus(a+"k_norm.weight")
			q := make([][][]float64, s) // (s, nh, hd)
			gate := make([][]float64, s)
			k, v := make([][][]float64, s), make([][][]float64, s)
			for i := range hn {
				qg := matvec(wq, 2*nh*hd, hn[i])
				kk, vv := matvec(wk, nkv*hd, hn[i]), matvec(wv, nkv*hd, hn[i])
				gate[i] = make([]float64, nh*hd)
				for hh := 0; hh < nh; hh++ {
					qh := rms(qg[hh*2*hd:hh*2*hd+hd], qn, c.NormEps)
					ropeAt(qh, i)
					q[i] = append(q[i], qh)
					copy(gate[i][hh*hd:], qg[hh*2*hd+hd:(hh+1)*2*hd])
				}
				for hh := 0; hh < nkv; hh++ {
					kh := rms(kk[hh*hd:(hh+1)*hd], kn, c.NormEps)
					ropeAt(kh, i)
					k[i] = append(k[i], kh)
					v[i] = append(v[i], vv[hh*hd:(hh+1)*hd])
				}
			}
			scale := 1 / math.Sqrt(float64(hd))
			for i := 0; i < s; i++ {
				o := make([]float64, nh*hd)
				for hh := 0; hh < nh; hh++ {
					kvh := hh / (nh / nkv) // repeat_kv
					sc := make([]float64, i+1)
					top := math.Inf(-1)
					for j := 0; j <= i; j++ {
						for e := 0; e < hd; e++ {
							sc[j] += q[i][hh][e] * k[j][kvh][e]
						}
						sc[j] *= scale
						top = math.Max(top, sc[j])
					}
					var sum float64
					for j := range sc {
						sc[j] = math.Exp(sc[j] - top)
						sum += sc[j]
					}
					for j := range sc {
						for e := 0; e < hd; e++ {
							o[hh*hd+e] += sc[j] / sum * v[j][kvh][e]
						}
					}
				}
				for e := range o {
					o[e] *= sigmoid(gate[i][e])
				}
				mix[i] = matvec(wo, d, o)
			}
		}
		post := onePlus(lp + "post_attention_layernorm.weight")
		wg, wu, wd := f16(t, w, lp+"mlp.gate_proj.weight"), f16(t, w, lp+"mlp.up_proj.weight"), f16(t, w, lp+"mlp.down_proj.weight")
		for i := range h {
			for j := range h[i] {
				h[i][j] += mix[i][j]
			}
			f := rms(h[i], post, c.NormEps)
			g, u := matvec(wg, c.IntermediateSize, f), matvec(wu, c.IntermediateSize, f)
			for j := range g {
				g[j] = silu(g[j]) * u[j]
			}
			down := matvec(wd, d, g)
			for j := range h[i] {
				h[i][j] += down[j]
			}
		}
		if l == stop {
			return h
		}
	}
	fin := onePlus(final)
	for i := range h {
		h[i] = rms(h[i], fin, c.NormEps)
	}
	return h
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

// logits multiplies hidden states by the output layer.
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

// buildProgram builds a program into a fresh weight file and returns it with its inputs and
// outputs and a digest of the program and the weights.
func buildProgram(t *testing.T, cfg llm.Cached, w llm.Weights, opts llm.Options) (p *milcpu.Program, ins, outs []mlpackage.Feature, digest string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	ins, outs, err = cfg.BuildCached(b, w, opts)
	if err != nil {
		t.Fatal(err)
	}
	var names []*mlpackage.Var
	for _, o := range outs {
		names = append(names, &mlpackage.Var{Name: o.Name})
	}
	prog := b.Finish(names...)
	if err := blobs.Close(); err != nil {
		t.Fatal(err)
	}
	raw, err := proto.MarshalOptions{Deterministic: true}.Marshal(prog)
	if err != nil {
		t.Fatal(err)
	}
	bin, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	p, err = milcpu.Load(prog, f)
	if err != nil {
		t.Fatal(err)
	}
	return p, ins, outs, fmt.Sprintf("program %x weights %x", sha256.Sum256(raw), sha256.Sum256(bin))
}

// runCPU builds a program without a cache and runs it on x with the CPU interpreter.
func runCPU(t *testing.T, cfg llm.Cached, w llm.Weights, opts llm.Options, x []float32) []float32 {
	t.Helper()
	p, ins, outs, _ := buildProgram(t, cfg, w, opts)
	res, err := p.Run(map[string]milcpu.Tensor{"x": {Shape: ins[0].Shape, Data: x}}, milcpu.Options{})
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

func relErr(got, ref []float32) float64 {
	var se, sr float64
	for i := range ref {
		d := float64(got[i] - ref[i])
		se += d * d
		sr += float64(ref[i]) * float64(ref[i])
	}
	return math.Sqrt(se / sr)
}

// The program computes the reference's logits.
func TestAgainstFP64(t *testing.T) {
	cfg := tinyConfig(t)
	w := llm.NewSynthetic(cfg)
	ids := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	x := embed16(t, cfg, w, ids)
	want := logits(t, cfg, w, reference(t, cfg, w, x, -1))
	got := runCPU(t, cfg, w, llm.Options{Seq: len(ids), Format: "fp16"}, x)
	e := relErr64(got, want, cfg.VocabSize)
	if e > 1e-4 {
		t.Errorf("relative error %g against the reference", e)
	}
	t.Logf("relative error %.2g", e)
}

// One layer at a time (Start, Layers, NoHead, and Start at the end for the final norm and the
// output layer), the program computes each layer's output of the reference, and chained, the
// logits of the whole program.
func TestLayerByLayer(t *testing.T) {
	cfg := tinyConfig(t)
	w := llm.NewSynthetic(cfg)
	ids := []int32{7, 2, 9, 4, 4, 1}
	x := embed16(t, cfg, w, ids)
	s := len(ids)
	h := x
	for l := 0; l < cfg.NumLayers; l++ {
		h = runCPU(t, cfg, w, llm.Options{Seq: s, Format: "fp16", Start: l, Layers: 1, NoHead: true}, h)
		if e := relErr64(h, reference(t, cfg, w, x, l), cfg.HiddenSize); e > 1e-4 {
			t.Errorf("layer %d: relative error %g against the reference", l, e)
		}
	}
	got := runCPU(t, cfg, w, llm.Options{Seq: s, Format: "fp16", Start: cfg.NumLayers}, h)
	whole := runCPU(t, cfg, w, llm.Options{Seq: s, Format: "fp16"}, x)
	if e := relErr(got, whole); e > 1e-5 {
		t.Errorf("chained layers: relative error %g against the whole program", e)
	}
}

// ParseConfig rejects what the conversion does not implement, and reads a text only config.
func TestParseConfig(t *testing.T) {
	edit := func(f func(top, text map[string]any)) []byte {
		var top map[string]any
		if err := json.Unmarshal([]byte(tinyJSON), &top); err != nil {
			t.Fatal(err)
		}
		f(top, top["text_config"].(map[string]any))
		raw, _ := json.Marshal(top)
		return raw
	}
	for _, c := range []struct {
		name string
		raw  []byte
		want string
	}{
		{"moe", edit(func(_, x map[string]any) { x["num_experts"] = 8 }), "mixture of experts"},
		{"act", edit(func(_, x map[string]any) { x["hidden_act"] = "gelu" }), "hidden_act"},
		{"bias", edit(func(_, x map[string]any) { x["attention_bias"] = true }), "attention_bias"},
		{"gate", edit(func(_, x map[string]any) { x["attn_output_gate"] = false }), "attn_output_gate"},
		{"layer type", edit(func(_, x map[string]any) { x["layer_types"].([]any)[2] = "sliding_attention" }), "unknown layer type"},
		{"layer count", edit(func(_, x map[string]any) { x["num_hidden_layers"] = 5 }), "layer_types"},
		{"rope", edit(func(_, x map[string]any) { x["rope_parameters"].(map[string]any)["rope_type"] = "yarn" }), "rope type"},
		{"mrope", edit(func(_, x map[string]any) { x["rope_parameters"].(map[string]any)["mrope_section"] = []int{2, 2, 2} }), "mrope_section"},
		{"value heads", edit(func(_, x map[string]any) { x["linear_num_value_heads"] = 12 }), "value heads"},
		{"text model", edit(func(_, x map[string]any) { x["model_type"] = "qwen3_text" }), "qwen3_5_text"},
		{"model type", edit(func(top, _ map[string]any) { top["model_type"] = "qwen3_next" }), "not supported"},
	} {
		if _, err := ParseConfig(c.raw); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one about %q", c.name, err, c.want)
		}
	}
	text := edit(func(top, x map[string]any) {
		for k := range top {
			delete(top, k)
		}
		for k, v := range x {
			top[k] = v
		}
		top["tie_word_embeddings"] = false
	})
	c, err := ParseConfig(text)
	if err != nil {
		t.Fatal(err)
	}
	if c.Embedding() != "model.embed_tokens.weight" || c.Head() != "lm_head.weight" {
		t.Errorf("text only checkpoint: embedding %s, head %s", c.Embedding(), c.Head())
	}
	full := tinyConfig(t)
	if full.Embedding() != "model.language_model.embed_tokens.weight" || full.Head() != full.Embedding() {
		t.Errorf("image and text checkpoint: embedding %s, head %s", full.Embedding(), full.Head())
	}
	a, err := llm.Detect([]byte(tinyJSON))
	if err != nil || a != Arch {
		t.Errorf("detected %v, %v", a, err)
	}
}

// Builds the tiny model in each weight format. The digest of the program and the weight file is
// logged, so that a refactoring which should not change the output can be checked by comparing go
// test -v output before and after.
func TestBuildTiny(t *testing.T) {
	cfg := tinyConfig(t)
	w := llm.NewSynthetic(cfg)
	for _, c := range []llm.Options{
		{Seq: 6},
		{Seq: 6, Format: "lut4"},
		{Seq: 6, Format: "fp16", Hidden: true},
		{Seq: 1, Cache: 8, Format: "fp16"},
		{Seq: 4, Cache: 8, Format: "lut6"},
	} {
		t.Run(fmt.Sprintf("seq%d cache%d %s", c.Seq, c.Cache, c.Format), func(t *testing.T) {
			_, ins, outs, digest := buildProgram(t, cfg, w, c)
			want := cfg.VocabSize
			if c.Hidden {
				want = cfg.HiddenSize
			}
			if got := outs[0].Shape[2]; got != want || ins[0].Shape[1] != c.Seq {
				t.Errorf("in %v, out %v", ins[0].Shape, outs[0].Shape)
			}
			t.Logf("%s", digest)
		})
	}
}

// The program has none of the Neural Engine's own silu and softplus, which were off by percents
// on an M3 (silu over the convolution's small outputs, softplus 0 where the decay needs it), and
// the model lost long prompts: they are written with exp, log and real_div (builder.silu,
// builder.softplus).
func TestNoNeuralEngineActivations(t *testing.T) {
	cfg := tinyConfig(t)
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	_, outs, err := BuildCached(b, cfg, llm.NewSynthetic(cfg), llm.Options{Seq: 4, Cache: 8, Format: "fp16"})
	if err != nil {
		t.Fatal(err)
	}
	var names []*mlpackage.Var
	for _, o := range outs {
		names = append(names, &mlpackage.Var{Name: o.Name})
	}
	prog := b.Finish(names...)
	blobs.Close()
	fn := prog.GetFunctions()["main"]
	ops := map[string]int{}
	for _, op := range fn.GetBlockSpecializations()[fn.GetOpset()].GetOperations() {
		ops[op.GetType()]++
	}
	for _, bad := range []string{"silu", "softplus"} {
		if ops[bad] > 0 {
			t.Errorf("%d %s operations", ops[bad], bad)
		}
	}
	if ops["log"] == 0 || ops["real_div"] == 0 {
		t.Errorf("no log or real_div: %v", ops)
	}
}

// The gated delta rule over chunks of rows (deltaRuleChunks, the default) gives the logits of the
// rule one row after the other (delta=rows) and of the float64 reference, for sequences of one
// row, of a whole chunk, and past it into a chunk not filled.
func TestDeltaChunks(t *testing.T) {
	cfg := tinyConfig(t)
	w := llm.NewSynthetic(cfg)
	for _, n := range []int{1, deltaChunk, deltaChunk + 5, 3 * deltaChunk} {
		ids := make([]int32, n)
		for i := range ids {
			ids[i] = int32((i*7 + 3) % cfg.VocabSize)
		}
		x := embed16(t, cfg, w, ids)
		rows := runCPU(t, cfg, w, llm.Options{Seq: n, Format: "fp16", Params: map[string]string{"delta": "rows"}}, x)
		chunks := runCPU(t, cfg, w, llm.Options{Seq: n, Format: "fp16"}, x)
		want := logits(t, cfg, w, reference(t, cfg, w, x, -1))
		if e := relErr(chunks, rows); e > 1e-3 {
			t.Errorf("%d rows: relative error %g against the rule row by row", n, e)
		}
		if e := relErr64(chunks, want, cfg.VocabSize); e > 1e-4 {
			t.Errorf("%d rows: relative error %g against the reference", n, e)
		}
	}
}
