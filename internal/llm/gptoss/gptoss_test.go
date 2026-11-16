package gptoss

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// tiny is a small gpt-oss with what gpt-oss-20b has: sliding and full layers in turn from a
// sliding one, grouped query attention with biases and sinks, YaRN with gpt-oss-20b's settings
// (truncate off; with head_dim 16 its ramp lies between pairs 2.0 and 4.6), 8 experts of which
// 2 are routed, and dense experts (no quantization_config: transformers' parameters, in by out).
// The swiglu limit of 0.5 clamps some of the synthetic gates and ups. Matrix rows are multiples
// of the lut4 group of 16; the window of 3 is shorter than the test sequences.
const tiny = `{"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss",
	"hidden_size": 64, "intermediate_size": 32, "num_hidden_layers": 4, "num_attention_heads": 4,
	"num_key_value_heads": 2, "head_dim": 16, "vocab_size": 48, "rms_norm_eps": 1e-5,
	"sliding_window": 3, "layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
	"num_local_experts": 8, "num_experts_per_tok": 2, "experts_per_token": 2, "swiglu_limit": 0.5,
	"attention_bias": true, "tie_word_embeddings": false, "hidden_act": "silu",
	"rope_theta": 150000, "max_position_embeddings": 131072, "initial_context_length": 4096,
	"rope_scaling": {"beta_fast": 32.0, "beta_slow": 1.0, "factor": 32.0, "original_max_position_embeddings": 4096,
		"rope_type": "yarn", "truncate": false}}`

func tinyConfig(t *testing.T) *Config {
	t.Helper()
	c, err := ParseConfig([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

var tokens = []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 47}

// r16 rounds to fp16.
func r16(x float64) float64 {
	return float64(mlpackage.Float16Value(mlpackage.Float16Bits(float32(x))))
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
		out[i] = r16(float64(x))
	}
	return out
}

// rows looks up the rows of a table for the tokens in fp16, as the host passes them.
func rows(t *testing.T, w llm.Weights, table string, ids []int32) []float32 {
	t.Helper()
	var out []float32
	for _, id := range ids {
		v, err := w.Rows(table, int(id), int(id)+1)
		if err != nil {
			t.Fatal(err)
		}
		for _, x := range v {
			out = append(out, float32(r16(float64(x))))
		}
	}
	return out
}

// refRun is the reference forward pass: the hidden state entering each layer and after the last
// (before the final norm), and the logits.
type refRun struct {
	hidden [][]float64
	logits []float64
}

// reference is GptOssForCausalLM's forward written out directly from modeling_gpt_oss.py, in
// float64: an oracle independent of the program builder and the interpreter. Weights, biases,
// sinks, the embeddings the host passes and the rotary tables are rounded to fp16, as the program
// holds them; YaRN's attention_factor multiplies cos and sin, as GptOssRotaryEmbedding does.
func reference(t *testing.T, c *Config, w llm.Weights, ids []int32) refRun {
	t.Helper()
	d, s, nh, nkv, hd, f, nE := c.HiddenSize, len(ids), c.NumHeads, c.NumKVHeads, c.HeadDim, c.IntermediateSize, c.Experts
	// GptOssRMSNorm: weight * (x * rsqrt(mean(x^2) + eps)), over rows of n
	rmsNorm := func(x []float64, n int, name string) []float64 {
		v := f16(t, w, name)
		out := make([]float64, len(x))
		for r := 0; r < len(x)/n; r++ {
			var ms float64
			for _, a := range x[r*n : (r+1)*n] {
				ms += a * a
			}
			inv := math.Pow(ms/float64(n)+c.NormEps, -0.5)
			for j := 0; j < n; j++ {
				out[r*n+j] = x[r*n+j] * inv * v[j]
			}
		}
		return out
	}
	// nn.Linear with a bias: weight (out, in)
	linear := func(x []float64, name, bias string, in, out int) []float64 {
		wt, b := f16(t, w, name), f16(t, w, bias)
		y := make([]float64, s*out)
		for r := 0; r < s; r++ {
			for o := 0; o < out; o++ {
				acc := b[o]
				for k := 0; k < in; k++ {
					acc += x[r*in+k] * wt[o*in+k]
				}
				y[r*out+o] = acc
			}
		}
		return y
	}
	add := func(a, b []float64) []float64 {
		out := make([]float64, len(a))
		for i := range a {
			out[i] = a[i] + b[i]
		}
		return out
	}
	// _compute_yarn_parameters (transformers' modeling_rope_utils), truncate false
	base, factor, orig := 150000.0, 32.0, 4096.0
	inv := make([]float64, hd/2)
	dim := float64(hd)
	corr := func(rot float64) float64 { return dim * math.Log(orig/(rot*2*math.Pi)) / (2 * math.Log(base)) }
	low, high := math.Max(corr(32), 0), math.Min(corr(1), dim-1)
	for j := range inv {
		pos := math.Pow(base, float64(2*j)/dim)
		ramp := math.Min(math.Max((float64(j)-low)/(high-low), 0), 1)
		extra := 1 - ramp
		inv[j] = 1/(factor*pos)*(1-extra) + 1/pos*extra
	}
	conc := 0.1*math.Log(factor) + 1
	// _apply_rotary_emb on each head of x (s, heads, hd): the halves rotated
	rope := func(x []float64, heads int) {
		half := hd / 2
		for r := 0; r < s; r++ {
			for h := 0; h < heads; h++ {
				v := x[(r*heads+h)*hd : (r*heads+h+1)*hd]
				out := make([]float64, hd)
				for j := 0; j < half; j++ {
					a := float64(r) * inv[j]
					cs, sn := r16(math.Cos(a))*conc, r16(math.Sin(a))*conc
					out[j] = v[j]*cs - v[j+half]*sn
					out[j+half] = v[j+half]*cs + v[j]*sn
				}
				copy(v, out)
			}
		}
	}
	lim := c.SwigluLimit

	var run refRun
	h := make([]float64, s*d)
	for i, x := range rows(t, w, c.Embedding(), ids) {
		h[i] = float64(x)
	}
	for i := 0; i < c.NumLayers; i++ {
		run.hidden = append(run.hidden, slices.Clone(h))
		l := fmt.Sprintf("model.layers.%d.", i)
		sliding := c.LayerTypes[i] == "sliding_attention"
		// GptOssAttention
		x := rmsNorm(h, d, l+"input_layernorm.weight")
		q := linear(x, l+"self_attn.q_proj.weight", l+"self_attn.q_proj.bias", d, nh*hd)
		k := linear(x, l+"self_attn.k_proj.weight", l+"self_attn.k_proj.bias", d, nkv*hd)
		v := linear(x, l+"self_attn.v_proj.weight", l+"self_attn.v_proj.bias", d, nkv*hd)
		rope(q, nh)
		rope(k, nkv)
		sinks := f16(t, w, l+"self_attn.sinks")
		o := make([]float64, s*nh*hd)
		for hh := 0; hh < nh; hh++ {
			kh := hh / (nh / nkv) // repeat_kv
			for qi := 0; qi < s; qi++ {
				// eager_attention_forward: the scores, the sink as one more column, the softmax,
				// the sink's column dropped
				var keys []int
				for ki := 0; ki <= qi; ki++ {
					if sliding && qi-ki >= c.Window {
						continue
					}
					keys = append(keys, ki)
				}
				scores := make([]float64, len(keys))
				top := sinks[hh]
				for n, ki := range keys {
					var dot float64
					for j := 0; j < hd; j++ {
						dot += q[(qi*nh+hh)*hd+j] * k[(ki*nkv+kh)*hd+j]
					}
					scores[n] = dot / math.Sqrt(float64(hd))
					top = math.Max(top, scores[n])
				}
				z := math.Exp(sinks[hh] - top)
				for n := range scores {
					scores[n] = math.Exp(scores[n] - top)
					z += scores[n]
				}
				for n, ki := range keys {
					for j := 0; j < hd; j++ {
						o[(qi*nh+hh)*hd+j] += scores[n] / z * v[(ki*nkv+kh)*hd+j]
					}
				}
			}
		}
		h = add(h, linear(o, l+"self_attn.o_proj.weight", l+"self_attn.o_proj.bias", nh*hd, d))
		// GptOssMLP: the router's top k and their softmax, the experts weighted by it
		x = rmsNorm(h, d, l+"post_attention_layernorm.weight")
		logits := linear(x, l+"mlp.router.weight", l+"mlp.router.bias", d, nE)
		gu, gub := f16(t, w, l+"mlp.experts.gate_up_proj"), f16(t, w, l+"mlp.experts.gate_up_proj_bias")
		dn, dnb := f16(t, w, l+"mlp.experts.down_proj"), f16(t, w, l+"mlp.experts.down_proj_bias")
		moe := make([]float64, s*d)
		for r := 0; r < s; r++ {
			lr := logits[r*nE : (r+1)*nE]
			order := make([]int, nE)
			for e := range order {
				order[e] = e
			}
			sort.Slice(order, func(a, b int) bool { return lr[order[a]] > lr[order[b]] })
			chosen := order[:c.TopK]
			var z float64
			for _, e := range chosen {
				z += math.Exp(lr[e] - lr[chosen[0]])
			}
			for _, e := range chosen {
				rw := math.Exp(lr[e]-lr[chosen[0]]) / z
				// GptOssExperts: gate_up (hidden, 2 inter) in by out, the gate on the even columns
				mid := make([]float64, f)
				for j := 0; j < f; j++ {
					gate, up := gub[e*2*f+2*j], gub[e*2*f+2*j+1]
					for k := 0; k < d; k++ {
						gate += x[r*d+k] * gu[(e*d+k)*2*f+2*j]
						up += x[r*d+k] * gu[(e*d+k)*2*f+2*j+1]
					}
					gate = math.Min(gate, lim)
					up = math.Min(math.Max(up, -lim), lim)
					glu := gate / (1 + math.Exp(-gate*1.702))
					mid[j] = (up + 1) * glu
				}
				for o := 0; o < d; o++ {
					acc := dnb[e*d+o]
					for j := 0; j < f; j++ {
						acc += mid[j] * dn[(e*f+j)*d+o]
					}
					moe[r*d+o] += rw * acc
				}
			}
		}
		h = add(h, moe)
	}
	run.hidden = append(run.hidden, slices.Clone(h))
	x := rmsNorm(h, d, "model.norm.weight")
	wt := f16(t, w, "lm_head.weight")
	run.logits = make([]float64, s*c.VocabSize)
	for r := 0; r < s; r++ {
		for o := 0; o < c.VocabSize; o++ {
			var acc float64
			for k := 0; k < d; k++ {
				acc += x[r*d+k] * wt[o*d+k]
			}
			run.logits[r*c.VocabSize+o] = acc
		}
	}
	return run
}

func f32(x []float64) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(v)
	}
	return out
}

func relErr(got []float32, ref []float64) float64 {
	var se, sr float64
	for i := range ref {
		d := float64(got[i]) - ref[i]
		se += d * d
		sr += ref[i] * ref[i]
	}
	return math.Sqrt(se / sr)
}

// program is a built program loaded on the CPU interpreter.
type program struct {
	p         *milcpu.Program
	ins, outs []mlpackage.Feature
}

func build(t *testing.T, c *Config, w llm.Weights, opts llm.Options) program {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	ins, outs, err := Build(b, c, w, opts)
	if err != nil {
		t.Fatal(err)
	}
	var vars []*mlpackage.Var
	for _, o := range outs {
		vars = append(vars, &mlpackage.Var{Name: o.Name})
	}
	prog := b.Finish(vars...)
	if err := blobs.Close(); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	p, err := milcpu.Load(prog, f)
	if err != nil {
		t.Fatal(err)
	}
	return program{p, ins, outs}
}

// run feeds the named inputs, which must be all the program takes, and returns every output.
func (p program) run(t *testing.T, feed map[string][]float32) map[string][]float32 {
	t.Helper()
	in := map[string]milcpu.Tensor{}
	for _, f := range p.ins {
		v, ok := feed[f.Name]
		if !ok {
			t.Fatalf("no value for the input %s %v", f.Name, f.Shape)
		}
		in[f.Name] = milcpu.Tensor{Shape: f.Shape, Data: v}
	}
	res, err := p.p.Run(in, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	out := map[string][]float32{}
	for _, f := range p.outs {
		out[f.Name] = res[f.Name].Data
	}
	return out
}

// The program in fp16 on the CPU gives the logits of the reference forward pass: the sinks, the
// window, YaRN, the routing and the experts' clamps included.
func TestMatchesReference(t *testing.T) {
	c := tinyConfig(t)
	w := llm.NewSynthetic(c)
	ref := reference(t, c, w, tokens)
	p := build(t, c, w, llm.Options{Seq: len(tokens), Format: "fp16"})
	out := p.run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})
	if e := relErr(out["y"], ref.logits); !(e <= 1e-4) {
		t.Errorf("logits: relative error %g", e)
	} else {
		t.Logf("logits: relative error %.2g", e)
	}
	if len(p.outs) != 1 || len(p.ins) != 1 {
		t.Errorf("the whole model has inputs %v and outputs %v besides x and y", p.ins[1:], p.outs[1:])
	}
}

// Each layer on its own, fed the reference's hidden state, gives the reference's next hidden
// state; so does the head after the last layer. This is what hwnpu-dev check does with a real
// checkpoint.
func TestLayerByLayer(t *testing.T) {
	c := tinyConfig(t)
	w := llm.NewSynthetic(c)
	ref := reference(t, c, w, tokens)
	for i := 0; i <= c.NumLayers; i++ {
		p := build(t, c, w, llm.Options{Seq: len(tokens), Format: "fp16", Start: i, Layers: 1, NoHead: i < c.NumLayers})
		feed := map[string][]float32{"x": f32(ref.hidden[i])}
		if i == 0 {
			feed["x"] = rows(t, w, c.Embedding(), tokens)
		}
		out := p.run(t, feed)
		want := ref.logits
		if i < c.NumLayers {
			want = ref.hidden[i+1]
		}
		e := relErr(out["y"], want)
		if !(e <= 1e-4) {
			t.Errorf("layer %d: relative error %g", i, e)
		}
		t.Logf("layer %d: relative error %.2g", i, e)
	}
}

// The settings the conversion does not build are refused.
func TestParseConfigRejects(t *testing.T) {
	for _, c := range []struct{ old, new, want string }{
		{`"model_type": "gpt_oss"`, `"model_type": "gpt_oss2"`, "model_type"},
		{`"attention_bias": true`, `"attention_bias": false`, "attention_bias"},
		{`"tie_word_embeddings": false`, `"tie_word_embeddings": true`, "tie_word_embeddings"},
		{`"num_key_value_heads": 2`, `"num_key_value_heads": 3`, "heads"},
		{`"num_experts_per_tok": 2`, `"num_experts_per_tok": 9`, "experts"},
		{`"rope_type": "yarn"`, `"rope_type": "llama3"`, "rope_type"},
		{`"sliding_attention", "full_attention", "sliding_attention", "full_attention"`, `"sliding_attention", "chunked_attention", "sliding_attention", "full_attention"`, "chunked_attention"},
		{`"layer_types": ["sliding_attention", "full_attention", `, `"layer_types": [`, "layer_types"},
		{`"attention_bias": true,`, `"attention_bias": true, "quantization_config": {"quant_method": "fp8"},`, "quantization_config"},
	} {
		raw := strings.Replace(tiny, c.old, c.new, 1)
		if raw == tiny {
			t.Fatalf("%q is not in the tiny config", c.old)
		}
		if _, err := ParseConfig([]byte(raw)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one about %s", c.new, err, c.want)
		}
	}
}

// gpt-oss-20b's config: YaRN's concentration 1.3466 (ref_gptoss.py's attention_scaling), its
// ramp, the full layers paged, the experts MXFP4.
func TestConfig20B(t *testing.T) {
	raw, err := os.ReadFile("testdata/gpt-oss-20b-config.json")
	if err != nil {
		t.Fatal(err)
	}
	c, err := ParseConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	if k := c.Concentration(); math.Abs(k-1.3465735902799727) > 1e-12 {
		t.Errorf("concentration %v", k)
	}
	if !c.mxfp4() || c.TopK != 4 || c.Experts != 32 || c.Window != 128 {
		t.Errorf("mxfp4 %v, top %d of %d, window %d", c.mxfp4(), c.TopK, c.Experts, c.Window)
	}
	if got := c.PagedLayers(); len(got) != 12 || got[0] != 1 || got[11] != 23 {
		t.Errorf("PagedLayers %v", got)
	}
	inv := c.RopeInvFreq()
	// truncate false: the ramp between 8.09 and 17.4 of 32 pairs; pair 0 extrapolated, 31
	// interpolated by the factor
	if math.Abs(inv[0]-1) > 1e-12 || math.Abs(inv[31]-math.Pow(150000, -62.0/64)/32) > 1e-15 {
		t.Errorf("inv_freq %v ... %v", inv[0], inv[31])
	}
}
