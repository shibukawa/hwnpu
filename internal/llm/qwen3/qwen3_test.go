package qwen3

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// tiny is a small Qwen3 with what Qwen3-0.6B has: grouped query attention (4 heads sharing 2
// key and value heads), a head_dim apart from hidden_size / heads, Q and K norms, and the output
// layer tied to the embeddings. Matrix rows are multiples of the lut4 group of 16.
const tiny = `{"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3", "attention_bias": false,
	"attention_dropout": 0.0, "bos_token_id": 151643, "eos_token_id": 151645, "head_dim": 16,
	"hidden_act": "silu", "hidden_size": 48, "intermediate_size": 64, "max_position_embeddings": 40960,
	"max_window_layers": 3, "num_attention_heads": 4, "num_hidden_layers": 3, "num_key_value_heads": 2,
	"rms_norm_eps": 1e-06, "rope_scaling": null, "rope_theta": 1000000, "sliding_window": null,
	"tie_word_embeddings": true, "use_cache": true, "use_sliding_window": false, "vocab_size": 48}`

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

// reference is Qwen3ForCausalLM's forward written out directly from modeling_qwen3.py, in
// float64: an oracle independent of the program builder and the interpreter. Weights, the
// embeddings the host passes and the rotary tables are rounded to fp16, as the program holds them.
func reference(t *testing.T, c *Config, w llm.Weights, ids []int32) refRun {
	t.Helper()
	d, s, nh, nkv, hd, f := c.HiddenSize, len(ids), c.NumHeads, c.NumKVHeads, c.HeadDim, c.IntermediateSize
	// Qwen3RMSNorm: weight * x * rsqrt(mean(x^2) + eps), over rows of n
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
	// nn.Linear without a bias: out rows of in columns
	linear := func(x []float64, name string, in, out int) []float64 {
		wt := f16(t, w, name)
		y := make([]float64, s*out)
		for r := 0; r < s; r++ {
			for o := 0; o < out; o++ {
				var acc float64
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
	// apply_rotary_pos_emb on each head of x (s, heads, hd): x*cos + rotate_half(x)*sin, with
	// Qwen3RotaryEmbedding's default inv_freq
	rope := func(x []float64, heads int) {
		half := hd / 2
		for r := 0; r < s; r++ {
			for h := 0; h < heads; h++ {
				v := x[(r*heads+h)*hd : (r*heads+h+1)*hd]
				out := make([]float64, hd)
				for j := 0; j < hd; j++ {
					a := float64(r) / math.Pow(c.RopeTheta, float64(2*(j%half))/float64(hd))
					rot := -v[(j+half)%hd]
					if j >= half {
						rot = v[j-half]
					}
					out[j] = v[j]*r16(math.Cos(a)) + rot*r16(math.Sin(a))
				}
				copy(v, out)
			}
		}
	}

	var run refRun
	h := make([]float64, s*d)
	for i, x := range rows(t, w, c.Embedding(), ids) {
		h[i] = float64(x)
	}
	for i := 0; i < c.NumLayers; i++ {
		run.hidden = append(run.hidden, slices.Clone(h))
		l := fmt.Sprintf("model.layers.%d.", i)
		// Qwen3Attention: q_norm and k_norm over each head, then RoPE
		x := rmsNorm(h, d, l+"input_layernorm.weight")
		q := rmsNorm(linear(x, l+"self_attn.q_proj.weight", d, nh*hd), hd, l+"self_attn.q_norm.weight")
		k := rmsNorm(linear(x, l+"self_attn.k_proj.weight", d, nkv*hd), hd, l+"self_attn.k_norm.weight")
		v := linear(x, l+"self_attn.v_proj.weight", d, nkv*hd)
		rope(q, nh)
		rope(k, nkv)
		o := make([]float64, s*nh*hd)
		for hh := 0; hh < nh; hh++ {
			kh := hh / (nh / nkv) // repeat_kv
			for qi := 0; qi < s; qi++ {
				scores := make([]float64, qi+1)
				maxS := math.Inf(-1)
				for ki := 0; ki <= qi; ki++ {
					var dot float64
					for j := 0; j < hd; j++ {
						dot += q[(qi*nh+hh)*hd+j] * k[(ki*nkv+kh)*hd+j]
					}
					scores[ki] = dot / math.Sqrt(float64(hd))
					maxS = math.Max(maxS, scores[ki])
				}
				var z float64
				for ki := range scores {
					scores[ki] = math.Exp(scores[ki] - maxS)
					z += scores[ki]
				}
				for ki := range scores {
					for j := 0; j < hd; j++ {
						o[(qi*nh+hh)*hd+j] += scores[ki] / z * v[(ki*nkv+kh)*hd+j]
					}
				}
			}
		}
		h = add(h, linear(o, l+"self_attn.o_proj.weight", nh*hd, d))
		// Qwen3MLP
		x = rmsNorm(h, d, l+"post_attention_layernorm.weight")
		gate := linear(x, l+"mlp.gate_proj.weight", d, f)
		up := linear(x, l+"mlp.up_proj.weight", d, f)
		for j := range gate {
			gate[j] = gate[j] / (1 + math.Exp(-gate[j])) * up[j]
		}
		h = add(h, linear(gate, l+"mlp.down_proj.weight", f, d))
	}
	run.hidden = append(run.hidden, slices.Clone(h))
	run.logits = linear(rmsNorm(h, d, "model.norm.weight"), c.head(), d, c.VocabSize)
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

// The program in fp16 on the CPU gives the logits of the reference forward pass, with the output
// layer tied to the embeddings (Qwen3-0.6B) and apart from them (Qwen3-4B and up).
func TestMatchesReference(t *testing.T) {
	for _, tied := range []bool{true, false} {
		raw := tiny
		if !tied {
			raw = strings.Replace(raw, `"tie_word_embeddings": true`, `"tie_word_embeddings": false`, 1)
		}
		c, err := ParseConfig([]byte(raw))
		if err != nil {
			t.Fatal(err)
		}
		w := llm.NewSynthetic(c)
		ref := reference(t, c, w, tokens)
		p := build(t, c, w, llm.Options{Seq: len(tokens), Format: "fp16"})
		out := p.run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})
		if e := relErr(out["y"], ref.logits); e > 1e-4 {
			t.Errorf("tied %v: logits: relative error %g", tied, e)
		} else {
			t.Logf("tied %v: logits: relative error %.2g", tied, e)
		}
		if len(p.outs) != 1 || len(p.ins) != 1 {
			t.Errorf("the whole model has inputs %v and outputs %v besides x and y", p.ins[1:], p.outs[1:])
		}
		if _, ok := c.Tensors()["lm_head.weight"]; ok == tied {
			t.Errorf("tied %v: reads lm_head %v", tied, ok)
		}
	}
}

// Each layer on its own, fed the reference's hidden state, gives the reference's next hidden
// state; so does the head after the last layer. This is what hwnpu-check does with a real
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
		if e > 1e-4 {
			t.Errorf("layer %d: relative error %g", i, e)
		}
		t.Logf("layer %d: relative error %.2g", i, e)
	}
}

// The settings the conversion does not build are refused.
func TestParseConfigRejects(t *testing.T) {
	for _, c := range []struct{ old, new, want string }{
		{`"model_type": "qwen3"`, `"model_type": "qwen2"`, "model_type"},
		{`"hidden_act": "silu"`, `"hidden_act": "gelu"`, "hidden_act"},
		{`"attention_bias": false`, `"attention_bias": true`, "attention_bias"},
		{`"use_sliding_window": false`, `"use_sliding_window": true`, "use_sliding_window"},
		{`"rope_scaling": null`, `"rope_scaling": {"rope_type": "yarn", "factor": 4.0}`, "rope_scaling"},
		{`"num_key_value_heads": 2`, `"num_key_value_heads": 3`, "heads"},
		{`"head_dim": 16`, `"head_dim": 15`, "head_dim"},
		{`"rope_theta": 1000000,`, `"rope_theta": 1000000, "layer_types": ["full_attention", "sliding_attention", "full_attention"],`, "sliding_attention"},
		{`"rope_theta": 1000000,`, `"rope_parameters": {"rope_type": "yarn", "rope_theta": 1000000},`, "rope_type"},
	} {
		raw := strings.Replace(tiny, c.old, c.new, 1)
		if raw == tiny {
			t.Fatalf("%q is not in the tiny config", c.old)
		}
		if _, err := ParseConfig([]byte(raw)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one about %s", c.new, err, c.want)
		}
	}
	// transformers 5's rope_parameters give rope_theta
	raw := strings.Replace(tiny, `"rope_theta": 1000000,`, `"rope_parameters": {"rope_type": "default", "rope_theta": 5000}, "layer_types": ["full_attention", "full_attention", "full_attention"],`, 1)
	c, err := ParseConfig([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if c.RopeTheta != 5000 {
		t.Errorf("rope_theta %g, want 5000 from rope_parameters", c.RopeTheta)
	}
}
