package plamo3

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

// tiny is a small PLaMo 3 with what PLaMo 3 Translate 2B has: sliding and full layers (every
// second one full), grouped query attention, YaRN on the full layers (the 2B's factor and
// initial context; with head_dim 16 its ramp lies between pairs 1.7 and 3.8), and an lm_head
// apart from the embeddings with rows padded past the vocabulary of 40 to 48. Matrix rows are
// multiples of the lut4 group of 16. The window of 3 is shorter than the test sequences.
const tiny = `{"architectures": ["Plamo3ForCausalLM"], "model_type": "plamo3",
	"hidden_size": 64, "intermediate_size": 48, "num_hidden_layers": 4, "num_attention_heads": 4,
	"num_key_value_heads": 2, "head_dim": 16, "vocab_size": 40, "rms_norm_eps": 1e-6,
	"window_size": 3, "sliding_window_pattern": 2, "rope_theta": 1000000, "rope_local_theta": 10000,
	"rope_scaling_factor": 256.0, "initial_context_length": 4096, "max_position_embeddings": 262144,
	"tie_word_embeddings": false, "scale_embedding": false, "image_feature_size": null,
	"image_proj_type": "linear", "linear_type": "normal", "bos_token_id": 1, "eos_token_id": 1,
	"rope_parameters": {"full_attention": {"rope_type": "yarn", "rope_theta": 1000000, "factor": 256.0,
		"original_max_position_embeddings": 4096, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": false},
		"sliding_attention": {"rope_type": "default", "rope_theta": 10000}}}`

func tinyConfig(t *testing.T) *Config {
	t.Helper()
	c, err := ParseConfig([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

var tokens = []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 39}

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

// reference is Plamo3ForCausalLM's forward written out directly from modeling_plamo.py, in
// float64: an oracle independent of the program builder and the interpreter. Weights, the
// embeddings the host passes and the rotary tables are rounded to fp16, as the program holds them;
// YaRN's concentration multiplies cos and sin, as RotaryEmbedding does.
func reference(t *testing.T, c *Config, w llm.Weights, ids []int32) refRun {
	t.Helper()
	d, s, nh, nkv, hd, f := c.HiddenSize, len(ids), c.NumHeads, c.NumKVHeads, c.HeadDim, c.IntermediateSize
	// _plamo3_rms_norm: x * rsqrt(mean(x^2) + eps) * (offset + weight), over rows of n; the
	// program holds offset + weight in fp16 (Hugging Face in bf16)
	rmsNorm := func(x []float64, n int, name string, offset float64) []float64 {
		v, _, err := w.Get(name)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float64, len(x))
		for r := 0; r < len(x)/n; r++ {
			var ms float64
			for _, a := range x[r*n : (r+1)*n] {
				ms += a * a
			}
			inv := math.Pow(ms/float64(n)+c.NormEps, -0.5)
			for j := 0; j < n; j++ {
				out[r*n+j] = x[r*n+j] * inv * r16(offset+float64(v[j]))
			}
		}
		return out
	}
	// nn.Linear over the rows [from, from+out) of a weight with in columns
	linear := func(x []float64, name string, from, in, out int) []float64 {
		wt := f16(t, w, name)
		y := make([]float64, s*out)
		for r := 0; r < s; r++ {
			for o := 0; o < out; o++ {
				var acc float64
				for k := 0; k < in; k++ {
					acc += x[r*in+k] * wt[(from+o)*in+k]
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
	full := func(i int) bool { return (i+1)%c.Pattern == 0 } // is_full_attn
	// RotaryEmbedding._compute_buffers: inv_freq and the concentration
	rotary := func(full bool) ([]float64, float64) {
		base := c.RopeLocalTheta
		if full {
			base = c.RopeTheta
		}
		inv := make([]float64, hd/2)
		for j := range inv {
			inv[j] = 1 / math.Pow(base, float64(2*j)/float64(hd))
		}
		if !full {
			return inv, 1
		}
		factor, orig := c.RopeScalingFactor, float64(*c.InitialContext)
		dh := float64(hd / 2)
		low := dh * math.Log(orig/(32*2*math.Pi)) / math.Log(base)
		high := dh * math.Log(orig/(1*2*math.Pi)) / math.Log(base)
		for j := range inv {
			ramp := (float64(j) - low) / (high - low)
			mask := 1 - math.Min(math.Max(ramp, 0), 1)
			inv[j] = inv[j]/factor*(1-mask) + inv[j]*mask
		}
		return inv, 0.1*math.Log(factor) + 1
	}
	// _rotary_pos_emb on each head of x (s, heads, hd): x*cos + rotate_half(x)*sin
	rope := func(x []float64, heads int, full bool) {
		inv, conc := rotary(full)
		half := hd / 2
		for r := 0; r < s; r++ {
			for h := 0; h < heads; h++ {
				v := x[(r*heads+h)*hd : (r*heads+h+1)*hd]
				out := make([]float64, hd)
				for j := 0; j < hd; j++ {
					a := float64(r) * inv[j%half]
					rot := -v[(j+half)%hd]
					if j >= half {
						rot = v[j-half]
					}
					out[j] = v[j]*r16(math.Cos(a))*conc + rot*r16(math.Sin(a))*conc
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
		l := fmt.Sprintf("model.layers.layers.%d.", i)
		// Attention
		x := rmsNorm(h, d, l+"pre_mixer_norm.weight", 1)
		qkv := l + "mixer.qkv_proj.weight"
		q := rmsNorm(linear(x, qkv, 0, d, nh*hd), hd, l+"mixer.q_norm.weight", 1)
		k := rmsNorm(linear(x, qkv, nh*hd, d, nkv*hd), hd, l+"mixer.k_norm.weight", 1)
		v := linear(x, qkv, (nh+nkv)*hd, d, nkv*hd)
		rope(q, nh, full(i))
		rope(k, nkv, full(i))
		o := make([]float64, s*nh*hd)
		for hh := 0; hh < nh; hh++ {
			kh := hh / (nh / nkv) // enable_gqa: kv heads repeat
			for qi := 0; qi < s; qi++ {
				scores := make([]float64, s)
				maxS := math.Inf(-1)
				for ki := 0; ki <= qi; ki++ {
					// swa_mask: keys within window_size positions, the query's own included
					if !full(i) && ki <= qi-c.Window {
						scores[ki] = math.Inf(-1)
						continue
					}
					var dot float64
					for j := 0; j < hd; j++ {
						dot += q[(qi*nh+hh)*hd+j] * k[(ki*nkv+kh)*hd+j]
					}
					scores[ki] = dot / math.Sqrt(float64(hd))
					maxS = math.Max(maxS, scores[ki])
				}
				var z float64
				for ki := 0; ki <= qi; ki++ {
					scores[ki] = math.Exp(scores[ki] - maxS)
					z += scores[ki]
				}
				for ki := 0; ki <= qi; ki++ {
					for j := 0; j < hd; j++ {
						o[(qi*nh+hh)*hd+j] += scores[ki] / z * v[(ki*nkv+kh)*hd+j]
					}
				}
			}
		}
		a := linear(o, l+"mixer.o_proj.weight", 0, nh*hd, d)
		h = add(h, rmsNorm(a, d, l+"post_mixer_norm.weight", 1.0/5))
		// MLP: _swiglu of gate_up_proj, the gate first
		x = rmsNorm(h, d, l+"pre_mlp_norm.weight", 1)
		gu := linear(x, l+"mlp.gate_up_proj.weight", 0, d, 2*f)
		g := make([]float64, s*f)
		for r := 0; r < s; r++ {
			for j := 0; j < f; j++ {
				a, b := gu[r*2*f+j], gu[r*2*f+f+j]
				g[r*f+j] = a / (1 + math.Exp(-a)) * b
			}
		}
		h = add(h, rmsNorm(linear(g, l+"mlp.down_proj.weight", 0, f, d), d, l+"post_mlp_norm.weight", 1/math.Pow(5, 1.5)))
	}
	run.hidden = append(run.hidden, slices.Clone(h))
	x := rmsNorm(h, d, "model.norm.weight", 1)
	// lm_head over the padded rows, then cut to the vocabulary
	run.logits = linear(x, "lm_head.weight", 0, d, c.VocabSize)
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

// The program in fp16 on the CPU gives the logits of the reference forward pass.
func TestMatchesReference(t *testing.T) {
	c := tinyConfig(t)
	w := llm.NewSynthetic(c)
	ref := reference(t, c, w, tokens)
	p := build(t, c, w, llm.Options{Seq: len(tokens), Format: "fp16"})
	out := p.run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})
	if e := relErr(out["y"], ref.logits); e > 1e-4 {
		t.Errorf("logits: relative error %g", e)
	} else {
		t.Logf("logits: relative error %.2g", e)
	}
	if len(p.outs) != 1 || len(p.ins) != 1 {
		t.Errorf("the whole model has inputs %v and outputs %v besides x and y", p.ins[1:], p.outs[1:])
	}
	if y := p.outs[0].Shape; y[2] != c.VocabSize {
		t.Errorf("y %v is not cut to the vocabulary of %d", y, c.VocabSize)
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
		{`"model_type": "plamo3"`, `"model_type": "plamo2"`, "model_type"},
		{`"scale_embedding": false`, `"scale_embedding": true`, "scale_embedding"},
		{`"image_feature_size": null`, `"image_feature_size": 1024`, "image"},
		{`"linear_type": "normal"`, `"linear_type": "fp8"`, "linear_type"},
		{`"num_key_value_heads": 2`, `"num_key_value_heads": 3`, "heads"},
		{`"initial_context_length": 4096,`, ``, "initial_context_length"},
		{`"initial_context_length": 4096`, `"initial_context_length": 64`, "ramp"},
		{`"tie_word_embeddings": false`, `"tie_word_embeddings": true`, "tied"},
		{`"rope_scaling_factor": 256.0`, `"rope_scaling_factor": 0.5`, "rope_scaling_factor"},
	} {
		raw := strings.Replace(tiny, c.old, c.new, 1)
		if raw == tiny {
			t.Fatalf("%q is not in the tiny config", c.old)
		}
		if _, err := ParseConfig([]byte(raw)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one about %s", c.new, err, c.want)
		}
	}
	// without YaRN, and tied with a vocabulary of whole sixteens, it builds
	raw := strings.Replace(tiny, `"rope_scaling_factor": 256.0, "initial_context_length": 4096`, `"rope_scaling_factor": 1`, 1)
	raw = strings.Replace(raw, `"vocab_size": 40`, `"vocab_size": 48`, 1)
	raw = strings.Replace(raw, `"tie_word_embeddings": false`, `"tie_word_embeddings": true`, 1)
	c, err := ParseConfig([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	if c.Concentration() != 1 || c.head() != c.Embedding() {
		t.Errorf("concentration %g, head %s", c.Concentration(), c.head())
	}
	if _, ok := c.Tensors()["lm_head.weight"]; ok {
		t.Error("a tied model reads lm_head")
	}
}
