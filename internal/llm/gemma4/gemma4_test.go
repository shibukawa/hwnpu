package gemma4

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

// tiny is a small Gemma 4 text model with everything E2B has: sliding and full layers with
// wider heads on the full ones, grouped query attention, proportional RoPE, per-layer
// embeddings, three layers that reuse the keys and values of layers 1 (sliding) and 2 (full)
// with a double wide MLP, and the logit soft cap. Matrix rows are multiples of the lut4 group of
// 16. The window of 3 is shorter than the test sequences.
const tiny = `{"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "text_config": {
	"model_type": "gemma4_text", "hidden_size": 64, "intermediate_size": 48, "num_hidden_layers": 6,
	"num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "global_head_dim": 32,
	"vocab_size": 48, "vocab_size_per_layer_input": 48, "hidden_size_per_layer_input": 16,
	"rms_norm_eps": 1e-6, "sliding_window": 3, "hidden_activation": "gelu_pytorch_tanh",
	"final_logit_softcapping": 2.0, "tie_word_embeddings": true, "num_kv_shared_layers": 3,
	"use_double_wide_mlp": true, "attention_bias": false, "attention_k_eq_v": false, "enable_moe_block": false,
	"use_bidirectional_attention": null, "num_global_key_value_heads": null,
	"layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "full_attention"],
	"rope_parameters": {
		"full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
		"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"}}}}`

func tinyConfig(t *testing.T) *Config {
	t.Helper()
	c, err := ParseConfig([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// weights are synthetic weights with the norm weights and layer scalars around 1, as trained
// ones are: Gemma 4's norms multiply by the weight itself, and with the synthetic values of a
// tenth or less the queries and keys would be so small that attention is nearly uniform and a
// wrong position or mask would hardly show.
type weights struct{ llm.Synthetic }

func aroundOne(name string) bool {
	return strings.HasSuffix(name, "norm.weight") || strings.HasSuffix(name, "layernorm.weight") || strings.HasSuffix(name, "layer_scalar")
}

func (w weights) Get(name string) ([]float32, []int, error) {
	v, shape, err := w.Synthetic.Get(name)
	if err == nil && aroundOne(name) {
		for i := range v {
			v[i] = 1 + 8*v[i]
		}
	}
	return v, shape, err
}

func (w weights) Rows(name string, begin, end int) ([]float32, error) {
	v, shape, err := w.Get(name)
	if err != nil {
		return nil, err
	}
	return v[begin*shape[1] : end*shape[1]], nil
}

func newWeights(c *Config) weights { return weights{llm.NewSynthetic(c)} }

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
// (before the final norm), each layer's per-layer input, the shared keys and values by layer
// type, and the logits.
type refRun struct {
	hidden     [][]float64 // layers+1 of seq x hidden
	perLayer   []float64   // seq x layers*per-layer dim, as the program lays it out
	kvs        map[string][2][]float64
	logits     []float64
	seq, vocab int
}

// reference is Gemma4ForConditionalGeneration's text forward written out directly from
// modeling_gemma4.py, in float64: an oracle independent of the program builder and the
// interpreter. Weights, the embeddings the host passes, and the rotary tables are rounded to
// fp16, as the program holds them.
func reference(t *testing.T, c *Config, w llm.Weights, ids []int32) refRun {
	t.Helper()
	d, s, nh, nkv, pd, nl := c.HiddenSize, len(ids), c.NumHeads, c.NumKVHeads, c.PerLayerDim, c.NumLayers
	p := c.Prefix
	// Written out as Gemma4TextAttention and Gemma4TextMLP decide them, not with the Config
	// methods the program uses.
	headDim := func(lt string) int {
		if lt == "full_attention" {
			return c.GlobalHeadDim
		}
		return c.HeadDim
	}
	first := nl - c.KVShared
	storeFull := func(i int) bool { // store_full_length_kv: the last layer of its type before the shared ones
		for j := i + 1; j < first; j++ {
			if c.LayerTypes[j] == c.LayerTypes[i] {
				return false
			}
		}
		return i < first
	}
	rmsNorm := func(x []float64, n int, g []float64) []float64 {
		out := make([]float64, len(x))
		for r := 0; r < len(x)/n; r++ {
			var ms float64
			for _, a := range x[r*n : (r+1)*n] {
				ms += a * a
			}
			inv := math.Pow(ms/float64(n)+c.NormEps, -0.5)
			for j := 0; j < n; j++ {
				out[r*n+j] = x[r*n+j] * inv
				if g != nil {
					out[r*n+j] *= g[j]
				}
			}
		}
		return out
	}
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
	gelu := func(a float64) float64 { return 0.5 * a * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(a+0.044715*a*a*a))) }
	add := func(a, b []float64) []float64 {
		out := make([]float64, len(a))
		for i := range a {
			out[i] = a[i] + b[i]
		}
		return out
	}
	// invFreq: _compute_default_rope_parameters, or _compute_proportional_rope_parameters with
	// the frequencies past partial_rotary_factor * head_dim / 2 zero.
	invFreq := func(lt string) []float64 {
		rp := c.RopeParameters[lt]
		hd := headDim(lt)
		n := hd / 2
		if rp.Type == "proportional" {
			n = int(*rp.PartialFactor * float64(hd) / 2)
		}
		inv := make([]float64, hd/2)
		for j := 0; j < n; j++ {
			inv[j] = 1 / math.Pow(rp.Theta, float64(2*j)/float64(hd))
		}
		return inv
	}
	// rope rotates each head of x (s, heads, hd): x*cos + rotate_half(x)*sin, cos and sin of
	// cat(freqs, freqs).
	rope := func(x []float64, heads int, lt string) {
		hd := headDim(lt)
		inv := invFreq(lt)
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
					out[j] = v[j]*r16(math.Cos(a)) + rot*r16(math.Sin(a))
				}
				copy(v, out)
			}
		}
	}

	run := refRun{seq: s, vocab: c.VocabSize, kvs: map[string][2][]float64{}}
	// Gemma4TextScaledWordEmbedding: the rows times sqrt(hidden).
	h := make([]float64, s*d)
	for i, x := range rows(t, w, c.Embedding(), ids) {
		h[i] = float64(x) * math.Sqrt(float64(d))
	}
	// get_per_layer_inputs and project_per_layer_inputs
	proj := linear(h, p+"per_layer_model_projection.weight", d, nl*pd)
	for i := range proj {
		proj[i] *= 1 / math.Sqrt(float64(d))
	}
	proj = rmsNorm(proj, pd, f16(t, w, p+"per_layer_projection_norm.weight"))
	tok := rows(t, w, c.perLayerTable(), ids)
	run.perLayer = make([]float64, len(proj))
	for i := range proj {
		run.perLayer[i] = (proj[i] + float64(tok[i])*math.Sqrt(float64(pd))) / math.Sqrt2
	}

	for i := 0; i < nl; i++ {
		run.hidden = append(run.hidden, slices.Clone(h))
		l := fmt.Sprintf("%slayers.%d.", p, i)
		lt := c.LayerTypes[i]
		hd := headDim(lt)
		// Gemma4TextAttention
		x := rmsNorm(h, d, f16(t, w, l+"input_layernorm.weight"))
		q := rmsNorm(linear(x, l+"self_attn.q_proj.weight", d, nh*hd), hd, f16(t, w, l+"self_attn.q_norm.weight"))
		rope(q, nh, lt)
		var k, v []float64
		if c.KVShared > 0 && i >= first {
			kv := run.kvs[lt]
			k, v = kv[0], kv[1]
		} else {
			k = rmsNorm(linear(x, l+"self_attn.k_proj.weight", d, nkv*hd), hd, f16(t, w, l+"self_attn.k_norm.weight"))
			rope(k, nkv, lt)
			v = rmsNorm(linear(x, l+"self_attn.v_proj.weight", d, nkv*hd), hd, nil)
			if storeFull(i) {
				run.kvs[lt] = [2][]float64{k, v}
			}
		}
		o := make([]float64, s*nh*hd)
		for hh := 0; hh < nh; hh++ {
			kh := hh / (nh / nkv) // repeat_kv
			for qi := 0; qi < s; qi++ {
				scores := make([]float64, s)
				maxS := math.Inf(-1)
				for ki := 0; ki <= qi; ki++ {
					if lt == "sliding_attention" && qi-ki >= c.SlidingWindow {
						scores[ki] = math.Inf(-1)
						continue
					}
					var dot float64
					for j := 0; j < hd; j++ {
						dot += q[(qi*nh+hh)*hd+j] * k[(ki*nkv+kh)*hd+j]
					}
					scores[ki] = dot // scaling 1.0
					maxS = math.Max(maxS, dot)
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
		a := rmsNorm(linear(o, l+"self_attn.o_proj.weight", nh*hd, d), d, f16(t, w, l+"post_attention_layernorm.weight"))
		h = add(h, a)
		// Gemma4TextMLP
		f := c.IntermediateSize
		if c.DoubleWideMLP && i >= first && first > 0 {
			f *= 2
		}
		x = rmsNorm(h, d, f16(t, w, l+"pre_feedforward_layernorm.weight"))
		g, u := linear(x, l+"mlp.gate_proj.weight", d, f), linear(x, l+"mlp.up_proj.weight", d, f)
		for j := range g {
			g[j] = gelu(g[j]) * u[j]
		}
		h = add(h, rmsNorm(linear(g, l+"mlp.down_proj.weight", f, d), d, f16(t, w, l+"post_feedforward_layernorm.weight")))
		// the per-layer input
		g = linear(h, l+"per_layer_input_gate.weight", d, pd)
		for r := 0; r < s; r++ {
			for j := 0; j < pd; j++ {
				g[r*pd+j] = gelu(g[r*pd+j]) * run.perLayer[r*nl*pd+i*pd+j]
			}
		}
		h = add(h, rmsNorm(linear(g, l+"per_layer_projection.weight", pd, d), d, f16(t, w, l+"post_per_layer_input_norm.weight")))
		scalar := f16(t, w, l+"layer_scalar")[0]
		for j := range h {
			h[j] *= scalar
		}
	}
	run.hidden = append(run.hidden, slices.Clone(h))
	x := rmsNorm(h, d, f16(t, w, p+"norm.weight"))
	run.logits = linear(x, c.Embedding(), d, c.VocabSize)
	for i, z := range run.logits {
		run.logits[i] = math.Tanh(z / *c.Softcap) * *c.Softcap
	}
	return run
}

// sharedKV lays out the reference's keys or values of a layer type (seq, kv heads, hd) as the
// program does, (1, kv heads, seq, hd).
func (r refRun) sharedKV(c *Config, kv int, lt string) []float32 {
	x := r.kvs[lt][kv]
	nkv, hd := c.NumKVHeads, c.headDim(lt)
	out := make([]float32, len(x))
	for s := 0; s < r.seq; s++ {
		for h := 0; h < nkv; h++ {
			for j := 0; j < hd; j++ {
				out[(h*r.seq+s)*hd+j] = float32(x[(s*nkv+h)*hd+j])
			}
		}
	}
	return out
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

func build(t *testing.T, c *Config, w llm.Weights, opts Options) program {
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

var tokens = []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 47}

// The program in fp16 on the CPU gives the logits of the reference forward pass.
func TestMatchesReference(t *testing.T) {
	c := tinyConfig(t)
	w := newWeights(c)
	ref := reference(t, c, w, tokens)
	p := build(t, c, w, Options{Options: llm.Options{Seq: len(tokens), Format: "fp16"}})
	out := p.run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens), "ple": rows(t, w, c.perLayerTable(), tokens)})
	if e := relErr(out["y"], ref.logits); e > 1e-4 {
		t.Errorf("logits: relative error %g", e)
	} else {
		t.Logf("logits: relative error %.2g", e)
	}
	if len(p.outs) != 1 {
		t.Errorf("the whole model has outputs %v besides y", p.outs[1:])
	}
}

// Each layer on its own, fed the reference's hidden state, per-layer inputs and shared keys and
// values, gives the reference's next hidden state; so does the head after the last layer. This
// is what hwnpu-check does with a real checkpoint.
func TestLayerByLayer(t *testing.T) {
	c := tinyConfig(t)
	w := newWeights(c)
	ref := reference(t, c, w, tokens)
	s := len(tokens)
	for i := 0; i <= c.NumLayers; i++ {
		opts := Options{Options: llm.Options{Seq: s, Format: "fp16", Start: i, Layers: 1, NoHead: i < c.NumLayers}}
		p := build(t, c, w, opts)
		feed := map[string][]float32{"x": f32(ref.hidden[i]), "per_layer_inputs": f32(ref.perLayer)}
		if i == 0 {
			feed["x"] = rows(t, w, c.Embedding(), tokens)
			feed["ple"] = rows(t, w, c.perLayerTable(), tokens)
		}
		for _, lt := range []string{"sliding_attention", "full_attention"} {
			feed[sharedName("k", lt)] = ref.sharedKV(c, 0, lt)
			feed[sharedName("v", lt)] = ref.sharedKV(c, 1, lt)
		}
		out := p.run(t, feed)
		var want []float64
		if i < c.NumLayers {
			want = ref.hidden[i+1]
		} else {
			want = ref.logits
		}
		e := relErr(out["y"], want)
		if e > 1e-4 {
			t.Errorf("layer %d: relative error %g", i, e)
		}
		var names []string
		for _, f := range p.ins {
			names = append(names, f.Name)
		}
		t.Logf("layer %d: relative error %.2g, inputs %v", i, e, names)
	}
}

// With a vocabulary over the Neural Engine's 16384 per dimension, the soft cap runs on the logits
// folded into rows, and gives the same logits.
func TestWideVocab(t *testing.T) {
	raw := strings.ReplaceAll(tiny, `"vocab_size": 48, "vocab_size_per_layer_input": 48`, `"vocab_size": 32768, "vocab_size_per_layer_input": 32768`)
	c, err := ParseConfig([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	w := newWeights(c)
	ids := []int32{3, 30000, 17, 32767}
	ref := reference(t, c, w, ids)
	p := build(t, c, w, Options{Options: llm.Options{Seq: len(ids), Format: "fp16"}})
	out := p.run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), ids), "ple": rows(t, w, c.perLayerTable(), ids)})
	if e := relErr(out["y"], ref.logits); e > 1e-4 {
		t.Errorf("logits: relative error %g", e)
	}
}
