package ouro

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/llm/decode"
	"github.com/shibukawa/bonsaiai/internal/milcpu"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
)

// tiny is a small Ouro: 3 layers run 3 times (Ouro-1.4B runs 24 four times), grouped query
// attention (4 heads sharing 2 key and value heads, where Ouro's checkpoints have as many as
// heads) and a head_dim apart from hidden_size / heads, the output layer apart from the embeddings
// as in the checkpoints. Matrix rows are multiples of the lut4 group of 16.
const tiny = `{"architectures": ["OuroForCausalLM"], "model_type": "ouro", "attention_dropout": 0.0,
	"bos_token_id": 0, "eos_token_id": 0, "head_dim": 16, "hidden_act": "silu", "hidden_size": 48,
	"intermediate_size": 64, "layer_types": ["full_attention", "full_attention", "full_attention"],
	"max_position_embeddings": 65536, "max_window_layers": 3, "num_attention_heads": 4,
	"num_hidden_layers": 3, "num_key_value_heads": 2, "rms_norm_eps": 1e-06, "rope_scaling": null,
	"rope_theta": 1000000.0, "sliding_window": null, "tie_word_embeddings": false,
	"total_ut_steps": 3, "early_exit_threshold": 1.0, "use_cache": true,
	"use_sliding_window": false, "vocab_size": 48}`

func tinyConfig(t *testing.T) *Config {
	t.Helper()
	c, err := ParseConfig([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

var tokens = []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 47}

// synthetic is llm.NewSynthetic with the norms' weights around 1, as a checkpoint's are
// (Ouro-1.4B-Thinking's up to 1.9): the synthetic weights' 0.1 would make the outputs of the
// attention and the MLP so small that the norms after them, which take them as they are, square
// them under fp16's smallest values (the programs' norms are chosen for the checkpoint's sizes).
func synthetic(c *Config) llm.Weights { return normsNearOne{llm.NewSynthetic(c)} }

type normsNearOne struct{ llm.Weights }

func (w normsNearOne) Get(name string) ([]float32, []int, error) {
	v, shape, err := w.Weights.Get(name)
	if err == nil && (strings.HasSuffix(name, "norm.weight") || strings.Contains(name, "layernorm")) {
		v = slices.Clone(v)
		for i := range v {
			v[i] += 1
		}
	}
	return v, shape, err
}

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

// refRun is the reference forward pass: the hidden state entering each layer of each pass and
// after the last (before the final norm), and the logits.
type refRun struct {
	hidden [][]float64
	logits []float64
}

// reference is OuroForCausalLM's forward written out directly from modeling_ouro.py, in float64:
// an oracle independent of the program builder and the interpreter. Weights, the embeddings the
// host passes and the rotary tables are rounded to fp16, as the program holds them. With
// early_exit_threshold 1.0 the logits are those of the last pass.
func reference(t *testing.T, c *Config, w llm.Weights, ids []int32) refRun {
	t.Helper()
	d, s, nh, nkv, hd, f := c.HiddenSize, len(ids), c.NumHeads, c.NumKVHeads, c.HeadDim, c.IntermediateSize
	// OuroRMSNorm: weight * x * rsqrt(mean(x^2) + eps), over rows of n
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
	// apply_rotary_pos_emb on each head of x (s, heads, hd): x*cos + rotate_half(x)*sin
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
	// OuroModel.forward: for current_ut in range(total_ut_steps), every decoder layer, then
	// self.norm, whose output the next pass starts from
	for ut := 0; ut < c.Passes; ut++ {
		for i := 0; i < c.NumLayers; i++ {
			run.hidden = append(run.hidden, slices.Clone(h))
			l := fmt.Sprintf("model.layers.%d.", i)
			// OuroDecoderLayer: input_layernorm, attention, input_layernorm_2, residual
			x := rmsNorm(h, d, l+"input_layernorm.weight")
			q := linear(x, l+"self_attn.q_proj.weight", d, nh*hd)
			k := linear(x, l+"self_attn.k_proj.weight", d, nkv*hd)
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
			a := linear(o, l+"self_attn.o_proj.weight", nh*hd, d)
			h = add(h, rmsNorm(a, d, l+"input_layernorm_2.weight"))
			// post_attention_layernorm, OuroMLP, post_attention_layernorm_2, residual
			x = rmsNorm(h, d, l+"post_attention_layernorm.weight")
			gate := linear(x, l+"mlp.gate_proj.weight", d, f)
			up := linear(x, l+"mlp.up_proj.weight", d, f)
			for j := range gate {
				gate[j] = gate[j] / (1 + math.Exp(-gate[j])) * up[j]
			}
			h = add(h, rmsNorm(linear(gate, l+"mlp.down_proj.weight", f, d), d, l+"post_attention_layernorm_2.weight"))
		}
		if ut < c.Passes-1 {
			h = rmsNorm(h, d, "model.norm.weight")
		}
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
	weights   int64 // the size of its weight file
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
	fi, err := f.Stat()
	if err != nil {
		t.Fatal(err)
	}
	p, err := milcpu.Load(prog, f)
	if err != nil {
		t.Fatal(err)
	}
	return program{p, ins, outs, fi.Size()}
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
// layer apart from the embeddings (the checkpoints) and tied to them, and for one pass and three.
func TestMatchesReference(t *testing.T) {
	for _, k := range []struct {
		tied   bool
		passes int
	}{{false, 3}, {true, 3}, {false, 1}} {
		raw := strings.Replace(tiny, `"total_ut_steps": 3`, fmt.Sprintf(`"total_ut_steps": %d`, k.passes), 1)
		if k.tied {
			raw = strings.Replace(raw, `"tie_word_embeddings": false`, `"tie_word_embeddings": true`, 1)
		}
		c, err := ParseConfig([]byte(raw))
		if err != nil {
			t.Fatal(err)
		}
		w := synthetic(c)
		ref := reference(t, c, w, tokens)
		p := build(t, c, w, llm.Options{Seq: len(tokens), Format: "fp16"})
		out := p.run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})
		if e := relErr(out["y"], ref.logits); e > 1e-4 {
			t.Errorf("tied %v, %d passes: logits: relative error %g", k.tied, k.passes, e)
		} else {
			t.Logf("tied %v, %d passes: logits: relative error %.2g", k.tied, k.passes, e)
		}
		if len(p.outs) != 1 || len(p.ins) != 1 {
			t.Errorf("the whole model has inputs %v and outputs %v besides x and y", p.ins[1:], p.outs[1:])
		}
		if _, ok := c.Tensors()["lm_head.weight"]; ok == k.tied {
			t.Errorf("tied %v: reads lm_head %v", k.tied, ok)
		}
	}
}

// The passes read the weights once: the program of three passes stores what the one of one pass
// does.
func TestPassesShareWeights(t *testing.T) {
	sizes := map[int]int64{}
	for _, passes := range []int{1, 3} {
		c, err := ParseConfig([]byte(strings.Replace(tiny, `"total_ut_steps": 3`, fmt.Sprintf(`"total_ut_steps": %d`, passes), 1)))
		if err != nil {
			t.Fatal(err)
		}
		w := synthetic(c)
		sizes[passes] = build(t, c, w, llm.Options{Seq: len(tokens), Format: "lut4"}).weights
	}
	if sizes[1] != sizes[3] {
		t.Errorf("weights of 1 pass %d bytes, of 3 passes %d", sizes[1], sizes[3])
	}
}

// Each layer of each pass on its own, fed the reference's hidden state, gives the reference's next
// hidden state (the last layer of a pass but the last with model.norm after it); so does the head
// after the last layer. This is what bonsaiai-dev check does with a real checkpoint.
func TestLayerByLayer(t *testing.T) {
	c := tinyConfig(t)
	w := synthetic(c)
	ref := reference(t, c, w, tokens)
	n := c.Layers()
	if n != 9 || len(ref.hidden) != n+1 {
		t.Fatalf("%d layers, %d reference states", n, len(ref.hidden))
	}
	for i := 0; i <= n; i++ {
		p := build(t, c, w, llm.Options{Seq: len(tokens), Format: "fp16", Start: i, Layers: 1, NoHead: i < n})
		feed := map[string][]float32{"x": f32(ref.hidden[i])}
		if i == 0 {
			feed["x"] = rows(t, w, c.Embedding(), tokens)
		}
		out := p.run(t, feed)
		want := ref.logits
		if i < n {
			want = ref.hidden[i+1]
		}
		e := relErr(out["y"], want)
		if e > 1e-4 {
			t.Errorf("layer %d: relative error %g", i, e)
		}
		t.Logf("layer %d: relative error %.2g", i, e)
	}
}

// The model in parts gives the logits of the whole, parts cut inside a pass and at its end.
func TestPartsMatchWhole(t *testing.T) {
	c := tinyConfig(t)
	w := synthetic(c)
	n := len(tokens)
	feed := map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)}
	whole := build(t, c, w, llm.Options{Seq: n, Format: "fp16"}).run(t, feed)["y"]
	for _, parts := range []int{2, 3, 4} {
		path, plan := writeParts(t, c, w, llm.Options{Seq: n, Format: "fp16"}, parts)
		p := cpuProgram(t, path)
		bonsaiai.EncodeF16(p.In["x"], feed["x"])
		if err := p.Run(); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, len(whole))
		bonsaiai.DecodeF16(got, p.Out["y"])
		if e := relErr32(got, whole); e > 2e-3 {
			t.Errorf("%d parts %v: relative error %g", parts, plan, e)
		}
	}
}

// Stepping the program with a KV cache through a sequence gives the logits of the program without
// one at every position, each pass with its own caches: RoPE positions and padded steps included,
// in one part and in several, split at the attention too.
func TestCachedMatchesFull(t *testing.T) {
	c := tinyConfig(t)
	w := synthetic(c)
	n := len(tokens)
	full := build(t, c, w, llm.Options{Seq: n, Format: "fp16"}).run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})["y"]
	for _, k := range []struct{ q, cache, parts int }{{1, 11, 1}, {1, 12, 3}, {3, 11, 2}, {4, 16, 2}, {11, 11, 1}} {
		t.Run(fmt.Sprintf("q%d cache%d parts%d", k.q, k.cache, k.parts), func(t *testing.T) {
			path, plan := writeParts(t, c, w, llm.Options{Seq: k.q, Cache: k.cache, Format: "fp16"}, k.parts)
			s, err := decode.New(cpuProgram(t, path), c)
			if err != nil {
				t.Fatal(err)
			}
			worst := 0.0
			for pos := 0; pos < n; {
				v := min(k.q, n-pos)
				logits := step(t, s, c, w, pos, pos+v)
				pos += v
				worst = math.Max(worst, relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]))
			}
			// the cache holds fp16 keys and values; the program without one keeps them in fp32
			if worst > 5e-3 {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			t.Logf("parts %v: worst relative error %.2g", plan, worst)
		})
	}
}

// The parts are planned by the weights each pass of a layer reads (llm.Looped), not as if the
// passes after the first had none.
func TestEstimateBytesPerPass(t *testing.T) {
	c := tinyConfig(t)
	_, layers, _ := llm.EstimateBytes(c, llm.Options{Format: "lut4"})
	if len(layers) != c.Layers() {
		t.Fatalf("%d layers estimated, want %d", len(layers), c.Layers())
	}
	for i, b := range layers {
		if b == 0 || b != layers[c.WeightLayer(i)] {
			t.Errorf("layer %d: %d bytes, layer %d %d", i, b, c.WeightLayer(i), layers[c.WeightLayer(i)])
		}
	}
}

// The settings the conversion does not build are refused.
func TestParseConfigRejects(t *testing.T) {
	for _, c := range []struct{ old, new, want string }{
		{`"model_type": "ouro"`, `"model_type": "qwen2"`, "model_type"},
		{`"hidden_act": "silu"`, `"hidden_act": "gelu"`, "hidden_act"},
		{`"use_sliding_window": false`, `"use_sliding_window": true`, "use_sliding_window"},
		{`"rope_scaling": null`, `"rope_scaling": {"rope_type": "yarn", "factor": 4.0}`, "rope_scaling"},
		{`"num_key_value_heads": 2`, `"num_key_value_heads": 3`, "heads"},
		{`"head_dim": 16`, `"head_dim": 15`, "head_dim"},
		{`"total_ut_steps": 3`, `"total_ut_steps": 0`, "total_ut_steps"},
		{`"early_exit_threshold": 1.0`, `"early_exit_threshold": 0.9`, "early_exit_threshold"},
		{`"early_exit_threshold": 1.0`, `"early_exit_step": 2`, "early_exit_step"},
		{`["full_attention", "full_attention", "full_attention"]`, `["full_attention", "sliding_attention", "full_attention"]`, "sliding_attention"},
	} {
		raw := strings.Replace(tiny, c.old, c.new, 1)
		if raw == tiny {
			t.Fatalf("%q is not in the tiny config", c.old)
		}
		if _, err := ParseConfig([]byte(raw)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: error %v, want one about %s", c.new, err, c.want)
		}
	}
	for _, field := range []struct {
		from, to string
	}{
		{`"head_dim": 16`, `"head_dim": 2147483648`},
		{`"num_attention_heads": 4`, `"num_attention_heads": 2147483648`},
	} {
		raw := strings.Replace(tiny, field.from, field.to, 1)
		if _, err := ParseConfig([]byte(raw)); err == nil {
			t.Errorf("%s: accepted an oversized dimension", field.to)
		}
	}
}

func TestBuildRejectsOversizedSequence(t *testing.T) {
	if strconv.IntSize < 64 {
		t.Skip("int cannot exceed int32 on this architecture")
	}
	seq := int(int64(1) << 31)
	if _, _, err := Build(nil, tinyConfig(t), nil, llm.Options{Seq: seq}); err == nil {
		t.Fatal("accepted an oversized sequence")
	}
}
