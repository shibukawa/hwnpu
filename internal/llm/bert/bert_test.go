package bert

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

const tiny = `{"architectures": ["BertModel"], "model_type": "bert", "vocab_size": 40, "hidden_size": 32,
	"num_hidden_layers": 2, "num_attention_heads": 4, "intermediate_size": 64, "hidden_act": "gelu",
	"max_position_embeddings": 16, "type_vocab_size": 2, "layer_norm_eps": 1e-12}`

// f16 reads a tensor rounded to fp16, as the program stores it.
func f16(t *testing.T, w llm.Weights, name string) []float64 {
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

// reference is BertModel's forward written out directly from modeling_bert.py, in float64:
// an oracle independent of the program builder and the interpreter.
func reference(t *testing.T, c *Config, w llm.Weights, ids []int32, mask []float64) []float64 {
	d, s, nh := c.HiddenSize, len(ids), c.NumHeads
	hd := d / nh
	p := c.Prefix
	layerNorm := func(x []float64, name string) {
		g, b := f16(t, w, name+".weight"), f16(t, w, name+".bias")
		for r := 0; r < s; r++ {
			row := x[r*d : (r+1)*d]
			var mean, v float64
			for _, a := range row {
				mean += a
			}
			mean /= float64(d)
			for _, a := range row {
				v += (a - mean) * (a - mean)
			}
			inv := 1 / math.Sqrt(v/float64(d)+c.LayerNormEps)
			for j := range row {
				row[j] = (row[j]-mean)*inv*g[j] + b[j]
			}
		}
	}
	linear := func(x []float64, name string, in, out int) []float64 {
		wt, b := f16(t, w, name+".weight"), f16(t, w, name+".bias")
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
	word := f16(t, w, c.Embedding())
	pos, typ := f16(t, w, p+"embeddings.position_embeddings.weight"), f16(t, w, p+"embeddings.token_type_embeddings.weight")
	h := make([]float64, s*d)
	for r, id := range ids {
		for j := 0; j < d; j++ {
			h[r*d+j] = word[int(id)*d+j] + pos[r*d+j] + typ[j]
		}
	}
	layerNorm(h, p+"embeddings.LayerNorm")
	for i := 0; i < c.NumLayers; i++ {
		l := p + "encoder.layer." + string(rune('0'+i)) + "."
		q, k, v := linear(h, l+"attention.self.query", d, d), linear(h, l+"attention.self.key", d, d), linear(h, l+"attention.self.value", d, d)
		ctx := make([]float64, s*d)
		for head := 0; head < nh; head++ {
			for a := 0; a < s; a++ {
				scores := make([]float64, s)
				m := math.Inf(-1)
				for b := 0; b < s; b++ {
					var dot float64
					for j := 0; j < hd; j++ {
						dot += q[a*d+head*hd+j] * k[b*d+head*hd+j]
					}
					scores[b] = dot/math.Sqrt(float64(hd)) + (1-mask[b])*-1e4
					m = math.Max(m, scores[b])
				}
				var z float64
				for b := range scores {
					scores[b] = math.Exp(scores[b] - m)
					z += scores[b]
				}
				for b := range scores {
					for j := 0; j < hd; j++ {
						ctx[a*d+head*hd+j] += scores[b] / z * v[b*d+head*hd+j]
					}
				}
			}
		}
		o := linear(ctx, l+"attention.output.dense", d, d)
		for j := range o {
			o[j] += h[j]
		}
		layerNorm(o, l+"attention.output.LayerNorm")
		h = o
		f := linear(h, l+"intermediate.dense", d, c.IntermediateSize)
		for j, a := range f {
			switch c.HiddenAct {
			case "gelu":
				f[j] = 0.5 * a * (1 + math.Erf(a/math.Sqrt2))
			case "gelu_new":
				f[j] = 0.5 * a * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(a+0.044715*a*a*a)))
			}
		}
		f = linear(f, l+"output.dense", c.IntermediateSize, d)
		for j := range f {
			f[j] += h[j]
		}
		layerNorm(f, l+"output.LayerNorm")
		h = f
	}
	return h
}

// run builds the program and runs it on the CPU.
func run(t *testing.T, c *Config, w llm.Weights, opts llm.Options, x, mask []float32) []float32 {
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
	res, err := p.Run(map[string]milcpu.Tensor{
		ins[0].Name: {Shape: ins[0].Shape, Data: x},
		ins[1].Name: {Shape: ins[1].Shape, Data: mask},
	}, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return res[out.Name].Data
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

// The program computes BertModel's last hidden state, with and without padding, for both GELU
// forms and for task checkpoints whose tensors carry the bert. prefix.
func TestAgainstReference(t *testing.T) {
	ids := []int32{2, 17, 5, 33, 9, 0}
	for _, c := range []struct {
		name, edit string
		mask       []float64
	}{
		{"no padding", "", []float64{1, 1, 1, 1, 1, 1}},
		{"padding", "", []float64{1, 1, 1, 1, 0, 0}},
		{"gelu_new", `"hidden_act": "gelu"→"hidden_act": "gelu_new"`, []float64{1, 1, 1, 1, 1, 0}},
		{"masked lm checkpoint", `"BertModel"→"BertForMaskedLM"`, []float64{1, 1, 1, 1, 1, 1}},
	} {
		t.Run(c.name, func(t *testing.T) {
			js := tiny
			if c.edit != "" {
				from, to, _ := strings.Cut(c.edit, "→")
				js = strings.Replace(js, from, to, 1)
			}
			cfg, err := ParseConfig([]byte(js))
			if err != nil {
				t.Fatal(err)
			}
			w := llm.NewSynthetic(cfg)
			x, err := llm.Embed(w, cfg, ids)
			if err != nil {
				t.Fatal(err)
			}
			mask := make([]float32, len(c.mask))
			for i, m := range c.mask {
				mask[i] = float32(m)
			}
			want := reference(t, cfg, w, ids, c.mask)
			got := run(t, cfg, w, llm.Options{Seq: len(ids), Format: "fp16"}, x, mask)
			if e := relErr(got, want); e > 1e-4 {
				t.Errorf("relative error %g against the reference", e)
			}
			// one layer at a time gives the same result
			h := x
			for i := 0; i < cfg.NumLayers; i++ {
				h = run(t, cfg, w, llm.Options{Seq: len(ids), Format: "fp16", Start: i, Layers: 1}, h, mask)
			}
			if e := relErr(h, want); e > 1e-4 {
				t.Errorf("layer by layer: relative error %g against the reference", e)
			}
			if cfg.Prefix != "" && !strings.HasPrefix(cfg.Embedding(), "bert.") {
				t.Errorf("prefix %q, embedding %s", cfg.Prefix, cfg.Embedding())
			}
		})
	}
}

func TestConfig(t *testing.T) {
	for _, bad := range []string{
		`{"hidden_size": 32, "num_attention_heads": 5}`,
		`{"hidden_size": 32, "num_attention_heads": 4, "position_embedding_type": "relative_key"}`,
		`{"hidden_size": 32, "num_attention_heads": 4, "hidden_act": "swish"}`,
	} {
		if _, err := ParseConfig([]byte(bad)); err == nil {
			t.Errorf("accepted %s", bad)
		}
	}
	cfg, _ := ParseConfig([]byte(tiny))
	b := mlpackage.NewBuilder("CoreML8", nil)
	if _, _, err := cfg.Build(b, llm.NewSynthetic(cfg), llm.Options{Seq: 17}); err == nil {
		t.Error("sequence longer than the position table accepted")
	}
	if a, err := llm.Detect([]byte(tiny)); err != nil || a != Arch {
		t.Errorf("detected %v, %v", a, err)
	}
}
