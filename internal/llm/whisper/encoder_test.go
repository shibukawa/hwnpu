package whisper

import (
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

const tiny = `{"architectures": ["WhisperForConditionalGeneration"], "model_type": "whisper", "d_model": 32,
	"encoder_layers": 2, "encoder_attention_heads": 4, "encoder_ffn_dim": 64,
	"decoder_layers": 2, "decoder_attention_heads": 4, "decoder_ffn_dim": 48,
	"num_mel_bins": 8, "max_source_positions": 6, "max_target_positions": 16, "vocab_size": 40,
	"activation_function": "gelu", "scale_embedding": false}`

func tinyEncoder(t *testing.T, edit string) *Encoder {
	t.Helper()
	js := tiny
	if edit != "" {
		from, to, _ := strings.Cut(edit, "→")
		js = strings.Replace(js, from, to, 1)
	}
	e, err := ParseEncoderConfig([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	return e
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

// ref holds the fp64 operations of the reference forwards, written from modeling_whisper.py
// directly: an oracle independent of the program builder and the interpreter.
type ref struct {
	t   *testing.T
	w   llm.Weights
	act string
}

func (r ref) layerNorm(x []float64, d int, name string) []float64 {
	g, b := f16(r.t, r.w, name+".weight"), f16(r.t, r.w, name+".bias")
	out := make([]float64, len(x))
	for row := 0; row < len(x)/d; row++ {
		v := x[row*d : (row+1)*d]
		var mean, va float64
		for _, a := range v {
			mean += a
		}
		mean /= float64(d)
		for _, a := range v {
			va += (a - mean) * (a - mean)
		}
		inv := 1 / math.Sqrt(va/float64(d)+layerNormEps)
		for j := range v {
			out[row*d+j] = (v[j]-mean)*inv*g[j] + b[j]
		}
	}
	return out
}

// linear is nn.Linear (weight (out, in)), with its bias unless bias is false.
func (r ref) linear(x []float64, name string, in, out int, bias bool) []float64 {
	wt := f16(r.t, r.w, name+".weight")
	var b []float64
	if bias {
		b = f16(r.t, r.w, name+".bias")
	}
	rows := len(x) / in
	y := make([]float64, rows*out)
	for i := 0; i < rows; i++ {
		for o := 0; o < out; o++ {
			var acc float64
			if b != nil {
				acc = b[o]
			}
			for k := 0; k < in; k++ {
				acc += x[i*in+k] * wt[o*in+k]
			}
			y[i*out+o] = acc
		}
	}
	return y
}

func (r ref) activation(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, a := range x {
		switch r.act {
		case "gelu":
			out[i] = 0.5 * a * (1 + math.Erf(a/math.Sqrt2))
		case "gelu_new", "gelu_pytorch_tanh":
			out[i] = 0.5 * a * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(a+0.044715*a*a*a)))
		case "relu":
			out[i] = math.Max(a, 0)
		}
	}
	return out
}

// attention is WhisperAttention of queries from x (rows of d) over keys and values from kv, query
// row a seeing key row b when visible(a, b) (nil: all).
func (r ref) attention(p string, x, kv []float64, d, nh int, visible func(a, b int) bool) []float64 {
	hd := d / nh
	q := r.linear(x, p+"q_proj", d, d, true)
	k := r.linear(kv, p+"k_proj", d, d, false)
	v := r.linear(kv, p+"v_proj", d, d, true)
	sq, sk := len(x)/d, len(kv)/d
	ctx := make([]float64, sq*d)
	for h := 0; h < nh; h++ {
		for a := 0; a < sq; a++ {
			scores := make([]float64, sk)
			m := math.Inf(-1)
			for b := 0; b < sk; b++ {
				if visible != nil && !visible(a, b) {
					scores[b] = math.Inf(-1)
					continue
				}
				var dot float64
				for j := 0; j < hd; j++ {
					dot += q[a*d+h*hd+j] / math.Sqrt(float64(hd)) * k[b*d+h*hd+j]
				}
				scores[b] = dot
				m = math.Max(m, dot)
			}
			var z float64
			for b := range scores {
				scores[b] = math.Exp(scores[b] - m)
				z += scores[b]
			}
			for b := range scores {
				for j := 0; j < hd; j++ {
					ctx[a*d+h*hd+j] += scores[b] / z * v[b*d+h*hd+j]
				}
			}
		}
	}
	return r.linear(ctx, p+"out_proj", d, d, true)
}

func add(x, y []float64) []float64 {
	out := make([]float64, len(x))
	for i := range x {
		out[i] = x[i] + y[i]
	}
	return out
}

// conv is nn.Conv1d of kernel 3 and padding 1 over x (time major rows of in), stride 1 or 2.
func (r ref) conv(x []float64, name string, in, out, stride int) []float64 {
	wt, b := f16(r.t, r.w, name+".weight"), f16(r.t, r.w, name+".bias")
	n := len(x) / in
	no := (n+2-3)/stride + 1
	y := make([]float64, no*out)
	for t := 0; t < no; t++ {
		for o := 0; o < out; o++ {
			acc := b[o]
			for j := 0; j < 3; j++ {
				s := t*stride + j - 1
				if s < 0 || s >= n {
					continue
				}
				for i := 0; i < in; i++ {
					acc += wt[(o*in+i)*3+j] * x[s*in+i]
				}
			}
			y[t*out+o] = acc
		}
	}
	return y
}

// encoderRef is WhisperEncoder's forward over mel frames (time major), returning the hidden state
// entering each layer and the last hidden state after the final LayerNorm.
func encoderRef(t *testing.T, e *Encoder, w llm.Weights, mel []float64) (hidden [][]float64, last []float64) {
	r := ref{t, w, e.Activation}
	d, p := e.DModel, e.encoder()
	h := r.activation(r.conv(mel, p+"conv1", e.MelBins, d, 1))
	h = r.activation(r.conv(h, p+"conv2", d, d, 2))
	h = add(h, f16(t, w, p+"embed_positions.weight"))
	for i := 0; i < e.EncoderLayers; i++ {
		hidden = append(hidden, h)
		l := e.encoderLayer(i)
		a := r.layerNorm(h, d, l+"self_attn_layer_norm")
		h = add(h, r.attention(l+"self_attn.", a, a, d, e.EncoderHeads, nil))
		f := r.layerNorm(h, d, l+"final_layer_norm")
		f = r.linear(r.activation(r.linear(f, l+"fc1", d, e.EncoderFFN, true)), l+"fc2", e.EncoderFFN, d, true)
		h = add(h, f)
	}
	hidden = append(hidden, h)
	return hidden, r.layerNorm(h, d, p+"layer_norm")
}

// program builds a model's program and loads it on the CPU interpreter.
func program(t *testing.T, build func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error)) (*milcpu.Program, []mlpackage.Feature, []mlpackage.Feature) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	ins, outs, err := build(b)
	if err != nil {
		t.Fatal(err)
	}
	vars := make([]*mlpackage.Var, len(outs))
	for i, o := range outs {
		vars[i] = &mlpackage.Var{Name: o.Name}
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
	return p, ins, outs
}

func runEncoder(t *testing.T, e *Encoder, w llm.Weights, opts llm.Options, x []float32) []float32 {
	t.Helper()
	p, ins, outs := program(t, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
		ins, out, err := e.Build(b, w, opts)
		return ins, []mlpackage.Feature{out}, err
	})
	res, err := p.Run(map[string]milcpu.Tensor{"x": {Shape: ins[0].Shape, Data: x}}, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return res[outs[0].Name].Data
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

func f32(x []float64) []float32 {
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = float32(v)
	}
	return out
}

// randomMel is mel frames in the range Features gives, rounded to fp16.
func randomMel(e *Encoder, seed uint64) []float64 {
	rng := rand.New(rand.NewPCG(seed, 1))
	mel := make([]float64, e.Frames()*e.MelBins)
	for i := range mel {
		mel[i] = r16(rng.Float64()*2 - 1)
	}
	return mel
}

// The encoder computes WhisperEncoder's last hidden state from mel frames, with the attention's
// queries in one piece and in chunks that do not divide the positions, for both GELU forms; one
// layer at a time gives each layer's output, and what follows the last layer (Start = Layers)
// the final LayerNorm.
func TestEncoderAgainstReference(t *testing.T) {
	for _, c := range []struct {
		name, edit, qchunk string
	}{
		{"whole", "", "0"},
		{"chunks of 4", "", "4"},
		{"chunks of 1", "", "1"},
		{"gelu_new", `"activation_function": "gelu"→"activation_function": "gelu_new"`, "4"},
	} {
		t.Run(c.name, func(t *testing.T) {
			e := tinyEncoder(t, c.edit)
			w := llm.NewSynthetic(e)
			mel := randomMel(e, 1)
			hidden, want := encoderRef(t, e, w, mel)
			params := map[string]string{"qchunk": c.qchunk}
			got := runEncoder(t, e, w, llm.Options{Format: "fp16", Params: params}, f32(mel))
			if err := relErr(got, want); err > 1e-4 {
				t.Errorf("relative error %g against the reference", err)
			}
			for i := 1; i < e.EncoderLayers; i++ { // a program from layer 0 starts with the embedding (below)
				out := runEncoder(t, e, w, llm.Options{Format: "fp16", Params: params, Start: i, Layers: 1, NoHead: true}, f32(hidden[i]))
				if err := relErr(out, hidden[i+1]); err > 1e-4 {
					t.Errorf("layer %d alone: relative error %g", i, err)
				}
			}
			first := runEncoder(t, e, w, llm.Options{Format: "fp16", Params: params, Layers: 1, NoHead: true}, f32(mel))
			if err := relErr(first, hidden[1]); err > 1e-4 {
				t.Errorf("the embedding and layer 0: relative error %g", err)
			}
			head := runEncoder(t, e, w, llm.Options{Format: "fp16", Params: params, Start: e.EncoderLayers}, f32(hidden[e.EncoderLayers]))
			if err := relErr(head, want); err > 1e-4 {
				t.Errorf("after the last layer: relative error %g", err)
			}
		})
	}
}

func TestConfig(t *testing.T) {
	for _, edit := range []string{
		`"model_type": "whisper"→"model_type": "bert"`,
		`"encoder_attention_heads": 4→"encoder_attention_heads": 5`,
		`"decoder_attention_heads": 4→"decoder_attention_heads": 3`,
		`"activation_function": "gelu"→"activation_function": "swish"`,
		`"WhisperForConditionalGeneration"→"WhisperForAudioClassification"`,
		`"scale_embedding": false→"use_weighted_layer_sum": true`,
		`"encoder_layers": 2→"encoder_layers": 0`,
	} {
		from, to, _ := strings.Cut(edit, "→")
		if !strings.Contains(tiny, from) {
			t.Fatalf("%q is not in the config", from)
		}
		if _, err := ParseConfig([]byte(strings.Replace(tiny, from, to, 1))); err == nil {
			t.Errorf("accepted %s", to)
		}
	}
	e := tinyEncoder(t, "")
	b := mlpackage.NewBuilder("CoreML8", nil)
	if _, _, err := e.Build(b, llm.NewSynthetic(e), llm.Options{Cache: 8}); err == nil {
		t.Error("an encoder with a KV cache was built")
	}
	if _, _, err := e.Build(b, llm.NewSynthetic(e), llm.Options{Params: map[string]string{"qchunk": "-1"}}); err == nil {
		t.Error("qchunk -1 accepted")
	}
	if got := fmt.Sprint(e.Frames(), e.Hidden(), e.Layers()); got != "12 32 2" {
		t.Errorf("frames, hidden, layers %s", got)
	}
}
