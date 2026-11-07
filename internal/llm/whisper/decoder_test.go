package whisper

import (
	"context"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

func tinyDecoder(t *testing.T, edit string) *Decoder {
	t.Helper()
	js := tiny
	if edit != "" {
		from, to, _ := strings.Cut(edit, "→")
		js = strings.Replace(js, from, to, 1)
	}
	d, err := ParseDecoderConfig([]byte(js))
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// weights holds the tensors of both the decoder and the cross program (one synthetic checkpoint).
func weights(d *Decoder) llm.Synthetic {
	w := llm.NewSynthetic(d)
	for n, s := range (&Cross{d.Config}).Tensors() {
		w[n] = s
	}
	return w
}

var tokens = []int32{3, 17, 5, 33, 9, 0, 21, 12, 39}

// randomEncoding is an encoder output, rounded to fp16.
func randomEncoding(c *Config, seed uint64) []float64 {
	rng := rand.New(rand.NewPCG(seed, 2))
	enc := make([]float64, c.SourcePositions*c.DModel)
	for i := range enc {
		enc[i] = r16(rng.NormFloat64())
	}
	return enc
}

// decoderRef is WhisperDecoder's forward over tokens from position 0 attending over the encoder
// output enc, with the output layer tied to the token embeddings: each layer's input, and the
// logits after each token.
func decoderRef(t *testing.T, d *Decoder, w llm.Weights, ids []int32, enc []float64) (hidden [][]float64, logits []float64) {
	r := ref{t, w, d.Activation}
	dm, p := d.DModel, d.decoder()
	table, pos := f16(t, w, d.Embedding()), f16(t, w, p+"embed_positions.weight")
	scale := 1.0
	if d.ScaleEmbedding {
		scale = r16(math.Sqrt(float64(dm))) // a constant of the program, in fp16 as the weights
	}
	h := make([]float64, len(ids)*dm)
	for s, id := range ids {
		for j := 0; j < dm; j++ {
			h[s*dm+j] = table[int(id)*dm+j]*scale + pos[s*dm+j]
		}
	}
	causal := func(a, b int) bool { return b <= a }
	for i := 0; i < d.DecoderLayers; i++ {
		hidden = append(hidden, h)
		l := d.decoderLayer(i)
		a := r.layerNorm(h, dm, l+"self_attn_layer_norm")
		h = add(h, r.attention(l+"self_attn.", a, a, dm, d.DecoderHeads, causal))
		a = r.layerNorm(h, dm, l+"encoder_attn_layer_norm")
		h = add(h, r.attention(l+"encoder_attn.", a, enc, dm, d.DecoderHeads, nil))
		f := r.layerNorm(h, dm, l+"final_layer_norm")
		h = add(h, r.linear(r.activation(r.linear(f, l+"fc1", dm, d.DecoderFFN, true)), l+"fc2", d.DecoderFFN, dm, true))
	}
	hidden = append(hidden, h)
	h = r.layerNorm(h, dm, p+"layer_norm")
	logits = make([]float64, len(ids)*d.VocabSize)
	for s := range ids {
		for v := 0; v < d.VocabSize; v++ {
			var acc float64
			for j := 0; j < dm; j++ {
				acc += h[s*dm+j] * table[v*dm+j]
			}
			logits[s*d.VocabSize+v] = acc
		}
	}
	return hidden, logits
}

// crossKV runs the cross program over enc and returns the decoder's inputs xk_N and xv_N.
func crossKV(t *testing.T, c *Config, w llm.Weights, enc []float64) map[string][]float32 {
	t.Helper()
	cr := &Cross{c}
	p, ins, outs := program(t, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
		ins, out, err := cr.Build(b, w, llm.Options{Format: "fp16"})
		return ins, []mlpackage.Feature{out}, err
	})
	res, err := p.Run(map[string]milcpu.Tensor{"x": {Shape: ins[0].Shape, Data: f32(enc)}}, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	y := res[outs[0].Name]
	if want := []int{1, c.CrossHeads(), c.SourcePositions, c.DModel / c.DecoderHeads}; fmt.Sprint(y.Shape) != fmt.Sprint(want) {
		t.Fatalf("cross output %v, want %v", y.Shape, want)
	}
	n := c.DecoderHeads * c.SourcePositions * c.DModel / c.DecoderHeads
	kv := map[string][]float32{}
	for i := 0; i < c.DecoderLayers; i++ {
		kv[fmt.Sprintf("xk_%d", i)] = y.Data[2*i*n : (2*i+1)*n]
		kv[fmt.Sprintf("xv_%d", i)] = y.Data[(2*i+1)*n : (2*i+2)*n]
	}
	return kv
}

func runDecoder(t *testing.T, d *Decoder, w llm.Weights, opts llm.Options, x []float32, kv map[string][]float32) []float32 {
	t.Helper()
	p, ins, outs := program(t, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
		ins, out, err := d.Build(b, w, opts)
		return ins, []mlpackage.Feature{out}, err
	})
	feed := map[string]milcpu.Tensor{}
	for _, in := range ins {
		v := x
		if in.Name != "x" {
			v = kv[in.Name]
		}
		feed[in.Name] = milcpu.Tensor{Shape: in.Shape, Data: v}
	}
	res, err := p.Run(feed, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return res[outs[0].Name].Data
}

// The decoder without a cache computes WhisperDecoder's logits over the cross attention's keys
// and values from the cross program, for both GELU forms and with scale_embedding; one layer at a
// time gives each layer's output, and what follows the last layer the logits.
func TestDecoderAgainstReference(t *testing.T) {
	for _, c := range []struct{ name, edit string }{
		{"gelu", ""},
		{"gelu_new", `"activation_function": "gelu"→"activation_function": "gelu_new"`},
		{"scale_embedding", `"scale_embedding": false→"scale_embedding": true`},
	} {
		t.Run(c.name, func(t *testing.T) {
			d := tinyDecoder(t, c.edit)
			w := weights(d)
			enc := randomEncoding(d.Config, 3)
			hidden, want := decoderRef(t, d, w, tokens, enc)
			kv := crossKV(t, d.Config, w, enc)
			x, err := llm.Embed(w, d, tokens)
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range x {
				x[i] = float32(r16(float64(v))) // as the session writes it, and the reference reads the table
			}
			n := len(tokens)
			got := runDecoder(t, d, w, llm.Options{Seq: n, Format: "fp16"}, x, kv)
			if err := relErr(got, want); err > 1e-4 {
				t.Errorf("relative error %g against the reference", err)
			}
			first := runDecoder(t, d, w, llm.Options{Seq: n, Format: "fp16", Layers: 1, NoHead: true}, x, kv)
			if err := relErr(first, hidden[1]); err > 1e-4 {
				t.Errorf("the embedding and layer 0: relative error %g", err)
			}
			for i := 1; i < d.DecoderLayers; i++ {
				out := runDecoder(t, d, w, llm.Options{Seq: n, Format: "fp16", Start: i, Layers: 1, NoHead: true}, f32(hidden[i]), kv)
				if err := relErr(out, hidden[i+1]); err > 1e-4 {
					t.Errorf("layer %d alone: relative error %g", i, err)
				}
			}
			head := runDecoder(t, d, w, llm.Options{Seq: n, Format: "fp16", Start: d.DecoderLayers}, f32(hidden[d.DecoderLayers]), kv)
			if err := relErr(head, want); err > 1e-4 {
				t.Errorf("after the last layer: relative error %g", err)
			}
		})
	}
}

// Programs of 1 and 3 tokens per call with a KV cache, sharing one sequence in a decode.Session
// with the clip's keys and values set once (SetInputF16), give the logits of the program without
// a cache after every step, a padded step and a rewind to a mark among them.
func TestDecoderCached(t *testing.T) {
	d := tinyDecoder(t, "")
	w := weights(d)
	enc := randomEncoding(d.Config, 4)
	kv := crossKV(t, d.Config, w, enc)
	_, want := decoderRef(t, d, w, tokens, enc)
	const cache = 16
	dir := t.TempDir()
	var paths []string
	for _, q := range []int{1, 3} {
		path, err := llm.Convert(Arch, d, w, filepath.Join(dir, fmt.Sprintf("d-q%d.mlpackage", q)), "none", 0, llm.Options{Seq: q, Cache: cache, Format: "fp16"}, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	s, cl, err := runner.OpenSession(context.Background(), nil, paths, d)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	for name, v := range kv {
		b := make([]byte, 2*len(v))
		hwnpu.EncodeF16(b, v)
		if err := s.SetInputF16(name, b); err != nil {
			t.Fatal(err)
		}
	}
	if err := s.SetInputF16("kv_k_0", nil); err == nil {
		t.Error("SetInputF16 took a cache input")
	}
	step := func(from, to int) []float32 {
		t.Helper()
		var rows [][]float32
		for i := from; i < to; i++ {
			row, err := llm.TokenRow(w, d, tokens[i])
			if err != nil {
				t.Fatal(err)
			}
			rows = append(rows, row)
		}
		logits, err := s.Step(rows)
		if err != nil {
			t.Fatal(err)
		}
		return logits
	}
	worst := 0.0
	check := func(what string, logits []float32, pos int) {
		t.Helper()
		e := relErr(logits, want[(pos-1)*d.VocabSize:pos*d.VocabSize])
		worst = math.Max(worst, e)
		if e > 5e-3 {
			t.Errorf("%s: relative error %g after %d tokens", what, e, pos)
		}
	}
	check("1", step(0, 1), 1)
	check("2 padded", step(1, 3), 3)
	m := s.Mark()
	check("3", step(3, 6), 6)
	s.Rewind(m)
	check("rewound, 1", step(3, 4), 4)
	check("3 more", step(4, 7), 7)
	check("2 padded", step(7, 9), 9)
	t.Logf("worst relative error %.2g", worst)
}

func TestDecoderStepInputs(t *testing.T) {
	d := tinyDecoder(t, "")
	in := d.StepInputs(14, 2, 3, 16)
	sel, mask := in["pos"], in["mask"]
	n := d.TargetPositions
	for row, want := range []int{14, 15, 15} { // the padding past the last position is clamped
		for j := 0; j < n; j++ {
			if (sel[row*n+j] == 1) != (j == want) {
				t.Fatalf("pos row %d: %v", row, sel[row*n:(row+1)*n])
			}
		}
	}
	keys := 16 + 3
	for row := 0; row < 3; row++ {
		for j := 0; j < keys; j++ {
			visible := j < 14 || j >= 16 && j-16 <= row
			if (mask[row*keys+j] == 0) != visible {
				t.Errorf("mask row %d key %d: %g", row, j, mask[row*keys+j])
			}
		}
	}
	b := mlpackage.NewBuilder("CoreML8", nil)
	if _, _, err := d.Build(b, weights(d), llm.Options{Seq: 17}); err == nil {
		t.Error("17 tokens over 16 positions accepted")
	}
	if _, _, err := d.BuildCached(b, weights(d), llm.Options{Seq: 1, Cache: 17}); err == nil {
		t.Error("a cache of 17 over 16 positions accepted")
	}
	if _, _, err := d.Build(b, weights(d), llm.Options{Seq: 4, Cache: 8}); err == nil {
		t.Error("Build with a cache accepted")
	}
	if a, err := llm.Detect([]byte(tiny)); err != nil || a != Arch {
		t.Errorf("detected %v, %v", a, err)
	}
}

// In 8 bit tables the output layer, whose 40 rows are not a multiple of the tables' 16, is padded
// with zero rows and cut back to the vocabulary: the logits are fp16's, within the tables' error.
func TestDecoderLUTPadded(t *testing.T) {
	d := tinyDecoder(t, "")
	w := weights(d)
	enc := randomEncoding(d.Config, 5)
	kv := crossKV(t, d.Config, w, enc)
	x, err := llm.Embed(w, d, tokens)
	if err != nil {
		t.Fatal(err)
	}
	n := len(tokens)
	ref := runDecoder(t, d, w, llm.Options{Seq: n, Format: "fp16"}, x, kv)
	got := runDecoder(t, d, w, llm.Options{Seq: n, Format: "lut8"}, x, kv)
	if len(got) != n*d.VocabSize {
		t.Fatalf("%d logits, want %d", len(got), n*d.VocabSize)
	}
	want := make([]float64, len(ref))
	for i, v := range ref {
		want[i] = float64(v)
	}
	if e := relErr(got, want); e > 0.05 {
		t.Errorf("lut8: relative error %g against fp16", e)
	}
}
