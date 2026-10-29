package main

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/bert"
	"github.com/shibukawa/hwnpu/internal/llm/check"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

const tinyLFM3 = `{"model_type": "lfm3", "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 3,
	"num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 8, "vocab_size": 48, "norm_eps": 1e-5,
	"conv_L_cache": 3, "conv_bias": true, "sliding_window": 3, "use_sandwich_norm": true,
	"layer_types": ["conv", "full_attention", "sliding_attention"],
	"rope_parameters": {"full_attention": {"rope_theta": 10000, "partial_rotary_factor": 0.5}, "sliding_attention": {"rope_theta": 10000}}}`

// scaled is a checkpoint with one tensor multiplied by a factor.
type scaled struct {
	llm.Synthetic
	name   string
	factor float32
}

func (w scaled) Get(name string) ([]float32, []int, error) {
	v, shape, err := w.Synthetic.Get(name)
	if name == w.name {
		for i := range v {
			v[i] *= w.factor
		}
	}
	return v, shape, err
}

// writeRef runs the model on the CPU in fp32 and writes the result as a reference at prefix.
func writeRef(t *testing.T, model llm.Model, w llm.Weights, ids []int32, prefix string) {
	t.Helper()
	nl := model.Layers()
	opts := llm.Options{Seq: len(ids), Format: "fp16"}
	h, err := llm.Embed(w, model, ids)
	if err != nil {
		t.Fatal(err)
	}
	hidden := append([]float32(nil), h...)
	for i := 0; i < nl; i++ {
		o := opts
		o.Start, o.Layers, o.NoHead = i, 1, true
		if h, err = buildRun(model, w, o, h, t.TempDir(), false); err != nil {
			t.Fatal(err)
		}
		hidden = append(hidden, h...)
	}
	o := opts
	o.Start = nl
	logits, err := buildRun(model, w, o, h, t.TempDir(), false)
	if err != nil {
		t.Fatal(err)
	}
	put := func(ext string, n int, at func(i int) uint32) {
		raw := make([]byte, 4*n)
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint32(raw[4*i:], at(i))
		}
		if err := os.WriteFile(prefix+ext, raw, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	put(".ids.bin", len(ids), func(i int) uint32 { return uint32(ids[i]) })
	put(".hidden.bin", len(hidden), func(i int) uint32 { return math.Float32bits(hidden[i]) })
	put(".logits.bin", len(logits), func(i int) uint32 { return math.Float32bits(logits[i]) })
}

func TestCheck(t *testing.T) {
	model, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(model)
	ids := []int32{3, 1, 4, 1, 5, 9, 2}
	opts := llm.Options{Format: "fp16", Params: map[string]string{"post-scale": "64"}}

	// a port that matches its reference passes every layer
	good := filepath.Join(t.TempDir(), "good")
	writeRef(t, model, w, ids, good)
	ref, err := check.LoadRef(good)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	first, err := checkModel(&out, model, w, ref, []int{0, 1, 2}, true, false, 1e-4, opts)
	if err != nil || first != -1 {
		t.Fatalf("first bad layer %d, %v\n%s", first, err, &out)
	}
	if !strings.Contains(out.String(), "tensors: 0 missing, 0 mismatched, 0 unused") || !strings.Contains(out.String(), "argmax agreement 7/7") {
		t.Errorf("unexpected report:\n%s", &out)
	}
	t.Logf("\n%s", &out)

	// a reference whose layer 1 differs: only layer 1 is flagged, the layers after it are fed
	// the reference's own input and still match
	bad := filepath.Join(t.TempDir(), "bad")
	writeRef(t, model, scaled{w, "model.layers.1.feed_forward.w2.weight", 1.5}, ids, bad)
	if ref, err = check.LoadRef(bad); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	first, err = checkModel(&out, model, w, ref, []int{0, 1, 2}, false, false, 1e-4, opts)
	if err != nil || first != 1 || strings.Count(out.String(), "<- over -tol") != 1 {
		t.Errorf("first bad layer %d, %v\n%s", first, err, &out)
	}

	// tensors the model does not read show up in the inventory
	extra := llm.Synthetic{}
	for k, v := range w {
		extra[k] = v
	}
	extra["model.layers.0.conv.in_proj.bias"] = []int{192}
	out.Reset()
	if _, err := checkModel(&out, model, extra, ref, nil, false, false, 1, opts); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "unused tensor: model.layers.0.conv.in_proj.bias") {
		t.Errorf("unused tensor not reported:\n%s", &out)
	}
}

// -package checks a written package (lut4 here) against the reference logits.
func TestCheckPackage(t *testing.T) {
	model, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(model)
	prefix := filepath.Join(t.TempDir(), "ref")
	writeRef(t, model, w, []int32{3, 1, 4, 1, 5, 9, 2}, prefix)
	dir := filepath.Join(t.TempDir(), "m.mlpackage")
	pkg, err := mlpackage.Create(dir)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	ins, y, err := model.Build(b, w, llm.Options{Seq: 7, Format: "lut4"})
	if err != nil {
		t.Fatal(err)
	}
	if err := pkg.Finish(b.Finish(&mlpackage.Var{Name: y.Name}), 9, ins, []mlpackage.Feature{y}); err != nil {
		t.Fatal(err)
	}
	if err := checkPackage(dir, prefix, true); err != nil {
		t.Fatal(err)
	}
	if err := checkPackage(dir, filepath.Join(t.TempDir(), "none"), false); err == nil {
		t.Error("missing reference accepted")
	}
}

// Encoders (BERT): a second input (the attention mask, fed ones) and hidden state outputs,
// compared without argmax and KL.
func TestCheckEncoder(t *testing.T) {
	model, err := bert.ParseConfig([]byte(`{"model_type": "bert", "vocab_size": 40, "hidden_size": 32, "num_hidden_layers": 2,
		"num_attention_heads": 4, "intermediate_size": 64, "max_position_embeddings": 16}`))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(model)
	prefix := filepath.Join(t.TempDir(), "ref")
	writeRef(t, model, w, []int32{2, 17, 5, 33, 9}, prefix)
	ref, err := check.LoadRef(prefix)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	first, err := checkModel(&out, model, w, ref, []int{0, 1}, true, false, 1e-4, llm.Options{Format: "fp16"})
	if err != nil || first != -1 || strings.Contains(out.String(), "argmax") || !strings.Contains(out.String(), "output 32;") {
		t.Errorf("first bad layer %d, %v\n%s", first, err, &out)
	}
	t.Logf("\n%s", &out)
}
