package lfm3

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// tinyConfig is a small LFM3 (matrix rows in multiples of the lut4 group of 16) with every layer type, GQA, partial RoPE and a conv bias.
func tinyConfig() *Config {
	half := 0.5
	c := &Config{HiddenSize: 64, IntermediateSize: 96, NumLayers: 3, NumHeads: 16, NumKVHeads: 8, HeadDim: 8, VocabSize: 48,
		NormEps: 1e-5, ConvKernel: 3, ConvBias: true, SlidingWindow: 3, UseSandwichNorm: true,
		LayerTypes: []string{"conv", "full_attention", "sliding_attention"}}
	c.RopeParameters = map[string]struct {
		Theta         float64  `json:"rope_theta"`
		PartialFactor *float64 `json:"partial_rotary_factor"`
		Type          string   `json:"rope_type"`
	}{
		"full_attention":    {Theta: 10000, PartialFactor: &half},
		"sliding_attention": {Theta: 10000},
	}
	return c
}

// build runs a build into a fresh weight file and returns the output description and a digest of
// the program and the weight file.
func build(t *testing.T, f func(b *mlpackage.Builder) (in, out mlpackage.Feature, err error)) (in, out mlpackage.Feature, digest string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	in, out, err = f(b)
	if err != nil {
		t.Fatal(err)
	}
	prog := b.Finish(&mlpackage.Var{Name: out.Name})
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
	return in, out, fmt.Sprintf("program %x weights %x", sha256.Sum256(raw), sha256.Sum256(bin))
}

// Builds the tiny model in each option combination. No Neural Engine is needed. The digest of
// the program and the weight file is logged, so that a refactoring which should not change the
// output can be checked by comparing go test -v output before and after.
func TestBuildTiny(t *testing.T) {
	cfg := tinyConfig()
	w := llm.NewSynthetic(cfg)
	cases := []struct {
		name string
		opts Options
	}{
		{"lut4", Options{Seq: 6}},
		{"fp16 conv", Options{Seq: 6, Format: "fp16", ConvMode: "conv"}},
		{"kmeans post-scale", Options{Seq: 6, Quant: "kmeans", PostScale: 64, KeepFP16: []string{"lm_head"}}},
		{"colblock hidden", Options{Seq: 5, ColBlock: 16, Hidden: true}},
		{"layers 1..2", Options{Seq: 6, Start: 1, Layers: 1, Format: "fp16"}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			in, out, digest := build(t, func(b *mlpackage.Builder) (mlpackage.Feature, mlpackage.Feature, error) {
				return Build(b, cfg, w, c.opts)
			})
			want := cfg.VocabSize
			if c.opts.Hidden || c.opts.Start+c.opts.Layers > 0 && c.opts.Start+c.opts.Layers < cfg.NumLayers {
				want = cfg.HiddenSize
			}
			if got := out.Shape[len(out.Shape)-1]; got != want || in.Shape[1] != c.opts.Seq {
				t.Errorf("in %v, out %v", in.Shape, out.Shape)
			}
			t.Logf("%s", digest)
		})
	}
}

// The path hwnpu-convert takes (detecting the architecture from config.json and passing LFM3's
// settings as parameters) builds the same program as calling Build directly.
func TestArch(t *testing.T) {
	var cfg map[string]any
	raw, _ := json.Marshal(tinyConfig())
	json.Unmarshal(raw, &cfg)
	cfg["model_type"] = "lfm3"
	raw, _ = json.Marshal(cfg)
	a, err := llm.Detect(raw)
	if err != nil || a != Arch {
		t.Fatalf("detected %v, %v", a, err)
	}
	model, err := a.LoadConfig(raw)
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(tinyConfig())
	for _, c := range []struct {
		params map[string]string
		direct Options
	}{
		{nil, Options{Seq: 6, PostScale: 64, ConvMode: "taps"}},
		{map[string]string{"post-scale": "1", "conv": "conv"}, Options{Seq: 6, ConvMode: "conv"}},
	} {
		_, _, got := build(t, func(b *mlpackage.Builder) (mlpackage.Feature, mlpackage.Feature, error) {
			ins, out, err := model.Build(b, w, llm.Options{Seq: 6, Params: c.params})
			if len(ins) != 1 {
				t.Fatalf("inputs %v", ins)
			}
			return ins[0], out, err
		})
		_, _, want := build(t, func(b *mlpackage.Builder) (mlpackage.Feature, mlpackage.Feature, error) {
			return Build(b, tinyConfig(), w, c.direct)
		})
		if got != want {
			t.Errorf("params %v: built differently from %+v", c.params, c.direct)
		}
	}
	if _, err := llm.Detect([]byte(`{"model_type": "llama", "architectures": ["LlamaForCausalLM"]}`)); err == nil {
		t.Error("llama detected as supported")
	}
	if _, err := ParseConfig([]byte(`{"num_hidden_layers": 1, "num_attention_heads": 1, "layer_types": ["mamba"]}`)); err == nil {
		t.Error("unknown layer type accepted")
	}
}

// runCPU builds a program and runs it with the CPU interpreter on x.
func runCPU(t *testing.T, cfg *Config, w llm.Weights, opts Options, x []float32) []float32 {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	in, out, err := Build(b, cfg, w, opts)
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
	res, err := p.Run(map[string]milcpu.Tensor{in.Name: {Shape: in.Shape, Data: x}}, milcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	return res[out.Name].Data
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

// Variants of the program that compute the same function agree on the CPU: the short convolution
// as taps or as one conv, PostScale on or off, and the model run whole or one layer at a time.
func TestCPUVariants(t *testing.T) {
	cfg := tinyConfig()
	w := llm.NewSynthetic(cfg)
	seq := 6
	emb, err := llm.Embed(w, cfg, []int32{3, 1, 4, 1, 5, 9})
	if err != nil {
		t.Fatal(err)
	}
	base := Options{Seq: seq, Format: "fp16"}
	ref := runCPU(t, cfg, w, base, emb)
	if len(ref) != seq*cfg.VocabSize {
		t.Fatalf("%d logits", len(ref))
	}
	for _, c := range []struct {
		name string
		opts Options
	}{
		{"conv op", Options{Seq: seq, Format: "fp16", ConvMode: "conv"}},
		{"post-scale 64", Options{Seq: seq, Format: "fp16", PostScale: 64}},
	} {
		if e := relErr(runCPU(t, cfg, w, c.opts, emb), ref); e > 1e-5 {
			t.Errorf("%s: relative error %g against the plain program", c.name, e)
		}
	}
	h := emb
	for i := 0; i < cfg.NumLayers; i++ {
		h = runCPU(t, cfg, w, Options{Seq: seq, Format: "fp16", Start: i, Layers: 1, NoHead: true}, h)
	}
	logits := runCPU(t, cfg, w, Options{Seq: seq, Format: "fp16", Start: cfg.NumLayers}, h)
	if e := relErr(logits, ref); e > 1e-6 {
		t.Errorf("layer by layer: relative error %g against the whole program", e)
	}
	if e := relErr(runCPU(t, cfg, w, Options{Seq: seq}, emb), ref); e > 0.2 || e == 0 {
		t.Errorf("lut4: relative error %g against fp16, expected small but not zero", e)
	}
}
