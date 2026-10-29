package lfm3

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/shibukawa/hwnpu/internal/llm"
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

// synthWeights returns deterministic pseudo random tensors for the names and shapes of a config.
type synthWeights map[string][]int

func newSynthWeights(c *Config) synthWeights {
	d, hd := c.HiddenSize, c.HeadDim
	w := synthWeights{
		"model.embed_tokens.weight":   {c.VocabSize, d},
		"lm_head.weight":              {c.VocabSize, d},
		"model.embedding_norm.weight": {d},
	}
	for i := 0; i < c.NumLayers; i++ {
		p := fmt.Sprintf("model.layers.%d.", i)
		for _, n := range []string{"operator_norm", "post_operator_norm", "ffn_norm", "post_ffn_norm"} {
			w[p+n+".weight"] = []int{d}
		}
		w[p+"feed_forward.w1.weight"] = []int{c.IntermediateSize, d}
		w[p+"feed_forward.w3.weight"] = []int{c.IntermediateSize, d}
		w[p+"feed_forward.w2.weight"] = []int{d, c.IntermediateSize}
		if c.IsAttention(i) {
			a := p + "self_attn."
			w[a+"q_proj.weight"] = []int{c.NumHeads * hd, d}
			w[a+"k_proj.weight"] = []int{c.NumKVHeads * hd, d}
			w[a+"v_proj.weight"] = []int{c.NumKVHeads * hd, d}
			w[a+"o_proj.weight"] = []int{d, c.NumHeads * hd}
			w[a+"gate_proj.weight"] = []int{c.NumHeads, d}
			w[a+"q_norm.weight"] = []int{hd}
			w[a+"k_norm.weight"] = []int{hd}
		} else {
			w[p+"conv.in_proj.weight"] = []int{3 * d, d}
			w[p+"conv.out_proj.weight"] = []int{d, d}
			w[p+"conv.conv.weight"] = []int{d, 1, c.ConvKernel}
			w[p+"conv.conv.bias"] = []int{d}
		}
	}
	return w
}

func (w synthWeights) Get(name string) ([]float32, []int, error) {
	shape, ok := w[name]
	if !ok {
		return nil, nil, fmt.Errorf("synth: no tensor %s", name)
	}
	n := 1
	for _, s := range shape {
		n *= s
	}
	h := fnv.New64a()
	h.Write([]byte(name))
	seed := h.Sum64()
	v := make([]float32, n)
	for i := range v {
		seed = seed*6364136223846793005 + 1442695040888963407
		v[i] = float32(int64(seed>>40)-(1<<23)) / (1 << 23) * 0.2
	}
	return v, shape, nil
}

func (w synthWeights) Rows(name string, begin, end int) ([]float32, error) {
	v, shape, err := w.Get(name)
	if err != nil {
		return nil, err
	}
	return v[begin*shape[1] : end*shape[1]], nil
}

func (w synthWeights) Source() string { return "synthetic" }
func (w synthWeights) Close() error   { return nil }

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
	w := newSynthWeights(cfg)
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
	w := newSynthWeights(tinyConfig())
	for _, c := range []struct {
		params map[string]string
		direct Options
	}{
		{nil, Options{Seq: 6, PostScale: 64, ConvMode: "taps"}},
		{map[string]string{"post-scale": "1", "conv": "conv"}, Options{Seq: 6, ConvMode: "conv"}},
	} {
		_, _, got := build(t, func(b *mlpackage.Builder) (mlpackage.Feature, mlpackage.Feature, error) {
			return model.Build(b, w, llm.Options{Seq: 6, Params: c.params})
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
