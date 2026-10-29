package lfm3

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// weightFileBytes builds the model with opts and returns the size of its weight file.
func weightFileBytes(t *testing.T, cfg *Config, w llm.Weights, opts llm.Options) int64 {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := cfg.Build(mlpackage.NewBuilder("CoreML8", blobs), w, opts); err != nil {
		t.Fatal(err)
	}
	if err := blobs.Close(); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info.Size()
}

// The size estimate that hwnpu-convert splits by matches the weight files the parts get, and a
// split covers every layer once with each part under its budget.
func TestSplitSizes(t *testing.T) {
	cfg := tinyConfig()
	cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize, cfg.HeadDim = 256, 512, 512, 16
	cfg.NumLayers, cfg.LayerTypes = 6, []string{"conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention"}
	w := llm.NewSynthetic(cfg)
	for _, opts := range []llm.Options{{Seq: 4, Format: "lut4"}, {Seq: 4, Format: "fp16"}, {Seq: 4, Format: "lut4", KeepFP16: []string{"lm_head"}}} {
		pre, layers, post := llm.EstimateBytes(cfg, opts)
		total := pre + post
		for _, n := range layers {
			total += n
		}
		got := weightFileBytes(t, cfg, w, opts)
		t.Logf("%s %v: estimated %d, weight file %d (%.3f%%)", opts.Format, opts.KeepFP16, total, got, 100*float64(got-total)/float64(got))
		if d := float64(got-total) / float64(got); d < 0 || d > 0.01 {
			t.Errorf("%s %v: estimated %d bytes, weight file %d", opts.Format, opts.KeepFP16, total, got)
		}

		budget := total * 2 / 5
		parts, err := llm.PlanSplit(cfg, opts, budget, 0)
		if err != nil {
			t.Fatal(err)
		}
		if len(parts) < 3 {
			t.Errorf("%s: %d parts for a budget of 2/5 of the model", opts.Format, len(parts))
		}
		fewer, _ := llm.PlanSplit(cfg, opts, budget, len(parts)-1)
		over := false
		for _, p := range fewer {
			over = over || p.Bytes > budget
		}
		if !over {
			t.Errorf("%s: %d parts, but %d fit: %+v", opts.Format, len(parts), len(fewer), fewer)
		}
		next := 0
		for _, p := range parts {
			if p.Start != next || p.Layers < 1 || p.Bytes > budget {
				t.Fatalf("%s: parts %+v", opts.Format, parts)
			}
			next += p.Layers
			o := opts
			o.Start, o.Layers = p.Start, p.Layers
			if size := weightFileBytes(t, cfg, w, o); float64(size) > float64(budget)/llm.SplitMargin {
				t.Errorf("%s: part %+v has a weight file of %d bytes, over the limit %d", opts.Format, p, size, int64(float64(budget)/llm.SplitMargin))
			}
		}
		if next != cfg.NumLayers {
			t.Errorf("%s: parts %+v do not cover %d layers", opts.Format, parts, cfg.NumLayers)
		}
		if forced, _ := llm.PlanSplit(cfg, opts, budget, 5); len(forced) != 5 {
			t.Errorf("%s: -parts 5 gave %d parts", opts.Format, len(forced))
		}
	}
	if _, err := llm.PlanSplit(cfg, llm.Options{Format: "fp16"}, 1000, 0); err == nil {
		t.Error("a budget smaller than one layer did not fail")
	}
}
