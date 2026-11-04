package lfm3

import (
	"io"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// The checkpoint split at its attention (HWNPU_LFM3_DIR, with HWNPU_LFM3_PAGED_CPU=1: slow, on
// the CPU interpreter) gives the logits of the same checkpoint converted whole with its cache in
// the program, step by step over a prompt that crosses several chunks: the real layout (6 full
// attention layers, sliding rings and convolutions in the parts), not only the tiny model's.
func TestPagedCheckpoint(t *testing.T) {
	dir := os.Getenv("HWNPU_LFM3_DIR")
	if dir == "" || os.Getenv("HWNPU_LFM3_PAGED_CPU") != "1" {
		t.Skip("HWNPU_LFM3_DIR and HWNPU_LFM3_PAGED_CPU=1 are not set")
	}
	a, m, err := llm.LoadConfig(filepath.Join(dir, "config.json"), "")
	if err != nil {
		t.Fatal(err)
	}
	cfg := m.(*Config)
	w, err := llm.OpenWeights(filepath.Join(dir, "model.safetensors"), a)
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	tok, err := tokenizer.Load(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatal(err)
	}
	ids := tok.Encode("日本の首都は東京です。東京には多くの美術館や公園があり、春には桜が咲きます。" +
		"The Neural Engine runs small language models with little power, so that a laptop answers questions without a server.")
	const q, cache, chunk = 8, 64, 16
	if len(ids) > cache {
		ids = ids[:cache]
	}
	t.Logf("%d tokens", len(ids))
	opts := llm.Options{Seq: q, Cache: cache, Format: "fp16", Params: map[string]string{"post-scale": "64", "attn": "grouped"}}
	start := time.Now()
	whole, err := llm.Convert(a, cfg, w, filepath.Join(t.TempDir(), "whole.mlpackage"), "none", 0, opts, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	opts.Paged = chunk
	split, err := llm.Convert(a, cfg, w, filepath.Join(t.TempDir(), "split.mlpackage"), "none", 0, opts, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	man, err := llm.ReadManifest(split)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("converted in %.0fs: whole, and split in %d parts", time.Since(start).Seconds(), len(man.Parts))
	ws, err := decode.New(cpuProgram(t, whole), cfg)
	if err != nil {
		t.Fatal(err)
	}
	allocated := 0
	ps, err := decode.New(cpuPaged(t, man, &allocated), cfg)
	if err != nil {
		t.Fatal(err)
	}
	worst, same, steps := 0.0, 0, 0
	for pos := 0; pos < len(ids); pos += q {
		var rows [][]float32
		for _, id := range ids[pos:min(pos+q, len(ids))] {
			r, err := llm.TokenRow(w, cfg, id)
			if err != nil {
				t.Fatal(err)
			}
			rows = append(rows, r)
		}
		t0 := time.Now()
		want, err := ws.Step(rows)
		if err != nil {
			t.Fatal(err)
		}
		t1 := time.Now()
		got, err := ps.Step(rows)
		if err != nil {
			t.Fatal(err)
		}
		e := relErr(got, want)
		worst = math.Max(worst, e)
		steps++
		if top1(got) == top1(want) {
			same++
		}
		t.Logf("tokens %d-%d: relative error %.2g, next token %d / %d (whole %.1fs, split %.1fs)", pos, pos+len(rows)-1, e, top1(want), top1(got), t1.Sub(t0).Seconds(), time.Since(t1).Seconds())
	}
	if worst > 1e-2 || same != steps {
		t.Errorf("worst relative error %.2g, the same next token at %d of %d steps", worst, same, steps)
	}
	t.Logf("the split model held %d KiB of cache for %d tokens", ps.HeldBytes()/1024, len(ids))
}

// top1 is the index of the largest value.
func top1(v []float32) int {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return best
}
