package gemma4

import (
	"context"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/chain"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
)

// writeIRParts is writeParts for Intel's NPU: the parts are OpenVINO IR.
func writeIRParts(t *testing.T, c *Config, w llm.Weights, opts llm.Options, n int) string {
	t.Helper()
	opts.Backend = llm.BackendIntel
	plan, err := llm.PlanSplit(c, opts, math.MaxInt64, n)
	if err != nil || len(plan) != n {
		t.Fatalf("plan of %d parts: %v, %v", n, plan, err)
	}
	dir := t.TempDir()
	man := &llm.Manifest{Seq: opts.Seq, Cache: opts.Cache, Backend: llm.BackendIntel}
	for i, p := range plan {
		o := opts
		if n > 1 {
			o.Start, o.Layers = p.Start, p.Layers
		}
		rec, err := llm.WritePart(filepath.Join(dir, fmt.Sprintf("m.q%d.part%d.xml", opts.Seq, i+1)), c, w, o, nil)
		if err != nil {
			t.Fatal(err)
		}
		man.Parts = append(man.Parts, rec)
	}
	path := filepath.Join(dir, fmt.Sprintf("m.q%d.parts.json", opts.Seq))
	if err := llm.WriteManifest(path, man); err != nil {
		t.Fatal(err)
	}
	return path
}

// TestCachedMatchesFull on OpenVINO's CPU: the program with a KV cache lowered to OpenVINO IR and
// run through internal/llm/chain (the per-layer inputs, the shared keys and values passed between
// parts by name, the rings of the sliding layers) gives the logits of the MIL program without a
// cache, also in steps padded past the tokens (whose rows once made NaN: decode.Session.run).
// Needs HWNPU_OV_PYTHON.
func TestIntelIRCached(t *testing.T) {
	if ovcpu.Python() == "" {
		t.Skip("HWNPU_OV_PYTHON is not set")
	}
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "openvino", "")
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	c := tinyConfig(t)
	w := newWeights(c)
	n := len(tokens)
	feed := map[string][]float32{"x": rows(t, w, c.Embedding(), tokens), "ple": rows(t, w, c.perLayerTable(), tokens)}
	full := build(t, c, w, Options{Options: llm.Options{Seq: n, Format: "fp16"}}).run(t, feed)["y"]
	for _, k := range []struct{ q, gen, cache, parts int }{{1, 0, 11, 1}, {3, 0, 11, 1}, {3, 0, 12, 2}, {4, 1, 16, 3}} {
		t.Run(fmt.Sprintf("q%d gen%d cache%d parts%d", k.q, k.gen, k.cache, k.parts), func(t *testing.T) {
			opts := llm.Options{Seq: k.q, Cache: k.cache, Format: "fp16"}
			ch, err := chain.Open(ctx, dev, writeIRParts(t, c, w, opts, k.parts))
			if err != nil {
				t.Fatal(err)
			}
			defer ch.Close()
			s, err := decode.New(ch.Program(ctx), c)
			if err != nil {
				t.Fatal(err)
			}
			if k.gen > 0 {
				opts.Seq = k.gen
				g, err := ch.Share(ctx, dev, writeIRParts(t, c, w, opts, k.parts))
				if err != nil {
					t.Fatal(err)
				}
				defer g.Close()
				if err := s.Add(g.Program(ctx)); err != nil {
					t.Fatal(err)
				}
			}
			worst := 0.0
			for pos := 0; pos < n; {
				v := min(k.q, n-pos)
				if k.gen > 0 && pos >= k.q {
					v = 1
				}
				var step [][]float32
				for i := 0; i < v; i++ {
					row, err := llm.TokenRow(w, c, tokens[pos+i])
					if err != nil {
						t.Fatal(err)
					}
					step = append(step, row)
				}
				logits, err := s.Step(step)
				if err != nil {
					t.Fatal(err)
				}
				pos += v
				worst = math.Max(worst, relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]))
			}
			if !(worst <= 5e-3) {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			t.Logf("worst relative error %.2g", worst)
		})
	}
}

// TestPagedMatchesFull on OpenVINO's CPU: the parts and the attention program of the model split
// at its attention, lowered to OpenVINO IR and run through internal/llm/chain, with one program
// of 4 tokens per call and one of 1 sharing the host's cache, give the logits of the MIL program
// without a cache. Needs HWNPU_OV_PYTHON.
func TestIntelIRPaged(t *testing.T) {
	if ovcpu.Python() == "" {
		t.Skip("HWNPU_OV_PYTHON is not set")
	}
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "openvino", "")
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	c := tinyConfig(t)
	w := newWeights(c)
	n := len(tokens)
	feed := map[string][]float32{"x": rows(t, w, c.Embedding(), tokens), "ple": rows(t, w, c.perLayerTable(), tokens)}
	full := build(t, c, w, Options{Options: llm.Options{Seq: n, Format: "fp16"}}).run(t, feed)["y"]
	dir := t.TempDir()
	var paths []string
	for _, q := range []int{4, 1} {
		opts := llm.Options{Seq: q, Cache: 13, Paged: 4, Format: "fp16", Backend: llm.BackendIntel}
		path, err := llm.Convert(Arch, c, w, filepath.Join(dir, fmt.Sprintf("m-q%d.xml", q)), "none", 0, opts, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	ch, err := chain.Open(ctx, dev, paths[0])
	if err != nil {
		t.Fatal(err)
	}
	defer ch.Close()
	if ch.Paged() != 4 {
		t.Fatalf("chunks of %d, want 4", ch.Paged())
	}
	s, err := decode.New(ch.Program(ctx), c)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	g, err := ch.Share(ctx, dev, paths[1])
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if err := s.Add(g.Program(ctx)); err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for pos := 0; pos < n; {
		v := min(4, n-pos)
		if pos >= 4 {
			v = 1
		}
		var step [][]float32
		for i := 0; i < v; i++ {
			row, err := llm.TokenRow(w, c, tokens[pos+i])
			if err != nil {
				t.Fatal(err)
			}
			step = append(step, row)
		}
		logits, err := s.Step(step)
		if err != nil {
			t.Fatal(err)
		}
		pos += v
		worst = math.Max(worst, relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]))
	}
	if !(worst <= 5e-3) {
		t.Errorf("relative error %g against the program without a cache", worst)
	}
	t.Logf("worst relative error %.2g", worst)
}
