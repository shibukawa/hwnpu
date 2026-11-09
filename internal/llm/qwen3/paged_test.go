package qwen3

import (
	"context"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// convertPaged converts the model split at its attention with hwnpu-convert's code path, one
// program per number of tokens per call, and returns their manifests.
func convertPaged(t *testing.T, c *Config, w llm.Weights, cache, chunk int, qs ...int) []string {
	t.Helper()
	dir := t.TempDir()
	var paths []string
	for _, q := range qs {
		opts := llm.Options{Seq: q, Cache: cache, Paged: chunk, Format: "fp16"}
		path, err := llm.Convert(Arch, c, w, filepath.Join(dir, fmt.Sprintf("m-q%d.mlpackage", q)), "none", 0, opts, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	return paths
}

// The model split at its attention, stepped through a sequence with the host attending over its
// own cache in chunks, gives the logits of the
// program without a cache at every position. Steps of one token and of several, over a
// chunk's boundary, with a cache that is not a multiple of the chunk.
func TestPagedMatchesFull(t *testing.T) {
	c := tinyConfig(t)
	w := llm.NewSynthetic(c)
	n := len(tokens)
	if got := c.PagedLayers(); !slices.Equal(got, []int{0, 1, 2}) {
		t.Fatalf("PagedLayers %v, want [0 1 2]", got)
	}
	full := build(t, c, w, llm.Options{Seq: n, Format: "fp16"}).run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})["y"]
	for _, k := range []struct{ q, cache, chunk int }{{1, 12, 4}, {3, 13, 4}, {4, 12, 5}} {
		t.Run(fmt.Sprintf("q%d cache%d chunk%d", k.q, k.cache, k.chunk), func(t *testing.T) {
			paths := convertPaged(t, c, w, k.cache, k.chunk, k.q)
			man, err := llm.ReadManifest(paths[0])
			if err != nil {
				t.Fatal(err)
			}
			if len(man.Parts) != 4 || man.Attention == nil {
				t.Fatalf("%d parts and attention %v, want 4 parts split at each of the 3 layers", len(man.Parts), man.Attention)
			}
			s, cl, err := runner.OpenSession(context.Background(), nil, paths, c)
			if err != nil {
				t.Fatal(err)
			}
			defer cl.Close()
			worst := 0.0
			for pos := 0; pos < n; {
				v := min(k.q, n-pos)
				logits := step(t, s, c, w, pos, pos+v)
				pos += v
				worst = math.Max(worst, relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]))
			}
			if worst > 5e-3 {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			t.Logf("worst relative error %.2g", worst)
		})
	}
}

// Two programs of 1 and 4 tokens per call continue one sequence split at their attention, and a
// mark and a saved sequence bring it back.
func TestPagedShareRewind(t *testing.T) {
	c := tinyConfig(t)
	w := llm.NewSynthetic(c)
	n := len(tokens)
	full := build(t, c, w, llm.Options{Seq: n, Format: "fp16"}).run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})["y"]
	paths := convertPaged(t, c, w, 16, 4, 1, 4)
	s, cl, err := runner.OpenSession(context.Background(), nil, paths, c)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	check := func(what string, logits []float32, pos int) {
		t.Helper()
		if e := relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]); e > 5e-3 {
			t.Errorf("%s: relative error %g after %d tokens", what, e, pos)
		}
	}
	step(t, s, c, w, 0, 4)
	check("4 then 1", step(t, s, c, w, 4, 5), 5)
	mark := s.Mark()
	check("4", step(t, s, c, w, 5, 9), 9)
	s.Rewind(mark)
	check("rewound, 1", step(t, s, c, w, 5, 6), 6)
	saved := s.Save()
	check("on", step(t, s, c, w, 6, 10), 10)
	s.Reset()
	step(t, s, c, w, 0, 3)
	s.Rewind(saved)
	check("saved, 4", step(t, s, c, w, 6, 10), 10)
	check("saved, 1", step(t, s, c, w, 10, 11), 11)
}

// A program that reads prompts with the output layer for the last row alone (LastRow), beside
// one of 1 token per call, gives the logits after the last token of each step, a step shorter
// than its 4 tokens too; its last part outputs one row of logits, and it needs a KV cache.
func TestPagedLastRow(t *testing.T) {
	c := tinyConfig(t)
	w := llm.NewSynthetic(c)
	n := len(tokens)
	full := build(t, c, w, llm.Options{Seq: n, Format: "fp16"}).run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})["y"]
	dir := t.TempDir()
	var paths []string
	for _, q := range []int{1, 4} {
		opts := llm.Options{Seq: q, Cache: 16, Paged: 4, LastRow: q > 1, Format: "fp16"}
		path, err := llm.Convert(Arch, c, w, filepath.Join(dir, fmt.Sprintf("m-q%d.mlpackage", q)), "none", 0, opts, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	man, err := llm.ReadManifest(paths[1])
	if err != nil {
		t.Fatal(err)
	}
	if y := man.Parts[len(man.Parts)-1].Output; !slices.Equal(y.Shape, []int{1, 1, c.VocabSize}) {
		t.Fatalf("y %v, want (1, 1, %d)", y.Shape, c.VocabSize)
	}
	s, cl, err := runner.OpenSession(context.Background(), nil, paths, c)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	worst, pos := 0.0, 0
	for _, v := range []int{4, 3, 1, 2, 1} {
		logits := step(t, s, c, w, pos, pos+v)
		pos += v
		worst = math.Max(worst, relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]))
	}
	if worst > 5e-3 {
		t.Errorf("relative error %g against the program without a cache", worst)
	}
	if _, err := llm.Convert(Arch, c, w, filepath.Join(dir, "n.mlpackage"), "none", 0, llm.Options{Seq: 4, LastRow: true, Format: "fp16"}, io.Discard); err == nil {
		t.Error("LastRow without a cache was built")
	}
	t.Logf("worst relative error %.2g", worst)
}
