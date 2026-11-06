package gemma4

import (
	"context"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
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
// own cache in chunks, gives the logits of the program without a cache at every position: the
// full attention layer that reuses layer 2's keys and values attends over layer 2's cache, the
// sliding layers keep their rings and their shared keys in the parts, and the per-layer inputs go
// from the first part to every later one. Steps of one token and of several, over a chunk's
// boundary, with a cache that is not a multiple of the chunk.
func TestPagedMatchesFull(t *testing.T) {
	c := tinyConfig(t)
	w := newWeights(c)
	n := len(tokens)
	if got := c.PagedLayers(); !slices.Equal(got, []int{2, 5}) {
		t.Fatalf("PagedLayers %v, want [2 5]", got)
	}
	if c.PagedKV(5) != 2 || c.PagedKV(2) != 2 {
		t.Fatalf("PagedKV(5) %d, PagedKV(2) %d, want 2 and 2", c.PagedKV(5), c.PagedKV(2))
	}
	feed := map[string][]float32{"x": rows(t, w, c.Embedding(), tokens), "ple": rows(t, w, c.perLayerTable(), tokens)}
	full := build(t, c, w, Options{Options: llm.Options{Seq: n, Format: "fp16"}}).run(t, feed)["y"]
	for _, k := range []struct{ q, cache, chunk int }{{1, 12, 4}, {3, 13, 4}, {4, 12, 5}} {
		t.Run(fmt.Sprintf("q%d cache%d chunk%d", k.q, k.cache, k.chunk), func(t *testing.T) {
			paths := convertPaged(t, c, w, k.cache, k.chunk, k.q)
			man, err := llm.ReadManifest(paths[0])
			if err != nil {
				t.Fatal(err)
			}
			if len(man.Parts) != 3 || man.Attention == nil {
				t.Fatalf("%d parts and attention %v, want 3 parts split at layers 2 and 5", len(man.Parts), man.Attention)
			}
			s, cl, err := runner.OpenSession(context.Background(), nil, paths, c)
			if err != nil {
				t.Fatal(err)
			}
			defer cl.Close()
			worst := 0.0
			for pos := 0; pos < n; {
				v := min(k.q, n-pos)
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
			// as TestCachedMatchesFull: the cache holds fp16 keys and values
			if worst > 5e-3 {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			// one layer's keys and values (layer 2's) are held, for the chunks the sequence reached
			chunks := (n + k.chunk - 1) / k.chunk
			nkv, hd := c.NumKVHeads, c.GlobalHeadDim
			if want := int64(2 * chunks * nkv * k.chunk * hd * 2); s.HeldBytes() != want {
				t.Errorf("the session holds %d bytes of cache, want %d (layer 2's keys and values, %d chunks)", s.HeldBytes(), want, chunks)
			}
			t.Logf("worst relative error %.2g, %d bytes of cache held", worst, s.HeldBytes())
		})
	}
}

// Two programs of 1 and 4 tokens per call continue one sequence split at their attention, and a
// mark and a saved sequence bring it back: the rings of the sliding layers, the shared keys and
// the host's cache stay in step.
func TestPagedShareRewind(t *testing.T) {
	c := tinyConfig(t)
	w := newWeights(c)
	n := len(tokens)
	feed := map[string][]float32{"x": rows(t, w, c.Embedding(), tokens), "ple": rows(t, w, c.perLayerTable(), tokens)}
	full := build(t, c, w, Options{Options: llm.Options{Seq: n, Format: "fp16"}}).run(t, feed)["y"]
	paths := convertPaged(t, c, w, 16, 4, 1, 4)
	s, cl, err := runner.OpenSession(context.Background(), nil, paths, c)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	if !slices.Equal(s.Qs(), []int{1, 4}) {
		t.Fatalf("programs of %v tokens per call", s.Qs())
	}
	step := func(from, to int) []float32 {
		t.Helper()
		var rs [][]float32
		for i := from; i < to; i++ {
			row, err := llm.TokenRow(w, c, tokens[i])
			if err != nil {
				t.Fatal(err)
			}
			rs = append(rs, row)
		}
		logits, err := s.Step(rs)
		if err != nil {
			t.Fatal(err)
		}
		return logits
	}
	check := func(what string, logits []float32, pos int) {
		t.Helper()
		if e := relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]); e > 5e-3 {
			t.Errorf("%s: relative error %g after %d tokens", what, e, pos)
		}
	}
	step(0, 4)
	check("4 then 1", step(4, 5), 5)
	mark := s.Mark()
	check("4", step(5, 9), 9)
	s.Rewind(mark)
	check("rewound, 1", step(5, 6), 6)
	saved := s.Save()
	check("on", step(6, 10), 10)
	s.Reset()
	step(0, 3)
	s.Rewind(saved)
	check("saved, 4", step(6, 10), 10)
	check("saved, 1", step(10, 11), 11)
}

// The two-level attention (decode.Sparse) over Gemma 4 split at its attention stays near the whole
// cache: the layer that reuses layer 2's keys and values reads layer 2's buckets with its own
// queries, and the mass program scores unscaled as the attention does. A mass program added to a
// manifest without one (llm.AddMass) with the model's scale gives the same logits as the one the
// conversion wrote. The weights are the plain synthetic ones: with norm weights around 1
// (newWeights) the tiny model's unscaled scores reach about 30, its attention falls on a key or
// two, and summaries of 4 keys stand for that poorly (relative error 1.3 to 1.6 with or without
// KV sharing, against 0.07 here), as they would for LFM3's.
func TestPagedSparse(t *testing.T) {
	c := tinyConfig(t)
	c.MaxPositions = 256
	w := llm.NewSynthetic(c)
	const n, cache, chunk = 200, 208, 8
	ids := make([]int32, n)
	for i := range ids {
		ids[i] = int32((i*7 + i*i) % c.VocabSize)
	}
	paths := convertPaged(t, c, w, cache, chunk, 1)
	man, err := llm.ReadManifest(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	if man.Mass == nil {
		t.Fatal("the conversion wrote no mass program")
	}
	open := func(sparse bool) *decode.Session {
		s, cl, err := runner.OpenSession(context.Background(), nil, paths, c)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { cl.Close() })
		if sparse {
			if err := s.SetSparse(decode.Sparse{Bucket: 4, Sinks: 1, Window: 4, Exact: 3, Refresh: 1}); err != nil {
				t.Fatal(err)
			}
		}
		return s
	}
	run := func(s *decode.Session) [][]float32 {
		var out [][]float32
		for _, id := range ids {
			row, err := llm.TokenRow(w, c, id)
			if err != nil {
				t.Fatal(err)
			}
			logits, err := s.Step([][]float32{row})
			if err != nil {
				t.Fatal(err)
			}
			out = append(out, logits)
		}
		return out
	}
	sparse := open(true)
	dense, got := run(open(false)), run(sparse)
	if sparse.AttentionCalls() == 0 {
		t.Fatal("the two-level attention never ran")
	}
	worst := 0.0
	for i := range dense {
		worst = math.Max(worst, relErr32(got[i], dense[i]))
	}
	if worst > 0.1 || math.IsNaN(worst) {
		t.Errorf("relative error %g against the whole cache", worst)
	}
	t.Logf("worst relative error %.2g, %d calls of the two-level attention", worst, sparse.AttentionCalls())

	old := *man
	old.Mass = nil
	if err := llm.WriteManifest(paths[0], &old); err != nil {
		t.Fatal(err)
	}
	if err := llm.AddMass(paths[0], "grouped", c.PagedShape().Scale); err != nil {
		t.Fatal(err)
	}
	again := run(open(true))
	for i := range got {
		if !slices.Equal(again[i], got[i]) {
			t.Fatalf("at %d the mass program AddMass wrote gives other logits than the conversion's", i)
		}
	}
}

// A program of 4 tokens per call with the output layer for its last token alone (LastRow), beside
// one of 1, gives the logits of the program without a cache after each step, a short one too.
func TestPagedLastRow(t *testing.T) {
	c := tinyConfig(t)
	w := newWeights(c)
	n := len(tokens)
	feed := map[string][]float32{"x": rows(t, w, c.Embedding(), tokens), "ple": rows(t, w, c.perLayerTable(), tokens)}
	full := build(t, c, w, Options{Options: llm.Options{Seq: n, Format: "fp16"}}).run(t, feed)["y"]
	dir := t.TempDir()
	var paths []string
	for _, q := range []int{1, 4} {
		opts := llm.Options{Seq: q, Cache: 12, Paged: 4, LastRow: q > 1, Format: "fp16"}
		path, err := llm.Convert(Arch, c, w, filepath.Join(dir, fmt.Sprintf("m-q%d.mlpackage", q)), "none", 0, opts, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	s, cl, err := runner.OpenSession(context.Background(), nil, paths, c)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	worst, pos := 0.0, 0
	for _, v := range []int{4, 3, 1, 2} {
		var step [][]float32
		for i := range v {
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
	if worst > 5e-3 {
		t.Errorf("relative error %g against the program without a cache", worst)
	}
	t.Logf("worst relative error %.2g", worst)
}
