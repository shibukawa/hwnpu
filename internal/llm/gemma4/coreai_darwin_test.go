package gemma4

import (
	"context"
	"io"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// The programs hwnpu-chat converts of Gemma 4 for Core AI (llm.ConvertAI), on the Neural Engine:
// one of 1 token per call and one of 4 that only writes the caches (KVOnly). The latter's one part
// (layers 0 and 1, and layer 2's keys and values) begins at the layer the former's first part
// begins at, so the two are functions of one Core AI program. Reading the beginning of the
// sequence on the program of 4 and going on on the program of 1 gives the logits of the program
// without a cache, as TestPagedKVOnly does with Core ML's packages on the CPU.
func TestConvertAIKVOnlyOnNeuralEngine(t *testing.T) {
	ctx := context.Background()
	if infos, err := (coreai.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
		t.Skip("no Core AI")
	}
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreai"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(t.TempDir(), "cache")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	c := tinyConfig(t)
	w := newWeights(c)
	n := len(tokens)
	feed := map[string][]float32{"x": rows(t, w, c.Embedding(), tokens), "ple": rows(t, w, c.perLayerTable(), tokens)}
	full := build(t, c, w, Options{Options: llm.Options{Seq: n, Format: "fp16"}}).run(t, feed)["y"]

	gen := llm.Options{Seq: 1, Cache: 16, Paged: 4, Format: "fp16", Params: map[string]string{}}
	kv := gen
	kv.Seq, kv.KVOnly = 4, true
	base := filepath.Join(t.TempDir(), "m-c16")
	paths, err := llm.ConvertAI(Arch, c, w, base+"-q1-q4-kv", "none", []llm.AIProgram{
		{Out: base + "-q1", Name: "q1", Options: gen},
		{Out: base + "-q4-kv", Name: "q4-kv", Options: kv},
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	gm, err := llm.ReadManifest(paths[0])
	if err != nil {
		t.Fatal(err)
	}
	km, err := llm.ReadManifest(paths[1])
	if err != nil {
		t.Fatal(err)
	}
	if !km.KVOnly || len(km.Parts) != 1 || len(gm.Parts) != 3 {
		t.Fatalf("KVOnly %v, %d parts and %d of the program of 1: want 1 and 3", km.KVOnly, len(km.Parts), len(gm.Parts))
	}
	ka, kf := llm.SplitFunction(km.Parts[0].Path)
	ga, gf := llm.SplitFunction(gm.Parts[0].Path)
	if ka != ga || kf != "q4-kv" || gf != "q1" {
		t.Fatalf("the part that writes the caches is %s#%s, the first of the program of 1 %s#%s: want functions of one program", ka, kf, ga, gf)
	}

	s, cl, err := runner.OpenSession(ctx, dev, paths, c)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
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
	pos := 0
	for _, v := range []int{3, 4} { // over a chunk's boundary, a short call first
		if logits := step(pos, pos+v); logits != nil {
			t.Fatalf("a step of %d tokens on the program that writes the caches gave logits", v)
		}
		pos += v
	}
	worst := 0.0
	for ; pos < n; pos++ {
		logits := step(pos, pos+1)
		worst = math.Max(worst, relErr32(logits, full[pos*c.VocabSize:(pos+1)*c.VocabSize]))
	}
	if worst > 3e-2 {
		t.Errorf("relative error %g against the program without a cache", worst)
	}
	t.Logf("worst relative error %.2g on the 4 tokens after reading 7 without logits (Core AI, the Neural Engine)", worst)
}
