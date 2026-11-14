package chat

import (
	"context"
	"io"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// The programs Prepare converts for Core AI (llm.ConvertAI: 1 and 4 tokens per call of a model
// split at its attention, the parts of the same layers the functions of one Core AI program, the
// one of 4 with the last row's logits alone) run on the Neural Engine with the logits of the CPU
// interpreter's programs from Core ML packages.
func TestConvertAIOnNeuralEngine(t *testing.T) {
	ctx := context.Background()
	if infos, err := (coreai.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
		t.Skip("no Core AI")
	}
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreai"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(t.TempDir(), "cache")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache, chunk = 40, 8
	opts := llm.Options{Seq: 1, Cache: cache, Paged: chunk, Format: "fp16", Params: map[string]string{"post-scale": "1"}}
	pre := opts
	pre.Seq, pre.LastRow = 4, true
	base := filepath.Join(t.TempDir(), "m-c40")
	paths, err := llm.ConvertAI(lfm3.Arch, cfg, w, base+"-q1-q4-last", "none", []llm.AIProgram{
		{Out: base + "-q1", Name: "q1", Options: opts},
		{Out: base + "-q4-last", Name: "q4-last", Options: pre},
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	sess, closer, err := runner.OpenSession(ctx, dev, paths, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { closer.Close() })
	defer func(b bool) { lastRow = b }(lastRow)
	lastRow = true
	ref := pagedSession(t, cfg, w, cache, chunk, 1, 4)
	worst := 0.0
	pos := 0
	for _, n := range []int{4, 4, 1, 1, 4, 1, 4, 4, 1} {
		var embs [][]float32
		for i := range n {
			v, _ := llm.Embed(w, cfg, []int32{int32((pos + i) * 7 % cfg.VocabSize)})
			embs = append(embs, v)
		}
		pos += n
		got, err := sess.Step(embs)
		if err != nil {
			t.Fatalf("at %d: %v", pos, err)
		}
		want, err := ref.Step(embs)
		if err != nil {
			t.Fatal(err)
		}
		worst = math.Max(worst, relErr(got, want))
	}
	if worst > 3e-2 {
		t.Errorf("relative error %g against the CPU interpreter", worst)
	}
	t.Logf("worst relative error %.2g over %d tokens", worst, pos)
}

// The attention over shorter chunks (llm.AddSmaller) as Core AI programs on the Neural Engine
// gives the logits of the programs without them there.
func TestShorterChunksOnNeuralEngine(t *testing.T) {
	ctx := context.Background()
	if infos, err := (coreai.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
		t.Skip("no Core AI")
	}
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreai"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(t.TempDir(), "cache")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	cfg, w, sess, ref := shorterSessions(t, dev, llm.BackendCoreAI)
	worst := checkShorter(t, cfg, w, sess, ref)
	if worst > 3e-2 {
		t.Errorf("relative error %g against the programs without them", worst)
	}
	t.Logf("worst relative error %.2g (Core AI, the Neural Engine)", worst)
}
