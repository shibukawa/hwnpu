package chat

import (
	"context"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
)

// A model split at its attention runs on the Neural Engine after the context it was loaded under
// is canceled (hwnpu-chat loads under a signal.NotifyContext and stops it once the model is up),
// with the logits of the CPU interpreter. The attention program kept the load's context, so every
// call returned "context canceled" while its submission ran on: the answers came out empty, the
// next bind failed and closing the session freed chunks under a running prediction.
func TestPagedOnNeuralEngine(t *testing.T) {
	ctx := context.Background()
	if infos, err := (coreml.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(t.TempDir(), "cache")))
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
	load, cancel := context.WithCancel(ctx)
	sess := pagedSessionOn(t, load, dev, cfg, w, cache, chunk, 4)
	cancel()
	t.Cleanup(func() { sess.Close() })
	ref := pagedSession(t, cfg, w, cache, chunk, 4)
	worst := 0.0
	for pos := 0; pos < 3*chunk; pos += 4 {
		var embs [][]float32
		for i := range 4 {
			v, _ := llm.Embed(w, cfg, []int32{int32((pos + i) * 7 % cfg.VocabSize)})
			embs = append(embs, v)
		}
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
	t.Logf("worst relative error %.2g over %d chunks", worst, 3)
}

// The two-level attention runs on the Neural Engine, the mass program included, with the logits
// of the CPU interpreter: a tiny LFM3 split at its attention in chunks of 8, past the length
// where reading buckets of 4 costs fewer calls than the whole cache.
func TestSparseOnNeuralEngine(t *testing.T) {
	ctx := context.Background()
	if infos, err := (coreml.Provider{}).Enumerate(ctx); err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	dev, err := hwnpu.Open(ctx, hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer), hwnpu.WithCacheDir(filepath.Join(t.TempDir(), "cache")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.Close() })
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache, chunk, n = 208, 8, 200
	sp := decode.Sparse{Bucket: 4, Sinks: 1, Window: 4, Exact: 3, Refresh: 1}
	sess := pagedSessionOn(t, ctx, dev, cfg, w, cache, chunk, 4)
	t.Cleanup(func() { sess.Close() })
	ref := pagedSession(t, cfg, w, cache, chunk, 4)
	for _, s := range []*decode.Session{sess, ref} {
		if err := s.SetSparse(sp); err != nil {
			t.Fatal(err)
		}
	}
	worst := 0.0
	for pos := 0; pos < n; pos += 4 {
		var embs [][]float32
		for i := range 4 {
			v, _ := llm.Embed(w, cfg, []int32{int32((pos + i) * 7 % cfg.VocabSize)})
			embs = append(embs, v)
		}
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
	if sess.AttentionCalls() == 0 || sess.AttentionCalls() != ref.AttentionCalls() {
		t.Fatalf("%d calls of the two-level attention on the Neural Engine, %d on the CPU", sess.AttentionCalls(), ref.AttentionCalls())
	}
	if worst > 5e-2 {
		t.Errorf("relative error %g against the CPU interpreter", worst)
	}
	t.Logf("worst relative error %.2g, %d calls of the two-level attention", worst, sess.AttentionCalls())
}
