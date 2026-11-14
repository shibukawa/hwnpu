package chat

import (
	"context"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// checkShorter steps sess and ref, the same model's programs with and without the attention over
// shorter chunks, through a sequence that passes each chunk, a rewind and a reset, and returns
// the largest relative error of sess's logits.
func checkShorter(t *testing.T, cfg *lfm3.Config, w llm.Weights, sess, ref *decode.Session) float64 {
	t.Helper()
	worst := 0.0
	pos := 0
	step := func(n int) {
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
	for range 3 {
		step(1)
	}
	m, rm := sess.Mark(), ref.Mark()
	at := pos
	for _, n := range []int{4, 4, 1, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4} { // past 8, 16 and 32
		step(n)
	}
	sess.Rewind(m)
	ref.Rewind(rm)
	pos = at
	for _, n := range []int{4, 1, 4} {
		step(n)
	}
	sess.Reset()
	ref.Reset()
	pos = 0
	for _, n := range []int{4, 4, 4, 4, 1} {
		step(n)
	}
	return worst
}

// shorterSessions converts a tiny LFM3 split at its attention in chunks of 32, programs of 1 and
// 4 tokens per call, and opens it on dev (nil: the CPU interpreter) twice: with the attention
// programs over chunks of 8 and 16 (llm.AddSmaller) and without.
func shorterSessions(t *testing.T, dev *hwnpu.Device, backend string) (*lfm3.Config, llm.Weights, *decode.Session, *decode.Session) {
	t.Helper()
	ctx := context.Background()
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache, chunk = 64, 32
	var with, without []string
	dir := t.TempDir()
	for _, q := range []int{1, 4} {
		opts := llm.Options{Seq: q, Cache: cache, Paged: chunk, Format: "fp16", Backend: backend, Params: map[string]string{"post-scale": "1"}}
		var paths []string
		for _, kind := range []string{"with", "without"} {
			base := filepath.Join(dir, fmt.Sprintf("m-%s-q%d", kind, q))
			var path string
			if backend == llm.BackendCoreAI {
				ps, err := llm.ConvertAI(lfm3.Arch, cfg, w, base, "none", []llm.AIProgram{{Out: base, Name: fmt.Sprintf("q%d", q), Options: opts}}, io.Discard)
				if err != nil {
					t.Fatal(err)
				}
				path = ps[0]
			} else if path, err = llm.Convert(lfm3.Arch, cfg, w, base+".mlpackage", "none", 0, opts, io.Discard); err != nil {
				t.Fatal(err)
			}
			paths = append(paths, path)
		}
		if err := llm.AddSmaller(paths[0], "grouped", 0, 8); err != nil {
			t.Fatal(err)
		}
		man, err := llm.ReadManifest(paths[0])
		if err != nil {
			t.Fatal(err)
		}
		if len(man.Smaller) != 2 || man.Smaller[0].Chunk() != 8 || man.Smaller[1].Chunk() != 16 {
			t.Fatalf("%d shorter chunks: want 8 and 16", len(man.Smaller))
		}
		with, without = append(with, paths[0]), append(without, paths[1])
	}
	open := func(paths []string) *decode.Session {
		s, closer, err := runner.OpenSession(ctx, dev, paths, cfg)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { closer.Close() })
		return s
	}
	return cfg, w, open(with), open(without)
}

// A model's programs with the attention over shorter chunks give the logits of those without, on
// the CPU interpreter, the cache moving from chunks of 8 to 16 and 32 and back.
func TestShorterChunks(t *testing.T) {
	cfg, w, sess, ref := shorterSessions(t, nil, "")
	if worst := checkShorter(t, cfg, w, sess, ref); worst > 1e-2 {
		t.Errorf("relative error %g against the programs without them", worst)
	}
}
