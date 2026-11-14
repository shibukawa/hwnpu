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

// hybridPaths converts a tiny LFM3 as a whole program with a cache of 16 (its caches inside) and
// split at its attention in chunks of 32 with a cache of 64, each for 1 and 4 tokens per call: on
// Core AI the four programs together (llm.ConvertAI).
func hybridPaths(t *testing.T, backend string) (cfg *lfm3.Config, w llm.Weights, whole, split []string) {
	t.Helper()
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w = llm.NewSynthetic(cfg)
	dir := t.TempDir()
	type prog struct {
		name string
		opts llm.Options
	}
	var ps []prog
	for _, q := range []int{1, 4} {
		ps = append(ps, prog{fmt.Sprintf("w-q%d", q), llm.Options{Seq: q, Cache: 16, Format: "fp16", Backend: backend, Params: map[string]string{"post-scale": "1"}}})
	}
	for _, q := range []int{1, 4} {
		ps = append(ps, prog{fmt.Sprintf("p-q%d", q), llm.Options{Seq: q, Cache: 64, Paged: 32, Format: "fp16", Backend: backend, Params: map[string]string{"post-scale": "1"}}})
	}
	var paths []string
	if backend == llm.BackendCoreAI {
		var aps []llm.AIProgram
		for _, p := range ps {
			aps = append(aps, llm.AIProgram{Out: filepath.Join(dir, p.name), Name: p.name, Options: p.opts})
		}
		if paths, err = llm.ConvertAI(lfm3.Arch, cfg, w, filepath.Join(dir, "m"), "none", aps, io.Discard); err != nil {
			t.Fatal(err)
		}
	} else {
		for _, p := range ps {
			path, err := llm.Convert(lfm3.Arch, cfg, w, filepath.Join(dir, p.name+".mlpackage"), "none", 0, p.opts, io.Discard)
			if err != nil {
				t.Fatal(err)
			}
			paths = append(paths, path)
		}
	}
	return cfg, w, paths[:2], paths[2:]
}

// checkMigrate steps a session of the whole programs to 14 tokens, moves it to the programs split
// at their attention (decode.Session.Migrate) and goes on to 50, against a session split from the
// start, and returns the largest relative error of its logits.
func checkMigrate(t *testing.T, dev *hwnpu.Device, backend string) float64 {
	t.Helper()
	ctx := context.Background()
	cfg, w, whole, split := hybridPaths(t, backend)
	sess, c1, err := runner.OpenSession(ctx, dev, whole, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c1.Close() })
	ref, c2, err := runner.OpenSession(ctx, dev, split, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c2.Close() })
	next, c3, err := runner.OpenPrograms(ctx, dev, split)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c3.Close() })
	worst, pos := 0.0, 0
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
	for _, n := range []int{4, 4, 1, 4, 1} {
		step(n)
	}
	if err := sess.Migrate(next[0]); err != nil {
		t.Fatal(err)
	}
	if err := sess.Add(next[1]); err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{4, 1, 4, 4, 4, 1, 4, 4, 4, 4, 2} {
		step(n)
	}
	if sess.Pos() != 50 || sess.Paged() != 32 {
		t.Fatalf("at %d, chunks of %d: want 50 and 32", sess.Pos(), sess.Paged())
	}
	return worst
}

// A sequence begun on a whole program with a short cache goes on on the programs split at their
// attention with the logits of a sequence split from the start, on the CPU interpreter.
func TestMigrateToPaged(t *testing.T) {
	if worst := checkMigrate(t, nil, ""); worst > 2e-2 {
		t.Errorf("relative error %g against the programs split from the start", worst)
	}
}
