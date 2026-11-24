package chat

import (
	"context"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/llm/decode"
	"github.com/shibukawa/bonsaiai/internal/llm/lfm3"
	"github.com/shibukawa/bonsaiai/internal/llm/plamo3"
	"github.com/shibukawa/bonsaiai/internal/llm/runner"
)

// tinyPlamo3 is a PLaMo 3 of 4 layers, every second one full, whose other layers see a window of
// 16 tokens.
const tinyPlamo3 = `{"architectures": ["Plamo3ForCausalLM"], "model_type": "plamo3",
	"hidden_size": 64, "intermediate_size": 48, "num_hidden_layers": 4, "num_attention_heads": 4,
	"num_key_value_heads": 2, "head_dim": 16, "vocab_size": 40, "rms_norm_eps": 1e-6,
	"window_size": 16, "sliding_window_pattern": 2, "rope_theta": 1000000, "rope_local_theta": 10000,
	"rope_scaling_factor": 256.0, "initial_context_length": 4096, "max_position_embeddings": 262144,
	"tie_word_embeddings": false, "scale_embedding": false, "image_feature_size": null,
	"image_proj_type": "linear", "linear_type": "normal", "bos_token_id": 1, "eos_token_id": 1,
	"rope_parameters": {"full_attention": {"rope_type": "yarn", "rope_theta": 1000000, "factor": 256.0,
		"original_max_position_embeddings": 4096, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": false},
		"sliding_attention": {"rope_type": "default", "rope_theta": 10000}}}`

// ringPaths converts the tiny PLaMo 3 split at its attention twice, each for 1 and 4 tokens per
// call: with a context of 8 in chunks of 8, where the caches of the window layers hold every
// position, and with one of 64 in chunks of 32, where they are rings of the window. On Core AI
// the four programs together (llm.ConvertAI), the functions of the same Core AI programs.
func ringPaths(t *testing.T, backend string) (cfg *plamo3.Config, w llm.Weights, short, long []string) {
	t.Helper()
	cfg, err := plamo3.ParseConfig([]byte(tinyPlamo3))
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
		ps = append(ps, prog{fmt.Sprintf("c8-q%d", q), llm.Options{Seq: q, Cache: 8, Paged: 8, Format: "fp16", Backend: backend}})
	}
	for _, q := range []int{1, 4} {
		ps = append(ps, prog{fmt.Sprintf("c64-q%d", q), llm.Options{Seq: q, Cache: 64, Paged: 32, Format: "fp16", Backend: backend}})
	}
	var paths []string
	if backend == llm.BackendCoreAI {
		var aps []llm.AIProgram
		for _, p := range ps {
			aps = append(aps, llm.AIProgram{Out: filepath.Join(dir, p.name), Name: p.name, Options: p.opts})
		}
		if paths, err = llm.ConvertAI(plamo3.Arch, cfg, w, filepath.Join(dir, "m"), "none", aps, io.Discard); err != nil {
			t.Fatal(err)
		}
	} else {
		for _, p := range ps {
			path, err := llm.Convert(plamo3.Arch, cfg, w, filepath.Join(dir, p.name+".mlpackage"), "none", 0, p.opts, io.Discard)
			if err != nil {
				t.Fatal(err)
			}
			paths = append(paths, path)
		}
	}
	return cfg, w, paths[:2], paths[2:]
}

// checkRings steps a session of the programs of the short context to 7 tokens, moves it to
// those of the long one (decode.Session.Migrate) and goes on past the window to 43, against a
// session on the long ones from the start; then both go back to a mark taken at 5 tokens, before
// the move (decode.Session.Remark), and on from there. It returns the largest relative error of
// its logits.
func checkRings(t *testing.T, dev *bonsaiai.Device, backend string) float64 {
	t.Helper()
	ctx := context.Background()
	cfg, w, short, long := ringPaths(t, backend)
	sess, c1, err := runner.OpenSession(ctx, dev, short, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c1.Close() })
	ref, c2, err := runner.OpenSession(ctx, dev, long, cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c2.Close() })
	next, c3, err := runner.OpenPrograms(ctx, dev, long)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c3.Close() })
	if sess.Cache() != 8 || ref.Cache() != 64 {
		t.Fatalf("caches of %d and %d, want 8 and 64", sess.Cache(), ref.Cache())
	}
	worst, pos, salt := 0.0, 0, 0
	step := func(n int) {
		t.Helper()
		var embs [][]float32
		for i := range n {
			v, _ := llm.Embed(w, cfg, []int32{int32(((pos+i)*7 + salt) % cfg.VocabSize)})
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
		if e := relErr(got, want); e > worst || math.IsNaN(e) {
			worst = e
		}
	}
	step(4)
	step(1)
	m, rm := sess.Mark(), ref.Mark()
	step(1)
	step(1)
	if err := sess.Migrate(next[0]); err != nil {
		t.Fatal(err)
	}
	if err := sess.Add(next[1]); err != nil {
		t.Fatal(err)
	}
	if sess.Cache() != 64 || sess.Pos() != 7 {
		t.Fatalf("after the move: a cache of %d at %d, want 64 at 7", sess.Cache(), sess.Pos())
	}
	// the mark from before the move, for the session after it: good while the sequence has not
	// gone round the ring
	nm := sess.Remark(m)
	if nm == nil {
		t.Fatal("no mark for the session after the move")
	}
	step(4)
	step(1)
	sess.Rewind(nm)
	ref.Rewind(rm)
	pos, salt = 5, 3
	for _, n := range []int{4, 1, 4, 4, 4, 1, 4, 4, 4, 4, 4} { // past the window of 16 and the chunk of 32
		step(n)
	}
	if sess.Pos() != 43 || sess.Paged() != 32 {
		t.Fatalf("at %d, chunks of %d: want 43 and 32", sess.Pos(), sess.Paged())
	}
	// round the ring: the mark from before the move is no longer the session's
	if sess.Remark(m) != nil {
		t.Error("a mark from before the move after the sequence went round the ring")
	}
	return worst
}

// A sequence begun on the programs of a short context, split at their attention, goes on on
// those of the model's context, where the window layers' caches are rings, with the logits of a
// sequence on the latter from the start, on the CPU interpreter.
func TestMigrateRings(t *testing.T) {
	worst := checkRings(t, nil, "")
	if worst > 2e-2 {
		t.Errorf("relative error %g against the programs of the long context from the start", worst)
	}
	t.Logf("worst relative error %.2g", worst)
}

// The engine runs a model whose other layers see a window on the programs of a short context
// while the conversation fits, and moves it to those of the model's context as it grows (the
// tiers of ringTiers): the logits stay those of reading the sequence from the start on the
// latter, also after going back to a mark from before the move, and a new short conversation
// starts on the short context again.
func TestEngineRingTiers(t *testing.T) {
	ctx := context.Background()
	cfg, w, short, long := ringPaths(t, "")
	tierOf := func(paths []string) tier {
		progs, c, err := runner.OpenPrograms(ctx, nil, paths)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { c.Close() })
		s, err := decode.New(progs[0], cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer s.Close()
		return tier{progs: progs, cache: s.Cache()}
	}
	e := &Engine{w: w, model: cfg, emb: map[int32][]float32{}, Stops: map[int32]bool{}}
	e.tiers = []tier{tierOf(short), tierOf(long)}
	var err error
	if e.sess, err = e.tiers[0].session(cfg); err != nil {
		t.Fatal(err)
	}
	want := func(seq []int32) []float32 {
		t.Helper()
		ref, c, err := runner.OpenSession(ctx, nil, long, cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		var logits []float32
		for _, tok := range seq {
			v, _ := llm.Embed(w, cfg, []int32{tok})
			if logits, err = ref.Step([][]float32{v}); err != nil {
				t.Fatal(err)
			}
		}
		return logits
	}
	check := func(name string, prompt []int32, wantTier int) {
		t.Helper()
		var st Stats
		got, err := e.prefill(ctx, prompt, &st)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if e.ti != wantTier {
			t.Errorf("%s: on tier %d (cache %d), want %d", name, e.ti, e.sess.Cache(), wantTier)
		}
		if d := relErr(got, want(prompt)); d > 2e-3 || math.IsNaN(d) {
			t.Errorf("%s: relative error %g against reading the prompt from the start", name, d)
		}
	}
	a := []int32{3, 14, 15, 9, 2}
	check("a short prompt", a, 0)
	// past the short context (8) and the window (16)
	var gen []int32
	st, err := e.Generate(ctx, Request{Prompt: a, MaxTokens: 16}, func(g []int32) (bool, error) {
		gen = g
		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if st.Generated != 16 || e.ti != 1 {
		t.Fatalf("generated %d tokens, on tier %d", st.Generated, e.ti)
	}
	all := append(slices.Clone(a), gen[:len(gen)-1]...)
	if d := relErr(e.cur.last, want(all)); d > 2e-3 || math.IsNaN(d) {
		t.Errorf("after the move: relative error %g against reading the sequence from the start", d)
	}
	// another answer to the same prompt: back to the mark at its end, taken on the short context
	b := append(slices.Clone(a), 31, 7, 22)
	var stats Stats
	got, err := e.prefill(ctx, b, &stats)
	if err != nil {
		t.Fatal(err)
	}
	if stats.Reused != len(a) || e.ti != 1 {
		t.Errorf("the prompt's %d tokens reused on tier %d: want %d on tier 1", stats.Reused, e.ti, len(a))
	}
	if d := relErr(got, want(b)); d > 2e-3 || math.IsNaN(d) {
		t.Errorf("from the mark before the move: relative error %g against reading the prompt from the start", d)
	}
	// a new short conversation is on the short context again
	check("a new short conversation", []int32{8, 1, 30}, 0)
}

// The programs of a short context are for a model whose other layers see a long window: a
// quarter of it (PLaMo 3's 2048: 512), none for a window under 4 times minRingTier or a model
// without one.
func TestRingTiers(t *testing.T) {
	c, err := plamo3.ParseConfig([]byte(tinyPlamo3))
	if err != nil {
		t.Fatal(err)
	}
	if got := ringTiers(c); got != nil {
		t.Errorf("a window of 16: %v, want none", got)
	}
	c.Window = 2048
	if got := ringTiers(c); !slices.Equal(got, []int{512}) {
		t.Errorf("a window of 2048: %v, want [512]", got)
	}
	c.Window = 1000
	if got := ringTiers(c); got != nil {
		t.Errorf("a window of 1000: %v, want none", got)
	}
	l, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	if got := ringTiers(l); got != nil {
		t.Errorf("LFM3: %v, want none", got)
	}
}
