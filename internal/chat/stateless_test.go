package chat

import (
	"context"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/llm/qwen3"
)

// newStatelessSession is a Qwen3 session of a program of 1 and one of 4 tokens a call.
func newStatelessSession(t *testing.T, cfg *qwen3.Config, w llm.Weights, cache int) *decode.Session {
	t.Helper()
	short := cpuPackage(t, cfg, w, llm.Options{Seq: 1, Cache: cache, Format: "fp16", Params: map[string]string{"post-scale": "1"}})
	long := cpuPackage(t, cfg, w, llm.Options{Seq: 4, Cache: cache, Format: "fp16", Params: map[string]string{"post-scale": "1"}})
	for name := range long.In {
		if llm.CacheInput(name) {
			long.In[name] = short.In[name]
		}
	}
	s, err := decode.New(short, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Add(long); err != nil {
		t.Fatal(err)
	}
	return s
}

const tinyQwen3 = `{"model_type": "qwen3", "hidden_size": 32, "intermediate_size": 48, "num_hidden_layers": 2,
	"num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8, "vocab_size": 48, "rms_norm_eps": 1e-6,
	"tie_word_embeddings": true, "max_position_embeddings": 96, "rope_theta": 10000}`

// A model whose state is its KV caches alone (Qwen3) goes back to where a prompt parts from the
// last one, inside it, with no mark there: the last message rewritten reads only from the
// parting, and the logits are those of reading the prompt from the start. A model with states
// (LFM3's convolution windows) is not stateless.
func TestEngineReuseStateless(t *testing.T) {
	cfg, err := qwen3.ParseConfig([]byte(tinyQwen3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache = 64
	newSess := func() *decode.Session { return newStatelessSession(t, cfg, w, cache) }
	sess := newSess()
	if !sess.Stateless() {
		t.Fatal("Qwen3's session is not stateless")
	}
	e := &Engine{sess: sess, w: w, model: cfg, emb: map[int32][]float32{}}
	rng := rand.New(rand.NewPCG(5, 6))
	toks := func(n int) []int32 {
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(rng.IntN(48))
		}
		return s
	}
	a, x, y, z := toks(24), toks(6), toks(5), toks(7)
	for _, c := range []struct {
		name   string
		prompt []int32
		reused int
	}{
		{"first", a, 0},
		{"continued", slices.Concat(a, x), 24},
		{"rewritten end", slices.Concat(a, x[:2], y), 26},
		{"inside the first prompt", slices.Concat(a[:15], z), 15},
		{"the same again", slices.Concat(a[:15], z), 22}, // the logits after it are kept
	} {
		var st Stats
		got, err := e.prefill(context.Background(), c.prompt, &st)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if st.Reused != c.reused {
			t.Errorf("%s: reused %d tokens, want %d", c.name, st.Reused, c.reused)
		}
		ref := &Engine{sess: newSess(), w: w, model: cfg, emb: map[int32][]float32{}}
		want, err := ref.prefill(context.Background(), c.prompt, &Stats{})
		if err != nil {
			t.Fatal(err)
		}
		if d := relErr(got, want); d > 2e-3 {
			t.Errorf("%s: relative error %g against reading the prompt from the start", c.name, d)
		}
	}
	for _, mk := range e.cur.marks {
		if mk.kind == markSig {
			t.Errorf("a mark where prompts part, at %d: a stateless session needs none", mk.m.Pos())
		}
	}
	lc, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	ls, err := decode.New(cpuProgram(t, lc, llm.NewSynthetic(lc), 1, 16), lc)
	if err != nil {
		t.Fatal(err)
	}
	if ls.Stateless() {
		t.Error("LFM3's session (convolution windows) is taken as stateless")
	}
}

// After a restart, the store brings a stateless session back to the whole shared beginning, not
// to the last mark inside it (every markEvery tokens).
func TestStoreStateless(t *testing.T) {
	cfg, err := qwen3.ParseConfig([]byte(tinyQwen3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache = 64
	defer func(a, b int) { markEvery, minStored = a, b }(markEvery, minStored)
	markEvery, minStored = 16, 8
	dir := t.TempDir()
	newEngine := func() *Engine {
		st, err := openStore(dir)
		if err != nil {
			t.Fatal(err)
		}
		return &Engine{sess: newStatelessSession(t, cfg, w, cache), w: w, model: cfg, emb: map[int32][]float32{}, store: st}
	}
	rng := rand.New(rand.NewPCG(7, 8))
	toks := func(n int) []int32 {
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(rng.IntN(48))
		}
		return s
	}
	a, z := toks(40), toks(6)
	if _, err := newEngine().prefill(context.Background(), a, &Stats{}); err != nil {
		t.Fatal(err)
	}
	dropping.Wait()
	prompt := slices.Concat(a[:37], z) // marks at 16, 32 and 40
	e := newEngine()
	var st Stats
	got, err := e.prefill(context.Background(), prompt, &st)
	if err != nil {
		t.Fatal(err)
	}
	if st.Restored != 37 {
		t.Errorf("restored %d tokens, want 37 (the shared beginning)", st.Restored)
	}
	ref := &Engine{sess: newStatelessSession(t, cfg, w, cache), w: w, model: cfg, emb: map[int32][]float32{}}
	want, err := ref.prefill(context.Background(), prompt, &Stats{})
	if err != nil {
		t.Fatal(err)
	}
	if d := relErr(got, want); d > 2e-3 {
		t.Errorf("relative error %g against reading the prompt from the start", d)
	}
}
