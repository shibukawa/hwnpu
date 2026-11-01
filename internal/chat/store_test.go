package chat

import (
	"context"
	"maps"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
)

// The prompt store brings a prompt that begins like one read in an earlier run back from disk, at
// the last mark inside the shared beginning, and the logits stay those of reading the prompt from
// the start. Entries grow with their conversations, split when one diverges, and the least
// recently used go.
func TestEngineStore(t *testing.T) {
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache = 40
	// a fresh engine stands for a restart: new programs, empty buffers, the store read from disk
	newEngine := func(dir string) (*Engine, *store) {
		t.Helper()
		short, long := cpuProgram(t, cfg, w, 1, cache), cpuProgram(t, cfg, w, 4, cache)
		for name := range long.In {
			if strings.HasPrefix(name, "kv_") || strings.HasPrefix(name, "win_") {
				long.In[name] = short.In[name]
			}
		}
		sess, err := decode.New(short, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := sess.Add(long); err != nil {
			t.Fatal(err)
		}
		st, err := openStore(dir)
		if err != nil {
			t.Fatal(err)
		}
		return &Engine{sess: sess, w: w, model: cfg, emb: map[int32][]float32{}, store: st}, st
	}
	defer func(a, b, c, d int) { markEvery, minStored, maxStored, minSaved = a, b, c, d }(markEvery, minStored, maxStored, minSaved)
	markEvery, minStored, maxStored, minSaved = 8, 8, 2, 4

	rng := rand.New(rand.NewPCG(5, 6))
	toks := func(n int) []int32 {
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(rng.IntN(cfg.VocabSize))
		}
		return s
	}
	cat := func(parts ...[]int32) []int32 {
		var s []int32
		for _, p := range parts {
			s = append(s, p...)
		}
		return s
	}
	check := func(e *Engine, name string, prompt []int32, reused, restored int) {
		t.Helper()
		var st Stats
		got, err := e.prefill(context.Background(), prompt, &st)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if st.Reused != reused || st.Restored != restored {
			t.Errorf("%s: reused %d (%d from the store), want %d (%d)", name, st.Reused, st.Restored, reused, restored)
		}
		if d := relErr(got, refLogits(t, cfg, w, cache, prompt)); d > 2e-3 {
			t.Errorf("%s: relative error %g against reading the prompt from the start", name, d)
		}
	}
	marks := func(st *store, tokens []int32) []int {
		t.Helper()
		for _, e := range st.entries {
			if slices.Equal(e.Tokens, tokens) {
				if e.Rows != len(tokens) {
					t.Errorf("the entry of %d tokens holds %d rows", len(tokens), e.Rows)
				}
				return e.Marks
			}
		}
		t.Errorf("no entry of %d tokens among %d", len(tokens), len(st.entries))
		return nil
	}

	dir := t.TempDir()
	a := toks(26)
	e1, st1 := newEngine(dir)
	check(e1, "first run", a, 0, 0)
	// read 4 at a time: marks where the reading crossed a multiple of 8, and at the end
	if got := marks(st1, a); !slices.Equal(got, []int{8, 16, 24, 26}) {
		t.Errorf("marks of the first prompt: %v", got)
	}

	// a restart: a prompt sharing 20 tokens comes back at the mark at 16 and reads the rest;
	// it diverged from the entry, so it gets one of its own with the shared marks
	e2, st2 := newEngine(dir)
	if len(st2.entries) != 1 {
		t.Fatalf("%d entries read back, want 1", len(st2.entries))
	}
	b := cat(a[:20], toks(6))
	check(e2, "after a restart, sharing a beginning", b, 16, 16)
	if got := marks(st2, b); !slices.Equal(got, []int{8, 16, 24, 26}) {
		t.Errorf("marks of the diverged prompt: %v", got)
	}
	if len(st2.entries) != 2 {
		t.Errorf("%d entries after the diverged prompt, want 2", len(st2.entries))
	}
	// the conversation goes on in memory, and its entry grows with it
	c := cat(b, toks(5))
	check(e2, "continued in memory", c, 26, 0)
	if got := marks(st2, c); !slices.Equal(got, []int{8, 16, 24, 26, 31}) {
		t.Errorf("marks of the continued prompt: %v", got)
	}
	// the same prompt again, in memory: nothing read
	check(e2, "the same again", c, 31, 0)

	// another restart: the whole conversation comes back, nothing is read
	e3, st3 := newEngine(dir)
	check(e3, "after a restart, the same prompt", c, 31, 31)
	// and goes on from there
	d := cat(c, toks(3))
	check(e3, "continued after the restart", d, 31, 0)
	if got := marks(st3, d); !slices.Equal(got, []int{8, 16, 24, 26, 31, 34}) {
		t.Errorf("marks after continuing: %v", got)
	}
	// a third conversation sends the least recently used entry (the first prompt's) away
	f := toks(20)
	check(e3, "a new conversation", f, 0, 0)
	if len(st3.entries) != 2 {
		t.Errorf("%d entries after a third conversation, want %d", len(st3.entries), maxStored)
	}
	for _, e := range st3.entries {
		if slices.Equal(e.Tokens, a) {
			t.Errorf("the first prompt's entry stayed")
		}
	}
	// a restart sees what the files hold
	e4, st4 := newEngine(dir)
	if len(st4.entries) != 2 {
		t.Fatalf("%d entries read back, want 2", len(st4.entries))
	}
	// its beginning is still in the entry of the conversation that diverged from it
	check(e4, "the first prompt after its entry went", a, 16, 16)
	// a short prompt is not kept
	check(e4, "short", toks(6), 0, 0)
	if len(st4.entries) != 2 {
		t.Errorf("%d entries after a short prompt, want 2", len(st4.entries))
	}
}

// Resume checks what it is given: the rows must be exactly the mark's position, and the mark the
// session's rings and windows. Rows of another sequence with a mark of this one give wrong logits.
func TestResume(t *testing.T) {
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache = 24
	rng := rand.New(rand.NewPCG(7, 8))
	toks := func(n int) []int32 {
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(rng.IntN(cfg.VocabSize))
		}
		return s
	}
	read := func(sess *decode.Session, seq []int32) []float32 {
		t.Helper()
		var logits []float32
		for i := 0; i < len(seq); i += 4 {
			var embs [][]float32
			for _, tok := range seq[i:min(i+4, len(seq))] {
				v, _ := llm.Embed(w, cfg, []int32{tok})
				embs = append(embs, v)
			}
			if logits, err = sess.Step(embs); err != nil {
				t.Fatal(err)
			}
		}
		return logits
	}
	a, b := toks(20), toks(20)
	sa, err := decode.New(cpuProgram(t, cfg, w, 4, cache), cfg)
	if err != nil {
		t.Fatal(err)
	}
	read(sa, a[:12])
	ma := sa.Mark()
	rowsA, err := sa.ExportRows(0, 12)
	if err != nil {
		t.Fatal(err)
	}
	if len(rowsA) != 12*sa.RowBytes() || sa.RowBytes() == 0 {
		t.Fatalf("%d bytes of rows, %d per position", len(rowsA), sa.RowBytes())
	}
	want := read(sa, a[12:])
	sb, err := decode.New(cpuProgram(t, cfg, w, 4, cache), cfg)
	if err != nil {
		t.Fatal(err)
	}
	read(sb, b[:12])
	rowsB, err := sb.ExportRows(0, 12)
	if err != nil {
		t.Fatal(err)
	}

	s, err := decode.New(cpuProgram(t, cfg, w, 4, cache), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Resume(rowsA, decode.NewMark(ma.Pos(), ma.Bufs())); err != nil {
		t.Fatal(err)
	}
	if s.Pos() != 12 {
		t.Fatalf("at %d after resuming, want 12", s.Pos())
	}
	if d := relErr(read(s, a[12:]), want); d > 1e-3 {
		t.Errorf("resumed: relative error %g", d)
	}
	// the wrong rows
	if err := s.Resume(rowsB, decode.NewMark(ma.Pos(), ma.Bufs())); err != nil {
		t.Fatal(err)
	}
	if d := relErr(read(s, a[12:]), want); d < 1e-2 {
		t.Errorf("rows of another sequence: relative error only %g", d)
	}
	// too few rows, and a mark of other buffers
	if err := s.Resume(rowsA[:11*s.RowBytes()], decode.NewMark(ma.Pos(), ma.Bufs())); err == nil {
		t.Errorf("11 rows for a mark at 12: no error")
	}
	bufs := map[string][]byte{}
	for name, b := range ma.Bufs() {
		bufs[name] = b[:len(b)-2]
	}
	if err := s.Resume(rowsA, decode.NewMark(ma.Pos(), bufs)); err == nil {
		t.Errorf("a mark with short buffers: no error")
	}
	delete(bufs, slices.Sorted(maps.Keys(bufs))[0])
	if err := s.Resume(rowsA, decode.NewMark(ma.Pos(), bufs)); err == nil {
		t.Errorf("a mark lacking a buffer: no error")
	}
}
