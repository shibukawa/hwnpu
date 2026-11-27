package chat

import (
	"context"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
)

// TestResumeDivergenceSignature verifies that an agent loop rewriting a trailing reminder block
// learns the divergence signature on step 1 and cuts + marks right before it on later steps,
// avoiding full-history re-prefill while keeping logits identical.
func TestResumeDivergenceSignature(t *testing.T) {
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache = 128
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
	e := &Engine{sess: sess, w: w, model: cfg, emb: map[int32][]float32{}}

	rng := rand.New(rand.NewPCG(42, 99))
	toks := func(n int) []int32 {
		s := make([]int32, n)
		for i := range s {
			s[i] = int32(10 + rng.IntN(cfg.VocabSize-10))
		}
		return s
	}
	// Fixed marker representing "<system-reminder>...</system-reminder>" re-appended every turn,
	// including a matching closing-tag tail to verify lead disambiguation.
	openTag := []int32{1, 2, 3, 4, 5}
	closeTag := []int32{6, 2, 3, 4, 5}
	history := toks(20)

	var reused []int
	totalRead := 0
	for step := range 6 {
		prompt := slices.Clone(history)
		prompt = append(prompt, openTag...)
		prompt = append(prompt, int32(10+step), int32(20+step), int32(30+step))
		prompt = append(prompt, closeTag...)
		var st Stats
		got, err := e.prefill(context.Background(), prompt, &st)
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		if d := relErr(got, refLogits(t, cfg, w, cache, prompt)); d > 2e-3 {
			t.Fatalf("step %d: relErr %g", step, d)
		}
		totalRead += len(prompt) - st.Reused
		reused = append(reused, st.Reused)
		history = append(history, toks(12)...)
	}
	t.Logf("totalRead=%d reused=%v", totalRead, reused)
	wantReused := []int{0, 0, 32, 44, 56, 68}
	if !slices.Equal(reused, wantReused) {
		t.Fatalf("reused=%v, want %v", reused, wantReused)
	}
}

// TestResumeValueEviction verifies that cost-aware mark eviction keeps a high-value signature mark
// over low-value early marks when the pool is full.
func TestResumeValueEviction(t *testing.T) {
	var r resumeTracker
	var marks []mark
	marks = r.addMark(marks, mark{m: decode.NewMark(100, nil), kind: markOther}, 2)
	marks = r.addMark(marks, mark{m: decode.NewMark(1000, nil), kind: markSig}, 2)
	// Adding a low-value mark at pos 200 should evict pos 100 (other), NOT pos 1000 (sig).
	marks = r.addMark(marks, mark{m: decode.NewMark(200, nil), kind: markEnd}, 2)
	if len(marks) != 2 {
		t.Fatalf("len=%d, want 2", len(marks))
	}
	var positions []int
	for _, mk := range marks {
		positions = append(positions, mk.m.Pos())
	}
	if !slices.Contains(positions, 1000) || slices.Contains(positions, 100) {
		t.Fatalf("positions=%v, expected 1000 kept and 100 evicted", positions)
	}
	// Adding an even lower-value mark at pos 50 should be rejected rather than evicting 200 or 1000.
	marks = r.addMark(marks, mark{m: decode.NewMark(50, nil), kind: markOther}, 2)
	positions = positions[:0]
	for _, mk := range marks {
		positions = append(positions, mk.m.Pos())
	}
	if slices.Contains(positions, 50) {
		t.Fatalf("positions=%v, low-value mark at 50 should have been rejected", positions)
	}
}
