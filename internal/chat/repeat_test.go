package chat

import (
	"context"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
)

// The tokens of the repeats tests: 0 ends a line, 1-9 are punctuation and numbers, from 10 on
// words.
func testKinds() []tokenKind {
	kinds := make([]tokenKind, 1000)
	kinds[0] = kindLineEnd
	for i := 10; i < len(kinds); i++ {
		kinds[i] = kindWord
	}
	return kinds
}

// lines makes lines of tokens, each ended by 0.
func lines(ls ...[]int32) []int32 {
	var out []int32
	for _, l := range ls {
		out = append(append(out, l...), 0)
	}
	return out
}

// firstLoop is the index of the token at which repeats tells a loop, or -1.
func firstLoop(toks []int32) int {
	r := newRepeats(testKinds())
	for i, t := range toks {
		if r.add(t) {
			return i
		}
	}
	return -1
}

func TestRepeats(t *testing.T) {
	// a model listing what it has listed: "- Gifu? Already." for 20 names in turn, over and over
	var already [][]int32
	for round := range 6 {
		for name := range 20 {
			// the order changes a little each time round
			already = append(already, []int32{1, int32(100 + (name*7+round%2)%20), 2, 11})
		}
	}
	// an answer that writes out the draft of its thinking, both 40 lines
	var draft [][]int32
	for i := range 40 {
		draft = append(draft, []int32{int32(100 + i), int32(300 + i), 12})
	}
	// tests of a class: each sets the account up the same way and checks something else
	var tests [][]int32
	for i := range 20 {
		tests = append(tests,
			[]int32{20, int32(400 + i), 3},     // def test_...():
			[]int32{21, 22, 4, 23, 5},          // acct = Account(100)
			[]int32{int32(500 + i), 24, 6},     // acct.withdraw(...)
			[]int32{25, int32(600 + i), 7, 26}, // self.assertEqual(...)
		)
	}
	// rows that differ in their numbers alone: a table, not a loop
	var rows [][]int32
	for i := range 120 {
		rows = append(rows, []int32{1, int32(i % 10), int32(i / 10 % 10), 1, 30, 31, 1})
	}
	// a loop of 100 tokens without a line break
	var long []int32
	for i := range 800 {
		long = append(long, int32(100+i%100))
	}
	for _, c := range []struct {
		name    string
		toks    []int32
		atLeast int // the loop is told no sooner (-1: never)
	}{
		{"names over and over", lines(already...), 2 * 20 * 5},
		{"the draft written out", lines(append(slices.Clone(draft), draft...)...), -1},
		{"tests of a class", lines(tests...), -1},
		{"numbered rows", lines(rows...), -1},
		{"a long loop in a line", long, 2 * 100},
	} {
		at := firstLoop(c.toks)
		switch {
		case c.atLeast < 0 && at >= 0:
			t.Errorf("%s: a loop at %d of %d tokens", c.name, at, len(c.toks))
		case c.atLeast >= 0 && at < 0:
			t.Errorf("%s: no loop in %d tokens", c.name, len(c.toks))
		case c.atLeast >= 0 && at < c.atLeast:
			t.Errorf("%s: a loop at %d, before %d", c.name, at, c.atLeast)
		}
	}
	// what the thinking repeated twice the answer may repeat once more
	r := newRepeats(testKinds())
	for _, tok := range lines(append(slices.Clone(draft), draft...)...) {
		r.add(tok)
	}
	r.reset()
	for _, tok := range lines(draft...) {
		if r.add(tok) {
			t.Fatal("a loop counted across reset")
		}
	}
}

// Thinking that goes round in circles is ended as at its budget, and the answer begins: a loop
// stops the generation only in the answer.
func TestThinkLoop(t *testing.T) {
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache = 400
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
	e := &Engine{sess: sess, w: w, model: cfg, emb: map[int32][]float32{}, Stops: map[int32]bool{}}
	// greedy decoding of the synthetic model falls into a loop, inside the thinking and out
	think := ThinkLimit{Budget: 300, Open: 46, Close: 47, End: []int32{45, 47}, Inside: true} // tokens it does not generate
	var gen []int32
	st, err := e.Generate(context.Background(), Request{Prompt: []int32{3, 1, 4, 1, 5, 9, 2, 6}, Think: think}, func(g []int32) (bool, error) {
		gen = g
		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	end := slices.Index(gen, think.Close)
	if end < 0 || !st.ThinkLooped || st.ThinkCut || !slices.Equal(gen[end-1:end+1], think.End) || end > think.Budget {
		t.Fatalf("the thinking was not cut going round in circles: looped %v, cut %v, closed at %d", st.ThinkLooped, st.ThinkCut, end)
	}
	if !st.Looped || st.Finish != "stop" || len(gen) >= cache-8 {
		t.Errorf("the answer did not stop repeating itself: looped %v, %s after %d tokens", st.Looped, st.Finish, len(gen))
	}
}
