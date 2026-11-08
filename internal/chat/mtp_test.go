package chat

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/qwen35"
	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// tinyQwen35MTP is tinyQwen35 with a multi-token prediction layer.
var tinyQwen35MTP = strings.Replace(tinyQwen35, `"model_type": "qwen3_5_text",`, `"model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1,`, 1)

// With drafts from the multi-token prediction layer, the engine generates what it generates one
// token at a time, and each first draft is what the layer guesses reading the sequence from the
// start: its entries follow the sequence through the steps, the drafts it takes back and the
// next request's prompt, which goes back to a mark.
func TestMTPDrafts(t *testing.T) {
	cfg, err := qwen35.ParseConfig([]byte(tinyQwen35MTP))
	if err != nil {
		t.Fatal(err)
	}
	mm, err := qwen35.ParseMTP([]byte(tinyQwen35MTP))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	for name, shape := range mm.Tensors() {
		w[name] = shape
	}
	const cache = 64
	hidden := map[string]string{"hidden-out": "1"}
	newEngine := func(mtp bool) *Engine {
		gen := cpuPackage(t, cfg, w, llm.Options{Seq: 4, Cache: cache, Format: "fp16", Params: map[string]string{"hidden-out": "1", "state-rows": "1"}})
		pre := cpuPackage(t, cfg, w, llm.Options{Seq: 8, Cache: cache, Format: "fp16", LastRow: true, Params: hidden})
		for name := range pre.In {
			if llm.CacheInput(name) {
				pre.In[name] = gen.In[name]
			}
		}
		sess, err := decode.New(gen, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := sess.Add(pre); err != nil {
			t.Fatal(err)
		}
		e := &Engine{sess: sess, w: w, model: cfg, emb: map[int32][]float32{}, Stops: map[int32]bool{}}
		if mtp {
			ms, err := decode.New(cpuPackage(t, mm, w, llm.Options{Seq: 4, Cache: cache, Format: "fp16", LastRow: true}), mm)
			if err != nil {
				t.Fatal(err)
			}
			e.mtp = &mtpDrafter{e: e, sess: ms, k: 2, base: -1}
		}
		return e
	}
	// the third goes back to the first prompt's end and parts from what followed it
	prompts := [][]int32{{3, 1, 4, 1, 5, 9, 2, 6}, nil, {3, 1, 4, 1, 5, 9, 2, 6, 11, 12, 13, 14}}
	const n = 16
	generate := func(e *Engine, prompt []int32) ([]int32, Stats) {
		var got []int32
		st, err := e.Generate(context.Background(), Request{Prompt: prompt, MaxTokens: n}, func(gen []int32) (bool, error) {
			got = gen
			return false, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return got, st
	}

	plain := newEngine(false)
	e := newEngine(true)
	type guess struct {
		seq   []int32
		first int32
	}
	var guesses []guess
	e.drafter = func(seq []int32, k int) []int32 {
		d, err := e.mtp.guess(seq, min(k, e.mtp.k))
		if err != nil {
			t.Fatal(err)
		}
		if len(d) > 0 {
			guesses = append(guesses, guess{slices.Clone(seq), d[0]})
		}
		return d
	}
	for i := range prompts {
		if i == 1 {
			// the next turn: the first prompt, the answer and more
			prompts[1] = append(append(slices.Clone(prompts[0]), plainOut(t, plain, prompts[0], n)...), 7, 0, 8)
		}
		want, _ := generate(plain, prompts[i])
		got, st := generate(e, prompts[i])
		if !slices.Equal(got, want) {
			t.Errorf("prompt %d: with drafts %v, one at a time %v", i, got, want)
		}
		if st.Drafted == 0 || st.Calls >= n {
			t.Errorf("prompt %d: %d drafts, %d calls for %d tokens", i, st.Drafted, st.Calls, n)
		}
		t.Logf("prompt %d: %d of %d drafts accepted, %d calls for %d tokens", i, st.Accepted, st.Drafted, st.Calls, n)
	}

	// each first draft against the layer over the whole sequence, without a cache
	mismatch := 0
	for _, g := range guesses {
		logits := mtpFromStart(t, cfg, mm, w, g.seq)
		if top := argmax(logits); top != g.first && logits[g.first] < logits[top]-1e-3 {
			mismatch++
			t.Errorf("after %v: drafted %d, the layer from the start guesses %d (%g against %g)", g.seq, g.first, top, logits[g.first], logits[top])
		}
	}
	t.Logf("%d first drafts checked", len(guesses))
	if len(guesses) < 4 {
		t.Errorf("only %d drafts checked", len(guesses))
	}
}

// plainOut generates n tokens after prompt one at a time on a fresh sequence.
func plainOut(t *testing.T, e *Engine, prompt []int32, n int) []int32 {
	var got []int32
	if _, err := e.Generate(context.Background(), Request{Prompt: prompt, MaxTokens: n}, func(gen []int32) (bool, error) {
		got = gen
		return false, nil
	}); err != nil {
		t.Fatal(err)
	}
	return got
}

// mtpFromStart is the layer's logits guessing the token after seq: the model's final hidden
// states over seq but its last token, and the layer over all of them without a cache.
func mtpFromStart(t *testing.T, cfg *qwen35.Config, mm *qwen35.MTP, w llm.Weights, seq []int32) []float32 {
	t.Helper()
	d := cfg.HiddenSize
	n := len(seq) - 1
	run := func(m llm.Model, opts llm.Options, in map[string][]float32) []float32 {
		p := cpuPackage(t, m, w, opts)
		vals := map[string]milcpu.Tensor{}
		for name, v := range in {
			vals[name] = milcpu.Tensor{Shape: p.InShape[name], Data: v}
		}
		for name, v := range vals {
			if len(v.Data) != count(v.Shape) {
				t.Fatalf("%s: %d values for %v", name, len(v.Data), v.Shape)
			}
		}
		for name, v := range in {
			hwnpu.EncodeF16(p.In[name], v)
		}
		if err := p.Run(); err != nil {
			t.Fatal(err)
		}
		out := make([]float32, len(p.Out["y"])/2)
		hwnpu.DecodeF16(out, p.Out["y"])
		return out
	}
	x, err := llm.Embed(w, cfg, seq)
	if err != nil {
		t.Fatal(err)
	}
	h := run(cfg, llm.Options{Seq: n, Format: "fp16", Hidden: true}, map[string][]float32{"x": x[:n*d]})
	y := run(mm, llm.Options{Seq: n, Format: "fp16"}, map[string][]float32{"x": x[d:], "hid": h})
	return y[(n-1)*cfg.VocabSize:]
}

// With an output layer of some tokens alone (qwen35-mtp's draft-vocab), the i-th logit is of the
// manifest's i-th token; without, of token i. The architecture's draft-vocab follows
// HWNPU_MTP_VOCAB, and one without the setting has none.
func TestMTPToken(t *testing.T) {
	m := &mtpDrafter{ids: []int32{2, 7, 40}}
	if got := m.token(1); got != 7 {
		t.Errorf("logit 1 is of token %d, want 7", got)
	}
	m.ids = nil
	if got := m.token(1); got != 1 {
		t.Errorf("without ids, logit 1 is of token %d", got)
	}
	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(tinyQwen35MTP), 0o644); err != nil {
		t.Fatal(err)
	}
	ma, _, err := llm.LoadConfig(path, "qwen35-mtp")
	if err != nil {
		t.Fatal(err)
	}
	if k := draftVocab(ma); k != mtpVocab {
		t.Errorf("draft-vocab %d, want %d", k, mtpVocab)
	}
	t.Setenv("HWNPU_MTP_VOCAB", "0")
	if k := draftVocab(ma); k != 0 {
		t.Errorf("HWNPU_MTP_VOCAB=0: draft-vocab %d", k)
	}
	if k := draftVocab(&llm.Arch{Name: "other"}); k != 0 {
		t.Errorf("an architecture without draft-vocab: %d", k)
	}
}
