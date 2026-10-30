package chat

import (
	"context"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

const tinyLFM3 = `{"hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 3, "num_attention_heads": 16,
 "num_key_value_heads": 8, "head_dim": 8, "vocab_size": 48, "norm_eps": 1e-5, "conv_L_cache": 3, "conv_bias": true,
 "sliding_window": 3, "use_sandwich_norm": true, "layer_types": ["conv", "full_attention", "sliding_attention"],
 "rope_parameters": {"full_attention": {"rope_theta": 10000, "partial_rotary_factor": 0.5}, "sliding_attention": {"rope_theta": 10000}}}`

// cpuProgram writes the model as one package with a KV cache and loads it on the CPU interpreter.
func cpuProgram(t *testing.T, cfg *lfm3.Config, w llm.Weights, q, cache int) decode.Program {
	t.Helper()
	return cpuPackage(t, cfg, w, llm.Options{Seq: q, Cache: cache, Format: "fp16", Params: map[string]string{"post-scale": "1"}})
}

// cpuPackage writes the model as one package and loads it on the CPU interpreter.
func cpuPackage(t *testing.T, model llm.Model, w llm.Weights, opts llm.Options) decode.Program {
	t.Helper()
	path := filepath.Join(t.TempDir(), "m.mlpackage")
	if _, err := llm.WritePart(path, model, w, opts, nil); err != nil {
		t.Fatal(err)
	}
	prog, f, err := milcpu.OpenPackage(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	p := decode.Program{In: map[string][]byte{}, Out: map[string][]byte{}, InShape: prog.Inputs(), OutShape: map[string][]int{}}
	for name, sh := range p.InShape {
		p.In[name] = make([]byte, 2*count(sh))
	}
	m, err := mlpackage.ReadModel(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, o := range m.GetDescription().GetOutput() {
		var sh []int
		for _, d := range o.GetType().GetMultiArrayType().GetShape() {
			sh = append(sh, int(d))
		}
		p.OutShape[o.GetName()] = sh
		p.Out[o.GetName()] = make([]byte, 2*count(sh))
	}
	p.Run = func() error {
		in := map[string]milcpu.Tensor{}
		for name, sh := range p.InShape {
			v := make([]float32, count(sh))
			hwnpu.DecodeF16(v, p.In[name])
			in[name] = milcpu.Tensor{Shape: sh, Data: v}
		}
		res, err := prog.Run(in, milcpu.Options{})
		for name, o := range res {
			hwnpu.EncodeF16(p.Out[name], o.Data)
		}
		return err
	}
	return p
}

func count(sh []int) int {
	n := 1
	for _, d := range sh {
		n *= d
	}
	return n
}

// The engine reuses what it can of earlier prompts (marks inside the sequence, sequences put
// aside) and still gives the logits of reading the prompt from the start.
func TestEngineReuse(t *testing.T) {
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache = 40
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
	defer func(n int) { minSaved = n }(minSaved)
	minSaved = 4
	e := &Engine{sess: sess, w: w, model: cfg, emb: map[int32][]float32{}}

	ref, err := decode.New(cpuProgram(t, cfg, w, 4, cache), cfg)
	if err != nil {
		t.Fatal(err)
	}
	want := func(prompt []int32) []float32 {
		ref.Reset()
		var logits []float32
		for i := 0; i < len(prompt); i += 4 {
			var embs [][]float32
			for _, tok := range prompt[i:min(i+4, len(prompt))] {
				v, _ := llm.Embed(w, cfg, []int32{tok})
				embs = append(embs, v)
			}
			if logits, err = ref.Step(embs); err != nil {
				t.Fatal(err)
			}
		}
		return logits
	}

	rng := rand.New(rand.NewPCG(1, 2))
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
	a, x, y, z, b, v := toks(18), toks(5), toks(3), toks(6), toks(15), toks(7)
	c, tail, other, more := toks(5), toks(20), toks(6), toks(3)
	for _, c := range []struct {
		name   string
		prompt []int32
		reused int
	}{
		{"first", a, 0},
		{"continued", cat(a, x), 18},
		{"diverged after the first prompt", cat(a, y), 18},
		{"another conversation", b, 0},
		{"back to the first", cat(a, y, z), 21},
		{"back to the second", cat(b, v), 15},
		{"the same again", cat(b, v), 22},
		{"diverged inside the first prompt", cat(a[:10], z), 0},
		{"short", c, 0},
		{"long after short", cat(c, tail), 5},
		{"diverged after short", cat(c, other), 5}, // puts the long one aside
		{"back to long", cat(c, tail, more), 25},
	} {
		var st Stats
		got, err := e.prefill(context.Background(), c.prompt, &st)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if st.Reused != c.reused {
			t.Errorf("%s: reused %d tokens, want %d", c.name, st.Reused, c.reused)
		}
		if d := relErr(got, want(c.prompt)); d > 2e-3 {
			t.Errorf("%s: relative error %g against reading the prompt from the start", c.name, d)
		}
	}
}

func relErr(got, ref []float32) float64 {
	var num, den float64
	for i := range ref {
		d := float64(got[i] - ref[i])
		num += d * d
		den += float64(ref[i]) * float64(ref[i])
	}
	return math.Sqrt(num / den)
}

// A thinking budget ends the thinking by feeding the End tokens after Budget tokens inside it,
// and the sequence in the cache stays what reading prompt and answer from the start gives.
func TestThinkBudget(t *testing.T) {
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache = 40
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
	prompt := []int32{3, 1, 4, 1, 5, 9, 2, 6}
	think := ThinkLimit{Budget: 3, Open: 40, Close: 41, End: []int32{42, 43, 41}, Inside: true}
	var last []int32
	emits := 0
	st, err := e.Generate(context.Background(), Request{Prompt: prompt, MaxTokens: 12, Think: think}, func(gen []int32) (bool, error) {
		last, emits = gen, emits+1
		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	// the last token generated is emitted but not read into the cache
	gen := last
	if fed := e.cur.fed[len(prompt):]; !slices.Equal(fed, gen[:len(gen)-1]) || len(gen) != st.Generated {
		t.Fatalf("the sequence has %v, emit saw %v, %d generated", fed, last, st.Generated)
	}
	// the thinking is cut after 3 tokens unless the model closed it itself
	closed := slices.Index(gen[:3], think.Close)
	switch {
	case closed < 0 && (!st.ThinkCut || !slices.Equal(gen[3:6], think.End)):
		t.Errorf("thinking not cut after 3 tokens: cut %v, %v", st.ThinkCut, gen)
	case closed >= 0 && st.ThinkCut:
		t.Errorf("thinking cut after the model closed it: %v", gen)
	}
	if emits != st.Generated-len(think.End)*btoi(st.ThinkCut)+btoi(st.ThinkCut) {
		t.Errorf("%d emits for %d tokens", emits, st.Generated)
	}
	// the logits after the sequence are those of reading it from the start
	ref, err := decode.New(cpuProgram(t, cfg, w, 4, cache), cfg)
	if err != nil {
		t.Fatal(err)
	}
	all := append(slices.Clone(prompt), gen[:len(gen)-1]...)
	var want []float32
	for i := 0; i < len(all); i += 4 {
		var embs [][]float32
		for _, tok := range all[i:min(i+4, len(all))] {
			v, _ := llm.Embed(w, cfg, []int32{tok})
			embs = append(embs, v)
		}
		if want, err = ref.Step(embs); err != nil {
			t.Fatal(err)
		}
	}
	if d := relErr(e.cur.last, want); d > 2e-3 {
		t.Errorf("relative error %g against reading the sequence from the start", d)
	}
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Speculative decoding generates what plain decoding generates: drafts that are right are kept
// in one call, a wrong draft is dropped with those after it, and the cache holds the sequence as
// if read one token at a time.
func TestSpeculate(t *testing.T) {
	cfg, err := lfm3.ParseConfig([]byte(tinyLFM3))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(cfg)
	const cache = 48
	newEngine := func(qs ...int) *Engine {
		var progs []decode.Program
		for _, q := range qs {
			progs = append(progs, cpuProgram(t, cfg, w, q, cache))
		}
		for _, p := range progs[1:] {
			for name := range p.In {
				if strings.HasPrefix(name, "kv_") || strings.HasPrefix(name, "win_") {
					p.In[name] = progs[0].In[name]
				}
			}
		}
		sess, err := decode.New(progs[0], cfg)
		if err != nil {
			t.Fatal(err)
		}
		for _, p := range progs[1:] {
			if err := sess.Add(p); err != nil {
				t.Fatal(err)
			}
		}
		return &Engine{sess: sess, w: w, model: cfg, emb: map[int32][]float32{}, Stops: map[int32]bool{}}
	}
	prompt := []int32{3, 1, 4, 1, 5, 9, 2, 6}
	const n = 20
	// plain: one token at a time
	plain := newEngine(1, 8)
	if plain.Draft() != 0 {
		t.Fatalf("drafts of %d with programs of 1 and 8 tokens", plain.Draft())
	}
	var want []int32
	if _, err := plain.Generate(context.Background(), Request{Prompt: prompt, MaxTokens: n}, func(gen []int32) (bool, error) {
		want = gen
		return false, nil
	}); err != nil {
		t.Fatal(err)
	}
	if len(want) != n {
		t.Fatalf("plain generated %d tokens", len(want))
	}
	// speculative: a drafter that knows the answer, wrong at every third guess
	e := newEngine(4, 8)
	if e.Draft() != 3 {
		t.Fatalf("drafts of %d with programs of 4 and 8 tokens", e.Draft())
	}
	guesses := 0
	e.drafter = func(seq []int32, k int) []int32 {
		at := len(seq) - len(prompt) // tokens of the answer so far; the next is want[at]
		var d []int32
		for i := 0; i < k && at+i < len(want); i++ {
			guesses++
			tok := want[at+i]
			if guesses%3 == 0 {
				tok = (tok + 1) % int32(cfg.VocabSize)
			}
			d = append(d, tok)
		}
		return d
	}
	var got []int32
	st, err := e.Generate(context.Background(), Request{Prompt: prompt, MaxTokens: n}, func(gen []int32) (bool, error) {
		got = gen
		return false, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, want) {
		t.Errorf("speculative generated %v, plain %v", got, want)
	}
	if st.Accepted == 0 || st.Accepted >= st.Drafted || st.Calls >= n-1 {
		t.Errorf("%d of %d drafts accepted in %d calls", st.Accepted, st.Drafted, st.Calls)
	}
	t.Logf("%d of %d drafts accepted, %d calls for %d tokens", st.Accepted, st.Drafted, st.Calls, n)
	// the cache holds the answer up to the last token (which may be in it, when a draft)
	fed := e.cur.fed[len(prompt):]
	if len(fed) < len(got)-1 || !slices.Equal(fed, got[:len(fed)]) {
		t.Errorf("the cache holds %v", fed)
	}
	if d := relErr(e.cur.last, refLogits(t, cfg, w, cache, e.cur.fed)); d > 2e-3 {
		t.Errorf("relative error %g against reading the sequence from the start", d)
	}
}

// refLogits reads seq from the start on a fresh program and returns the logits after it.
func refLogits(t *testing.T, cfg *lfm3.Config, w llm.Weights, cache int, seq []int32) []float32 {
	t.Helper()
	ref, err := decode.New(cpuProgram(t, cfg, w, 4, cache), cfg)
	if err != nil {
		t.Fatal(err)
	}
	var logits []float32
	for i := 0; i < len(seq); i += 4 {
		var embs [][]float32
		for _, tok := range seq[i:min(i+4, len(seq))] {
			v, _ := llm.Embed(w, cfg, []int32{tok})
			embs = append(embs, v)
		}
		if logits, err = ref.Step(embs); err != nil {
			t.Fatal(err)
		}
	}
	return logits
}

func TestLookupDraft(t *testing.T) {
	seq := []int32{7, 8, 9, 1, 2, 3, 4, 5, 6, 1, 2, 3}
	if got := LookupDraft(seq, 4); !slices.Equal(got, []int32{4, 5, 6, 1}) {
		t.Errorf("3-gram: %v", got)
	}
	seq = []int32{1, 2, 7, 8, 9, 1, 2}
	if got := LookupDraft(seq, 4); !slices.Equal(got, []int32{7, 8, 9, 1}) {
		t.Errorf("2-gram: %v", got)
	}
	if got := LookupDraft([]int32{1, 2, 3, 4}, 4); got != nil {
		t.Errorf("no match: %v", got)
	}
	// the latest occurrence is taken, and the draft stops at the end
	seq = []int32{1, 2, 3, 9, 1, 2, 3, 8, 1, 2, 3}
	if got := LookupDraft(seq, 4); !slices.Equal(got, []int32{8, 1, 2, 3}) {
		t.Errorf("latest: %v", got)
	}
}
