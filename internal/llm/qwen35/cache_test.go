package qwen35

import (
	"fmt"
	"math"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// cpuProgram builds the program with a cache and runs it on the CPU interpreter as a
// decode.Program. The inputs that hold the state between steps (llm.CacheInput) are taken from
// shared when given, so that two programs continue one sequence.
func cpuProgram(t *testing.T, cfg *Config, w llm.Weights, opts llm.Options, shared map[string][]byte) decode.Program {
	t.Helper()
	prog, ins, outs, _ := buildProgram(t, cfg, w, opts)
	p := decode.Program{In: map[string][]byte{}, Out: map[string][]byte{}, InShape: map[string][]int{}, OutShape: map[string][]int{}}
	for _, in := range ins {
		p.InShape[in.Name] = in.Shape
		if b, ok := shared[in.Name]; ok && llm.CacheInput(in.Name) {
			p.In[in.Name] = b
			continue
		}
		p.In[in.Name] = make([]byte, 2*count(in.Shape))
	}
	for _, o := range outs {
		p.OutShape[o.Name] = o.Shape
		p.Out[o.Name] = make([]byte, 2*count(o.Shape))
	}
	p.Run = func() error {
		in := map[string]milcpu.Tensor{}
		for name, sh := range p.InShape {
			v := make([]float32, count(sh))
			hwnpu.DecodeF16(v, p.In[name])
			in[name] = milcpu.Tensor{Shape: sh, Data: v}
		}
		res, err := prog.Run(in, milcpu.Options{})
		if err != nil {
			return err
		}
		for name, o := range res {
			hwnpu.EncodeF16(p.Out[name], o.Data)
		}
		return nil
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

// Stepping the program with a cache through a sequence gives the logits of the program without
// one at every position: the KV caches, the convolution windows and the recurrent states carried
// from step to step, RoPE at the step's positions, and a first step padded past its tokens, which
// must leave the recurrent state as it is for the steps after it.
func TestCachedMatchesFull(t *testing.T) {
	cfg := tinyConfig(t)
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	n := len(tokens)
	emb := embed16(t, cfg, w, tokens)
	full := runCPU(t, cfg, w, llm.Options{Seq: n, Format: "fp16"}, emb)
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	for _, c := range []struct{ q, cache int }{{1, 10}, {3, 12}, {4, 10}, {10, 10}} {
		t.Run(fmt.Sprintf("q%d cache%d", c.q, c.cache), func(t *testing.T) {
			s, err := decode.New(cpuProgram(t, cfg, w, llm.Options{Seq: c.q, Cache: c.cache, Format: "fp16"}, nil), cfg)
			if err != nil {
				t.Fatal(err)
			}
			worst := 0.0
			for pos := 0; pos < n; {
				v := min(c.q, n-pos)
				if pos == 0 && c.q > 1 {
					v = c.q - 1 // a padded step before others
				}
				var step [][]float32
				for i := 0; i < v; i++ {
					step = append(step, row(pos+i))
				}
				logits, err := s.Step(step)
				if err != nil {
					t.Fatal(err)
				}
				pos += v
				worst = math.Max(worst, relErr(logits, full[(pos-1)*cfg.VocabSize:pos*cfg.VocabSize]))
			}
			// the host keeps the caches and states in fp16; the program without a cache, in fp32
			if worst > 2e-3 {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			t.Logf("worst relative error %.2g", worst)
		})
	}
}

// A session that drives two programs over one state (a long step for prompts, a short one to
// generate), goes back to a mark after a detour, and keeps a beginning of a speculative step's
// tokens, gives the logits of the program without a cache. The recurrent states come back with
// the mark, and the step that kept fewer tokens than it ran leaves the state after those.
func TestShareRewindSpeculate(t *testing.T) {
	cfg := tinyConfig(t)
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	detour := []int32{7, 7, 0, 8, 2, 7}
	n := len(tokens)
	emb := embed16(t, cfg, w, append(append([]int32(nil), tokens...), detour...))
	full := runCPU(t, cfg, w, llm.Options{Seq: n, Format: "fp16"}, emb[:n*cfg.HiddenSize])
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	const cache = 16
	short := cpuProgram(t, cfg, w, llm.Options{Seq: 1, Cache: cache, Format: "fp16"}, nil)
	long := cpuProgram(t, cfg, w, llm.Options{Seq: 4, Cache: cache, Format: "fp16"}, short.In)
	s, err := decode.New(short, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Add(long); err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	check := func(logits []float32, last int) {
		t.Helper()
		if last < n {
			worst = math.Max(worst, relErr(logits, full[last*cfg.VocabSize:(last+1)*cfg.VocabSize]))
		}
	}
	step := func(rows ...int) {
		t.Helper()
		var e [][]float32
		for _, r := range rows {
			e = append(e, row(r))
		}
		logits, err := s.Step(e)
		if err != nil {
			t.Fatal(err)
		}
		check(logits, rows[len(rows)-1])
	}
	step(0, 1, 2, 3) // long
	step(4)          // short
	m := s.Mark()
	step(n, n+1, n+2, n+3) // a detour that changes every state
	step(n + 4)
	s.Rewind(m)
	if s.Pos() != 5 {
		t.Fatalf("position %d after rewinding, want 5", s.Pos())
	}
	// tokens 5, 6 and then two drafts that are not the sequence's: only 5 and 6 are kept
	e := [][]float32{row(5), row(6), row(n), row(n + 1)}
	logits, kept, err := s.Speculate(e, func(r func(i int) []float32) int {
		check(r(0), 5)
		check(r(1), 6)
		return 2
	})
	if err != nil || kept != 2 || s.Pos() != 7 {
		t.Fatalf("speculate kept %d to position %d, %v; want 2 to 7", kept, s.Pos(), err)
	}
	check(logits, 6)
	for i := 7; i < n; i++ {
		step(i)
	}
	if worst > 2e-3 {
		t.Errorf("relative error %g against the program without a cache", worst)
	}
	t.Logf("worst relative error %.2g", worst)
}
