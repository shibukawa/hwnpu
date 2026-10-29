package lfm3

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// cpuProgram loads the parts of a manifest on the CPU interpreter as a decode.Program: y of each
// part feeds x of the next, every other input is one buffer shared by the parts that take it.
func cpuProgram(t *testing.T, manifest string) decode.Program {
	t.Helper()
	list, err := llm.ReadManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	p := decode.Program{In: map[string][]byte{}, Out: map[string][]byte{}, InShape: map[string][]int{}, OutShape: map[string][]int{}}
	var progs []*milcpu.Program
	for i, part := range list.Parts {
		prog, f, err := milcpu.OpenPackage(part.Path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { f.Close() })
		progs = append(progs, prog)
		for name, sh := range prog.Inputs() {
			if name == "x" && i > 0 {
				continue
			}
			p.InShape[name] = sh
			p.In[name] = make([]byte, 2*count(sh))
		}
		m, err := mlpackage.ReadModel(part.Path)
		if err != nil {
			t.Fatal(err)
		}
		for _, o := range m.GetDescription().GetOutput() {
			if o.GetName() == "y" && i < len(list.Parts)-1 {
				continue
			}
			var sh []int
			for _, d := range o.GetType().GetMultiArrayType().GetShape() {
				sh = append(sh, int(d))
			}
			p.OutShape[o.GetName()] = sh
			p.Out[o.GetName()] = make([]byte, 2*count(sh))
		}
	}
	p.Run = func() error {
		var y milcpu.Tensor
		for i, prog := range progs {
			in := map[string]milcpu.Tensor{}
			for name, sh := range prog.Inputs() {
				if name == "x" && i > 0 {
					in[name] = y
					continue
				}
				v := make([]float32, count(sh))
				hwnpu.DecodeF16(v, p.In[name])
				in[name] = milcpu.Tensor{Shape: sh, Data: v}
			}
			res, err := prog.Run(in, milcpu.Options{})
			if err != nil {
				return err
			}
			for name, o := range res {
				if name == "y" {
					y = o
					if i < len(progs)-1 {
						continue
					}
				}
				hwnpu.EncodeF16(p.Out[name], o.Data)
			}
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

// Stepping the program with a KV cache through a sequence gives the logits of the program without
// one at every position: conv windows, cached keys and values, the sliding window, RoPE positions
// and padded steps included, in one part and in several.
func TestCachedMatchesFull(t *testing.T) {
	cfg := tinyConfig()
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6}
	n := len(tokens)
	emb, err := llm.Embed(w, cfg, tokens)
	if err != nil {
		t.Fatal(err)
	}
	full := runCPU(t, cfg, w, Options{Seq: n, Format: "fp16"}, emb)
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	for _, c := range []struct{ q, cache, parts int }{{1, 8, 1}, {1, 10, 3}, {3, 9, 2}, {8, 8, 1}} {
		t.Run(fmt.Sprintf("q%d cache%d parts%d", c.q, c.cache, c.parts), func(t *testing.T) {
			opts := llm.Options{Seq: c.q, Cache: c.cache, Format: "fp16", Params: map[string]string{"post-scale": "1"}}
			s, err := decode.New(cpuProgram(t, writeParts(t, cfg, w, opts, c.parts)), cfg)
			if err != nil {
				t.Fatal(err)
			}
			worst := 0.0
			for pos := 0; pos < n; {
				v := min(c.q, n-pos)
				var step [][]float32
				for i := 0; i < v; i++ {
					step = append(step, row(pos+i))
				}
				logits, err := s.Step(step)
				if err != nil {
					t.Fatal(err)
				}
				pos += v
				want := full[(pos-1)*cfg.VocabSize : pos*cfg.VocabSize]
				worst = math.Max(worst, relErr(logits, want))
			}
			// The cache holds fp16 keys, values and windows; the program without one keeps them in fp32.
			if worst > 2e-3 {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			t.Logf("worst relative error %.2g", worst)
		})
	}
}

// A session that drives two programs over one cache (a long step for prompts, a short one to
// generate) and goes back to a mark after taking another path gives the logits of the program
// without a cache: the rings of the sliding layer and the convolution windows come back with the
// mark.
func TestCachedShareAndRewind(t *testing.T) {
	cfg := tinyConfig()
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	detour := []int32{7, 7, 0, 8, 2, 7}
	n := len(tokens)
	emb, err := llm.Embed(w, cfg, append(append([]int32(nil), tokens...), detour...))
	if err != nil {
		t.Fatal(err)
	}
	full := runCPU(t, cfg, w, Options{Seq: n, Format: "fp16"}, emb[:n*cfg.HiddenSize])
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	const cache = 12
	program := func(q int) decode.Program {
		opts := llm.Options{Seq: q, Cache: cache, Format: "fp16", Params: map[string]string{"post-scale": "1"}}
		return cpuProgram(t, writeParts(t, cfg, w, opts, 2))
	}
	short, long := program(1), program(4)
	for name := range long.In {
		if strings.HasPrefix(name, "kv_") || strings.HasPrefix(name, "win_") {
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
	if s.Q() != 4 {
		t.Fatalf("Q is %d, want 4", s.Q())
	}
	worst := 0.0
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
		if last := rows[len(rows)-1]; last < n {
			worst = math.Max(worst, relErr(logits, full[last*cfg.VocabSize:(last+1)*cfg.VocabSize]))
		}
	}
	step(0, 1, 2, 3) // long
	step(4)          // short
	m := s.Mark()
	step(n, n+1, n+2, n+3) // a detour that overwrites the rings and the windows
	step(n + 4)
	step(n + 5)
	s.Rewind(m)
	if s.Pos() != 5 {
		t.Fatalf("position %d after rewinding, want 5", s.Pos())
	}
	step(5, 6)
	for i := 7; i < n; i++ {
		step(i)
	}
	if worst > 2e-3 {
		t.Errorf("relative error %g against the program without a cache", worst)
	}
	t.Logf("worst relative error %.2g", worst)

	other := program(4) // its own buffers
	if err := s.Add(other); err == nil {
		t.Error("Add took a program that does not share the cache")
	}
}

// A session moved to a tier with a longer cache (decode.Session.Migrate) goes on exactly as the
// longer program would have from the start: the sequence is run on a cache of 5, moved to one of
// 12 when that is full, and compared with the program without a cache at every position.
func TestMigrateTiers(t *testing.T) {
	cfg := tinyConfig()
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6}
	n := len(tokens)
	emb, err := llm.Embed(w, cfg, tokens)
	if err != nil {
		t.Fatal(err)
	}
	full := runCPU(t, cfg, w, Options{Seq: n, Format: "fp16"}, emb)
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	opts := llm.Options{Seq: 1, Format: "fp16", Params: map[string]string{"post-scale": "1"}}
	small, large := opts, opts
	small.Cache, large.Cache = 5, 12
	s, err := decode.New(cpuProgram(t, writeParts(t, cfg, w, small, 2)), cfg)
	if err != nil {
		t.Fatal(err)
	}
	largeProgram := cpuProgram(t, writeParts(t, cfg, w, large, 3))
	worst := 0.0
	for pos := 0; pos < n; pos++ {
		if s.Pos() == s.Cache() {
			if err := s.Migrate(largeProgram); err != nil {
				t.Fatal(err)
			}
			if s.Cache() != 12 {
				t.Fatalf("cache %d after the move", s.Cache())
			}
		}
		logits, err := s.Step([][]float32{row(pos)})
		if err != nil {
			t.Fatal(err)
		}
		worst = math.Max(worst, relErr(logits, full[pos*cfg.VocabSize:(pos+1)*cfg.VocabSize]))
	}
	if worst > 2e-3 {
		t.Errorf("relative error %g against the program without a cache", worst)
	}
	t.Logf("worst relative error %.2g", worst)
	// Moving back to the shorter cache is refused.
	if err := s.Migrate(cpuProgram(t, writeParts(t, cfg, w, small, 2))); err == nil {
		t.Error("moving to a shorter cache did not fail")
	}
}
