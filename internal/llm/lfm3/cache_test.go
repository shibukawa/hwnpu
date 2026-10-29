package lfm3

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// cpuProgram loads the parts in dir on the CPU interpreter as a decode.Program: y of each part
// feeds x of the next, every other input is one buffer shared by the parts that take it.
func cpuProgram(t *testing.T, dir string) decode.Program {
	t.Helper()
	pdir, list, err := llm.ReadParts(dir)
	if err != nil {
		t.Fatal(err)
	}
	p := decode.Program{In: map[string][]byte{}, Out: map[string][]byte{}, InShape: map[string][]int{}, OutShape: map[string][]int{}}
	var progs []*milcpu.Program
	for i, part := range list.Parts {
		prog, f, err := milcpu.OpenPackage(filepath.Join(pdir, part.Package))
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
		m, err := mlpackage.ReadModel(filepath.Join(pdir, part.Package))
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
			dir := filepath.Join(t.TempDir(), "parts")
			opts := llm.Options{Seq: c.q, Cache: c.cache, Format: "fp16", Params: map[string]string{"post-scale": "1"}}
			if _, err := llm.WriteParts(dir, cfg, w, opts, c.parts, nil); err != nil {
				t.Fatal(err)
			}
			s, err := decode.New(cpuProgram(t, dir), cfg)
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
