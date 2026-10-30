package gemma4

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

// writeParts converts the model in n parts, as hwnpu-convert does, and returns the manifest.
func writeParts(t *testing.T, c *Config, w llm.Weights, opts llm.Options, n int) (string, []llm.Part) {
	t.Helper()
	plan, err := llm.PlanSplit(c, opts, math.MaxInt64, n)
	if err != nil || len(plan) != n {
		t.Fatalf("plan of %d parts: %v, %v", n, plan, err)
	}
	dir := t.TempDir()
	man := &llm.Manifest{Seq: opts.Seq, Cache: opts.Cache}
	for i, p := range plan {
		o := opts
		if n > 1 {
			o.Start, o.Layers = p.Start, p.Layers
		}
		rec, err := llm.WritePart(filepath.Join(dir, fmt.Sprintf("m.part%d.mlpackage", i+1)), c, w, o, nil)
		if err != nil {
			t.Fatal(err)
		}
		man.Parts = append(man.Parts, rec)
	}
	path := filepath.Join(dir, "m.parts.json")
	if err := llm.WriteManifest(path, man); err != nil {
		t.Fatal(err)
	}
	return path, plan
}

// cpuProgram loads the parts of a manifest on the CPU interpreter as a decode.Program, bound as
// internal/llm/chain binds them: y of each part feeds x of the next, an input named like an
// earlier part's output takes that output, and every other input is one buffer shared by the
// parts that take it.
func cpuProgram(t *testing.T, manifest string) decode.Program {
	t.Helper()
	list, err := llm.ReadManifest(manifest)
	if err != nil {
		t.Fatal(err)
	}
	p := decode.Program{In: map[string][]byte{}, Out: map[string][]byte{}, InShape: map[string][]int{}, OutShape: map[string][]int{}}
	var progs []*milcpu.Program
	passed := map[string]bool{}
	for i, part := range list.Parts {
		prog, f, err := milcpu.OpenPackage(part.Path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { f.Close() })
		progs = append(progs, prog)
		for name, sh := range prog.Inputs() {
			switch {
			case name == "x" && i > 0:
			case p.Out[name] != nil:
				passed[name] = true
			default:
				p.InShape[name] = sh
				p.In[name] = make([]byte, 2*count(sh))
			}
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
	if len(passed) == 0 && len(list.Parts) > 1 {
		t.Errorf("%d parts pass nothing but x between them", len(list.Parts))
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
				b := p.In[name]
				if passed[name] {
					b = p.Out[name]
				}
				v := make([]float32, count(sh))
				hwnpu.DecodeF16(v, b)
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

func relErr32(got, ref []float32) float64 {
	var se, sr float64
	for i := range ref {
		d := float64(got[i] - ref[i])
		se += d * d
		sr += float64(ref[i]) * float64(ref[i])
	}
	return math.Sqrt(se / sr)
}

// The model in parts gives the logits of the whole: the per-layer inputs and the shared keys and
// values go from the part that computes them to the parts that use them.
func TestPartsMatchWhole(t *testing.T) {
	c := tinyConfig(t)
	w := newWeights(c)
	n := len(tokens)
	feed := map[string][]float32{"x": rows(t, w, c.Embedding(), tokens), "ple": rows(t, w, c.perLayerTable(), tokens)}
	whole := build(t, c, w, Options{Options: llm.Options{Seq: n, Format: "fp16"}}).run(t, feed)["y"]
	for parts := 2; parts <= 4; parts++ {
		path, plan := writeParts(t, c, w, llm.Options{Seq: n, Format: "fp16"}, parts)
		p := cpuProgram(t, path)
		for name, v := range feed {
			hwnpu.EncodeF16(p.In[name], v)
		}
		if err := p.Run(); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, len(whole))
		hwnpu.DecodeF16(got, p.Out["y"])
		e := relErr32(got, whole)
		if e > 2e-3 {
			t.Errorf("%d parts %v: relative error %g", parts, plan, e)
		}
		t.Logf("%d parts %v: relative error %.2g", parts, plan, e)
	}
}

// Stepping the program with a KV cache through a sequence gives the logits of the program without
// one at every position: the rings of the sliding layers, the shared keys and values of a step
// (the cache followed by the step's), RoPE positions and padded steps included, in one part and
// in several.
func TestCachedMatchesFull(t *testing.T) {
	c := tinyConfig(t)
	w := newWeights(c)
	n := len(tokens)
	feed := map[string][]float32{"x": rows(t, w, c.Embedding(), tokens), "ple": rows(t, w, c.perLayerTable(), tokens)}
	full := build(t, c, w, Options{Options: llm.Options{Seq: n, Format: "fp16"}}).run(t, feed)["y"]
	for _, k := range []struct{ q, cache, parts int }{{1, 11, 1}, {1, 12, 3}, {3, 11, 2}, {4, 16, 2}, {11, 11, 1}} {
		t.Run(fmt.Sprintf("q%d cache%d parts%d", k.q, k.cache, k.parts), func(t *testing.T) {
			path, plan := writeParts(t, c, w, llm.Options{Seq: k.q, Cache: k.cache, Format: "fp16"}, k.parts)
			s, err := decode.New(cpuProgram(t, path), c)
			if err != nil {
				t.Fatal(err)
			}
			worst := 0.0
			for pos := 0; pos < n; {
				v := min(k.q, n-pos)
				var step [][]float32
				for i := 0; i < v; i++ {
					row, err := llm.TokenRow(w, c, tokens[pos+i])
					if err != nil {
						t.Fatal(err)
					}
					step = append(step, row)
				}
				logits, err := s.Step(step)
				if err != nil {
					t.Fatal(err)
				}
				pos += v
				worst = math.Max(worst, relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]))
			}
			// The cache holds fp16 keys and values; the program without one keeps them in fp32. With
			// unscaled attention scores (up to about 30 here) the tiny model is sensitive to that:
			// rounding every operation to fp16 moves its logits by 0.2 to 3 percent.
			if worst > 5e-3 {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			t.Logf("parts %v: worst relative error %.2g", plan, worst)
		})
	}
}
