package plamo3

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
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
// earlier part's output takes that output (none in PLaMo 3, whose parts pass x alone), and every
// other input is one buffer shared by the parts that take it.
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

// step runs tokens [from, to) through a session and returns the logits after the last.
func step(t *testing.T, s interface {
	Step([][]float32) ([]float32, error)
}, c *Config, w llm.Weights, from, to int) []float32 {
	t.Helper()
	var rs [][]float32
	for i := from; i < to; i++ {
		row, err := llm.TokenRow(w, c, tokens[i])
		if err != nil {
			t.Fatal(err)
		}
		rs = append(rs, row)
	}
	logits, err := s.Step(rs)
	if err != nil {
		t.Fatal(err)
	}
	return logits
}

// The model in parts gives the logits of the whole.
func TestPartsMatchWhole(t *testing.T) {
	c := tinyConfig(t)
	w := llm.NewSynthetic(c)
	n := len(tokens)
	feed := map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)}
	whole := build(t, c, w, llm.Options{Seq: n, Format: "fp16"}).run(t, feed)["y"]
	for parts := 2; parts <= 4; parts++ {
		path, plan := writeParts(t, c, w, llm.Options{Seq: n, Format: "fp16"}, parts)
		p := cpuProgram(t, path)
		hwnpu.EncodeF16(p.In["x"], feed["x"])
		if err := p.Run(); err != nil {
			t.Fatal(err)
		}
		got := make([]float32, len(whole))
		hwnpu.DecodeF16(got, p.Out["y"])
		if e := relErr32(got, whole); e > 2e-3 {
			t.Errorf("%d parts %v: relative error %g", parts, plan, e)
		}
	}
}

// Stepping the program with a KV cache through a sequence gives the logits of the program without
// one at every position: the rings of the sliding layers, RoPE positions (YaRN's on the full
// layers) and padded steps included, in one part and in several.
func TestCachedMatchesFull(t *testing.T) {
	c := tinyConfig(t)
	w := llm.NewSynthetic(c)
	n := len(tokens)
	full := build(t, c, w, llm.Options{Seq: n, Format: "fp16"}).run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})["y"]
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
				logits := step(t, s, c, w, pos, pos+v)
				pos += v
				worst = math.Max(worst, relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]))
			}
			// the cache holds fp16 keys and values; the program without one keeps them in fp32
			if worst > 5e-3 {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			t.Logf("parts %v: worst relative error %.2g", plan, worst)
		})
	}
}

// A mark taken before a sliding layer's ring is full holds only the slots in use (decode's
// ringRows), and brings the sequence back even after the ring has wrapped past it; a saved
// sequence resumed in another session from the mark's buffers (as the prompt store does) too.
func TestCachedMarkBeforeRingFull(t *testing.T) {
	raw := strings.Replace(tiny, `"window_size": 3`, `"window_size": 8`, 1)
	c, err := ParseConfig([]byte(raw))
	if err != nil {
		t.Fatal(err)
	}
	w := llm.NewSynthetic(c)
	n := len(tokens)
	full := build(t, c, w, llm.Options{Seq: n, Format: "fp16"}).run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})["y"]
	path, _ := writeParts(t, c, w, llm.Options{Seq: 1, Cache: 16, Format: "fp16"}, 1)
	open := func() *decode.Session {
		s, err := decode.New(cpuProgram(t, path), c)
		if err != nil {
			t.Fatal(err)
		}
		return s
	}
	check := func(what string, logits []float32, pos int) {
		t.Helper()
		if e := relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]); e > 5e-3 {
			t.Errorf("%s: relative error %g after %d tokens", what, e, pos)
		}
	}
	one := func(s *decode.Session, from, to int) []float32 {
		var logits []float32
		for i := from; i < to; i++ {
			logits = step(t, s, c, w, i, i+1)
		}
		return logits
	}
	s := open()
	one(s, 0, 5)
	mark := s.Mark()
	ringBytes := 0
	for _, b := range mark.Bufs() {
		ringBytes += len(b)
	}
	// 2 sliding layers, keys and values, 2 kv heads, 5 of 8 slots, 16 values of fp16
	if want := 2 * 2 * 2 * 5 * 16 * 2; ringBytes != want {
		t.Errorf("the mark holds %d bytes of rings, want %d (the 5 slots in use)", ringBytes, want)
	}
	check("on past the window", one(s, 5, 11), 11) // positions 8 to 10 wrap to slots 0 to 2
	s.Rewind(mark)
	check("back to 5, on", one(s, 5, 11), 11)
	// the prompt store keeps the rows before a mark and the mark's buffers, and resumes them
	// in a new session: the full layers' rows 0 to 4 are as they were at the mark
	rows, err := s.ExportRows(0, 5)
	if err != nil {
		t.Fatal(err)
	}
	r := open()
	if err := r.Resume(rows, decode.NewMark(5, mark.Bufs())); err != nil {
		t.Fatal(err)
	}
	check("resumed elsewhere, on", one(r, 5, 11), 11)
}
