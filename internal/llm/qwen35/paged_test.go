package qwen35

import (
	"fmt"
	"io"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// convertPaged converts the model split at its attention with hwnpu-convert's code path.
func convertPaged(t *testing.T, cfg *Config, w llm.Weights, q, cache, chunk int) *llm.Manifest {
	t.Helper()
	opts := llm.Options{Seq: q, Cache: cache, Paged: chunk, Format: "fp16"}
	path, err := llm.Convert(Arch, cfg, w, filepath.Join(t.TempDir(), "m.mlpackage"), "none", 0, opts, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	man, err := llm.ReadManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	return man
}

// cpuBuf is a decode.Buf in plain memory.
type cpuBuf []byte

func (b cpuBuf) Bytes() []byte { return b }
func (b cpuBuf) Close() error  { return nil }

// cpuPaged loads a model split at its attention on the CPU interpreter as a decode.Program, as
// internal/llm/chain does on a device (adapted from internal/llm/lfm3's test): x of each part is y of the one before, an input named like
// an earlier part's output takes it, and the stages end with the parts that output attn_q_N.
// allocated counts the bytes NewBuf gave.
func cpuPaged(t *testing.T, man *llm.Manifest, allocated *int) decode.Program {
	t.Helper()
	p := decode.Program{In: map[string][]byte{}, Out: map[string][]byte{}, InShape: map[string][]int{}, OutShape: map[string][]int{}, Context: man.Cache}
	type part struct {
		prog *milcpu.Program
		outs []string
	}
	var parts []part
	var stages [][]int
	var cur []int
	outShape := func(path string) map[string][]int {
		m, err := mlpackage.ReadModel(path)
		if err != nil {
			t.Fatal(err)
		}
		out := map[string][]int{}
		for _, o := range m.GetDescription().GetOutput() {
			var sh []int
			for _, d := range o.GetType().GetMultiArrayType().GetShape() {
				sh = append(sh, int(d))
			}
			out[o.GetName()] = sh
		}
		return out
	}
	for i, rec := range man.Parts {
		prog, f, err := milcpu.OpenPackage(rec.Path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { f.Close() })
		pt := part{prog: prog}
		for name, sh := range outShape(rec.Path) {
			if name == "y" && i < len(man.Parts)-1 {
				continue
			}
			p.OutShape[name], p.Out[name] = sh, make([]byte, 2*count(sh))
			pt.outs = append(pt.outs, name)
		}
		for name, sh := range prog.Inputs() {
			if name == "x" && i > 0 || p.Out[name] != nil {
				continue
			}
			p.InShape[name], p.In[name] = sh, make([]byte, 2*count(sh))
		}
		parts = append(parts, pt)
		cur = append(cur, i)
		for _, name := range pt.outs {
			var layer int
			if _, err := fmt.Sscanf(name, "attn_q_%d", &layer); err == nil {
				stages, cur = append(stages, cur), nil
				p.Attend = append(p.Attend, layer)
			}
		}
	}
	stages = append(stages, cur)
	var y milcpu.Tensor
	p.RunPart = func(s int) error {
		for _, i := range stages[s] {
			in := map[string]milcpu.Tensor{}
			for name, sh := range parts[i].prog.Inputs() {
				if name == "x" && i > 0 {
					in[name] = y
					continue
				}
				b := p.In[name]
				if b == nil {
					b = p.Out[name]
				}
				v := make([]float32, count(sh))
				hwnpu.DecodeF16(v, b)
				in[name] = milcpu.Tensor{Shape: sh, Data: v}
			}
			res, err := parts[i].prog.Run(in, milcpu.Options{})
			if err != nil {
				return err
			}
			for name, o := range res {
				if name == "y" {
					y = o
					if i < len(parts)-1 {
						continue
					}
				}
				hwnpu.EncodeF16(p.Out[name], o.Data)
			}
		}
		return nil
	}
	aprog, f, err := milcpu.OpenPackage(man.Attention.Path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	ash := outShape(man.Attention.Path)
	in := aprog.Inputs()
	a := &decode.Attention{Chunk: man.Paged, Q: make([]byte, 2*count(in["q"])), O: make([]byte, 2*count(ash["o"])), M: make([]byte, 2*count(ash["m"])), L: make([]byte, 2*count(ash["l"]))}
	a.NewBuf = func(n int) (decode.Buf, error) {
		*allocated += n
		return cpuBuf(make([]byte, n)), nil
	}
	a.Run = func(k, v, mask decode.Buf) error {
		vals := map[string]milcpu.Tensor{}
		for name, b := range map[string][]byte{"q": a.Q, "k": k.Bytes(), "v": v.Bytes(), "mask": mask.Bytes()} {
			x := make([]float32, count(in[name]))
			if 2*len(x) != len(b) {
				return fmt.Errorf("attention input %s: %d bytes for %v", name, len(b), in[name])
			}
			hwnpu.DecodeF16(x, b)
			vals[name] = milcpu.Tensor{Shape: in[name], Data: x}
		}
		res, err := aprog.Run(vals, milcpu.Options{})
		if err != nil {
			return err
		}
		hwnpu.EncodeF16(a.O, res["o"].Data)
		hwnpu.EncodeF16(a.M, res["m"].Data)
		hwnpu.EncodeF16(a.L, res["l"].Data)
		return nil
	}
	if man.Mass != nil {
		mprog, f, err := milcpu.OpenPackage(man.Mass.Path)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { f.Close() })
		min := mprog.Inputs()
		ms := &decode.Mass{Q: make([]byte, 2*count(min["q"])), R: make([]byte, 2*count(min["r"])), W: make([]byte, 2*count(outShape(man.Mass.Path)["w"]))}
		ms.Run = func(k, mask decode.Buf) error {
			vals := map[string]milcpu.Tensor{}
			for name, b := range map[string][]byte{"q": ms.Q, "r": ms.R, "k": k.Bytes(), "mask": mask.Bytes()} {
				x := make([]float32, count(min[name]))
				hwnpu.DecodeF16(x, b)
				vals[name] = milcpu.Tensor{Shape: min[name], Data: x}
			}
			res, err := mprog.Run(vals, milcpu.Options{})
			if err != nil {
				return err
			}
			hwnpu.EncodeF16(ms.W, res["w"].Data)
			return nil
		}
		a.Mass = ms
	}
	p.Attn = a
	return p
}

// The model split at its two full attention layers (three parts, the host attending over its
// cache in chunks in between) gives the logits of the program without a cache, with the Gated
// DeltaNet's states and windows in the parts' own inputs; and so do two such programs sharing one
// sequence through a padded step, a rewind after a detour over a chunk boundary, a speculative
// step of which a beginning is kept, a saved sequence and one resumed from exported rows.
func TestPagedMatchesFull(t *testing.T) {
	cfg := tinyConfig(t)
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5}
	detour := []int32{7, 7, 0, 8, 2, 7, 1}
	n := len(tokens)
	emb := embed16(t, cfg, w, append(append([]int32(nil), tokens...), detour...))
	full := runCPU(t, cfg, w, llm.Options{Seq: n, Format: "fp16"}, emb[:n*cfg.HiddenSize])
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	worst := 0.0
	check := func(logits []float32, last int) {
		t.Helper()
		if last < n {
			worst = math.Max(worst, relErr(logits, full[last*cfg.VocabSize:(last+1)*cfg.VocabSize]))
		}
	}
	for _, c := range []struct{ q, cache, chunk int }{{1, 12, 4}, {3, 13, 4}, {4, 16, 5}} {
		t.Run(fmt.Sprintf("q%d cache%d chunk%d", c.q, c.cache, c.chunk), func(t *testing.T) {
			man := convertPaged(t, cfg, w, c.q, c.cache, c.chunk)
			if len(man.Parts) != 3 || man.Paged != c.chunk {
				t.Fatalf("%d parts, chunks of %d", len(man.Parts), man.Paged)
			}
			allocated := 0
			s, err := decode.New(cpuPaged(t, man, &allocated), cfg)
			if err != nil {
				t.Fatal(err)
			}
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
				check(logits, pos-1)
			}
		})
	}

	const cache, chunk = 16, 4
	allocated := 0
	short := cpuPaged(t, convertPaged(t, cfg, w, 1, cache, chunk), &allocated)
	long := cpuPaged(t, convertPaged(t, cfg, w, 4, cache, chunk), &allocated)
	for name := range long.In {
		if llm.CacheInput(name) {
			long.In[name] = short.In[name]
		}
	}
	newSession := func() *decode.Session {
		s, err := decode.New(short, cfg)
		if err != nil {
			t.Fatal(err)
		}
		if err := s.Add(long); err != nil {
			t.Fatal(err)
		}
		return s
	}
	step := func(s *decode.Session, rows ...int) {
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
	s := newSession()
	step(s, 0, 1, 2) // long, padded
	step(s, 3, 4)
	m := s.Mark()
	step(s, n, n+1, n+2, n+3) // a detour over the chunk boundary
	step(s, n+4)
	s.Rewind(m)
	logits, kept, err := s.Speculate([][]float32{row(5), row(6), row(n), row(n + 1)}, func(r func(i int) []float32) int {
		check(r(1), 6)
		return 2
	})
	if err != nil || kept != 2 || s.Pos() != 7 {
		t.Fatalf("speculate kept %d to %d: %v", kept, s.Pos(), err)
	}
	check(logits, 6)
	step(s, 7)
	saved := s.Save()
	rows, err := s.ExportRows(0, s.Pos())
	if err != nil {
		t.Fatal(err)
	}
	mark := s.Mark()
	s.Reset()
	step(s, n+5, n+6) // another sequence over the saved one's rows
	s.Rewind(saved)
	step(s, 8)
	step(s, 9, 10)
	r := newSession()
	if err := r.Resume(rows, mark); err != nil {
		t.Fatal(err)
	}
	step(r, 8, 9, 10)
	if worst > 2e-3 {
		t.Errorf("relative error %g against the program without a cache", worst)
	}
	t.Logf("worst relative error %.2g", worst)
}
