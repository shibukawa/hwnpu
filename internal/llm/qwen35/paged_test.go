package qwen35

import (
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"
	"testing"
	"unsafe"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/llm/decode"
	"github.com/shibukawa/bonsaiai/internal/milcpu"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
)

// convertPaged converts the model split at its attention with bonsaiai-dev convert's code path.
func convertPaged(t *testing.T, cfg *Config, w llm.Weights, q, cache, chunk int) *llm.Manifest {
	t.Helper()
	return convertPagedModel(t, cfg, w, q, cache, chunk)
}

// convertPagedModel is convertPaged for the model or its MTP layer.
func convertPagedModel(t *testing.T, m llm.Model, w llm.Weights, q, cache, chunk int) *llm.Manifest {
	t.Helper()
	a := Arch
	if _, ok := m.(*MTP); ok {
		a = MTPArch
	}
	opts := llm.Options{Seq: q, Cache: cache, Paged: chunk, Format: "fp16"}
	path, err := llm.Convert(a, m, w, filepath.Join(t.TempDir(), "m.mlpackage"), "none", 0, opts, io.Discard)
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
	// what a stage hands on (decode.Program.Carry): the hidden state, kept between the stages in
	// handed (a stage begins from it), the attention's output and the outputs a later part takes
	handed := make([][]float32, len(stages))
	for i := range handed {
		handed[i] = make([]float32, count(p.InShape["x"]))
	}
	takes := func(s int) map[string]bool {
		names := map[string]bool{}
		for _, i := range stages[s] {
			for name := range parts[i].prog.Inputs() {
				names[name] = true
			}
		}
		return names
	}
	p.StageIn = func(s int) []string {
		var names []string
		for name := range takes(s) {
			if p.In[name] != nil {
				names = append(names, name)
			}
		}
		return names
	}
	p.Carry = func(s int) [][]byte {
		out := [][]byte{unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(handed[s]))), 4*len(handed[s]))}
		for _, f := range []string{"attn_o_%d", "attn_ot_%d"} {
			if b := p.In[fmt.Sprintf(f, p.Attend[s])]; b != nil {
				out = append(out, b)
			}
		}
		for later := s + 1; later < len(stages); later++ {
			for name := range takes(later) {
				made := false
				for _, st := range stages[:s+1] {
					for _, i := range st {
						made = made || slices.Contains(parts[i].outs, name)
					}
				}
				if made && !slices.ContainsFunc(out, func(b []byte) bool { return unsafe.SliceData(b) == unsafe.SliceData(p.Out[name]) }) {
					out = append(out, p.Out[name])
				}
			}
		}
		return out
	}
	p.RunPart = func(s int) error {
		if s > 0 {
			y = milcpu.Tensor{Shape: p.InShape["x"], Data: slices.Clone(handed[s-1])}
		}
		defer func() {
			if s < len(stages)-1 {
				copy(handed[s], y.Data)
			}
		}()
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
				bonsaiai.DecodeF16(v, b)
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
				bonsaiai.EncodeF16(p.Out[name], o.Data)
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
			bonsaiai.DecodeF16(x, b)
			vals[name] = milcpu.Tensor{Shape: in[name], Data: x}
		}
		res, err := aprog.Run(vals, milcpu.Options{})
		if err != nil {
			return err
		}
		bonsaiai.EncodeF16(a.O, res["o"].Data)
		bonsaiai.EncodeF16(a.M, res["m"].Data)
		bonsaiai.EncodeF16(a.L, res["l"].Data)
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
				bonsaiai.DecodeF16(x, b)
				vals[name] = milcpu.Tensor{Shape: min[name], Data: x}
			}
			res, err := mprog.Run(vals, milcpu.Options{})
			if err != nil {
				return err
			}
			bonsaiai.EncodeF16(ms.W, res["w"].Data)
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

// A program that reads prompts with the output layer for the last row alone (LastRow) continues
// the sequence of one of 1 token per call (the Gated DeltaNet's states and windows shared) and
// gives the logits after the last token of each step, a step shorter than its 4 tokens too; its
// last part outputs one row of logits, it checks no drafts, and it needs a KV cache.
func TestPagedLastRow(t *testing.T) {
	cfg := tinyConfig(t)
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5}
	n := len(tokens)
	emb := embed16(t, cfg, w, tokens)
	full := runCPU(t, cfg, w, llm.Options{Seq: n, Format: "fp16"}, emb)
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	const cache, chunk = 16, 4
	allocated := 0
	short := cpuPaged(t, convertPaged(t, cfg, w, 1, cache, chunk), &allocated)
	opts := llm.Options{Seq: 4, Cache: cache, Paged: chunk, LastRow: true, Format: "fp16"}
	path, err := llm.Convert(Arch, cfg, w, filepath.Join(t.TempDir(), "m.mlpackage"), "none", 0, opts, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	man, err := llm.ReadManifest(path)
	if err != nil {
		t.Fatal(err)
	}
	long := cpuPaged(t, man, &allocated)
	if sh := long.OutShape["y"]; !slices.Equal(sh, []int{1, 1, cfg.VocabSize}) {
		t.Fatalf("y %v, want (1, 1, %d)", sh, cfg.VocabSize)
	}
	for name := range long.In {
		if llm.CacheInput(name) {
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
	worst, pos := 0.0, 0
	for _, v := range []int{4, 3, 1, 2, 1} {
		var e [][]float32
		for i := range v {
			e = append(e, row(pos+i))
		}
		logits, err := s.Step(e)
		if err != nil {
			t.Fatal(err)
		}
		pos += v
		worst = math.Max(worst, relErr(logits, full[(pos-1)*cfg.VocabSize:pos*cfg.VocabSize]))
	}
	if worst > 2e-3 {
		t.Errorf("relative error %g against the program without a cache", worst)
	}
	if _, _, err := s.Speculate([][]float32{row(0), row(1), row(2)}, func(func(int) []float32) int { return 1 }); err == nil {
		t.Error("the last row's program checked drafts")
	}
	opts.Cache, opts.Paged = 0, 0
	if _, err := llm.Convert(Arch, cfg, w, filepath.Join(t.TempDir(), "n.mlpackage"), "none", 0, opts, io.Discard); err == nil {
		t.Error("LastRow without a cache was built")
	}
	t.Logf("worst relative error %.2g", worst)
}

// Several steps run a stage at a time (decode.Session.Steps: every step through the parts before
// the first attention layer, then every step through the next) give the logits, the marks and the
// states of the steps run in turn, bit for bit: with a padded last step, a mark inside the wave,
// and going on from the wave's end and from its mark. And so they do through the program's own
// run of a stage for all the steps (decode.Program.RunMany).
func TestPagedWave(t *testing.T) {
	for _, many := range []bool{false, true} {
		t.Run(fmt.Sprintf("RunMany %v", many), func(t *testing.T) { testPagedWave(t, many) })
	}
}

func testPagedWave(t *testing.T, many bool) {
	cfg := tinyConfig(t)
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3}
	emb := embed16(t, cfg, w, tokens)
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	const q, cache, chunk = 4, 24, 8
	man := convertPaged(t, cfg, w, q, cache, chunk)
	allocated := 0
	var sessions [2]*decode.Session
	var ran []int // the stages the wave's session ran, in order
	direct := 0
	for i := range sessions {
		p := cpuPaged(t, man, &allocated)
		if i == 1 {
			run := p.RunPart
			p.RunPart = func(s int) error {
				ran = append(ran, s)
				return run(s)
			}
			if many {
				p.RunPart = func(s int) error {
					direct++ // a step outside a wave
					return run(s)
				}
				p.RunMany = func(s, n int, step func(c int) error, done func(c int, inputs []string, last bool) error) error {
					for c := range n {
						if err := step(c); err != nil {
							return err
						}
						ran = append(ran, s)
						if err := run(s); err != nil {
							return err
						}
						if err := done(c, p.StageIn(s), true); err != nil {
							return err
						}
					}
					return nil
				}
			}
		}
		s, err := decode.New(p, cfg)
		if err != nil {
			t.Fatal(err)
		}
		sessions[i] = s
	}
	one, wave := sessions[0], sessions[1]
	// 4 + 4 + 4 + 3 tokens (the last step padded), a mark after the second; then 3 more
	var steps [][][]float32
	for pos := 0; pos < 15; pos += q {
		var step [][]float32
		for i := pos; i < min(pos+q, 15); i++ {
			step = append(step, row(i))
		}
		steps = append(steps, step)
	}
	rest := [][]float32{row(15), row(16), row(17)}
	var want []float32
	var wantMark *decode.Mark
	var wantAt []float32
	for i, step := range steps {
		var err error
		if want, err = one.Step(step); err != nil {
			t.Fatal(err)
		}
		if i == 1 {
			wantMark, wantAt = one.Mark(), want
		}
	}
	asked := 0
	got, marks, err := wave.Steps(steps, []int{1}, func() error { asked++; return nil })
	if err != nil {
		t.Fatal(err)
	}
	if asked < len(steps)*len(man.Parts) {
		t.Errorf("asked %d times whether to go on, in a wave of %d steps of %d stages", asked, len(steps), len(man.Parts))
	}
	same := func(what string, a, b []float32) {
		t.Helper()
		if len(a) != len(b) || len(a) == 0 {
			t.Fatalf("%s: %d and %d values", what, len(a), len(b))
		}
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("%s: value %d is %g a stage at a time, %g a step at a time", what, i, a[i], b[i])
			}
		}
	}
	if want := []int{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2}; !slices.Equal(ran, want) || direct != 0 {
		t.Fatalf("the stages ran in the order %v, want %v (%d by RunPart of a program with RunMany)", ran, want, direct)
	}
	same("the logits after the wave", got, want)
	if wave.Pos() != one.Pos() || wave.Pos() != 15 {
		t.Fatalf("at %d and %d", wave.Pos(), one.Pos())
	}
	if len(marks) != 1 || marks[0].Mark.Pos() != wantMark.Pos() {
		t.Fatalf("%d marks", len(marks))
	}
	same("the logits at the mark", marks[0].Logits, wantAt)
	if len(marks[0].Mark.Bufs()) != len(wantMark.Bufs()) || len(wantMark.Bufs()) == 0 {
		t.Fatalf("the mark holds %d states, want %d", len(marks[0].Mark.Bufs()), len(wantMark.Bufs()))
	}
	for name, b := range wantMark.Bufs() {
		if !slices.Equal(marks[0].Mark.Bufs()[name], b) {
			t.Errorf("the mark's %s differs", name)
		}
	}
	next := func(s *decode.Session) []float32 {
		t.Helper()
		l, err := s.Step(rest)
		if err != nil {
			t.Fatal(err)
		}
		return l
	}
	same("the step after the wave", next(wave), next(one))
	// back to the marks, each session to its own, and on from there
	wave.Rewind(marks[0].Mark)
	one.Rewind(wantMark)
	again := [][]float32{row(8), row(9), row(10), row(11)}
	a, err := wave.Step(again)
	if err != nil {
		t.Fatal(err)
	}
	b, err := one.Step(again)
	if err != nil {
		t.Fatal(err)
	}
	same("the step after the mark", a, b)

	// a wave that would pass the cache's end stops before the step that does not fit
	wave.Reset()
	var over [][][]float32
	for range cache/q + 1 {
		over = append(over, steps[0])
	}
	if _, _, err := wave.Steps(over, nil, nil); err == nil {
		t.Error("no error past the cache's end")
	}
}
