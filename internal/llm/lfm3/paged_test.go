package lfm3

import (
	"fmt"
	"io"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// pagedConfig is tinyConfig with two full attention layers, so that the model splits into three
// parts with the host's attention twice in between, and a sliding layer and convolutions that
// keep their state in the program.
func pagedConfig() *Config {
	c := tinyConfig()
	c.NumLayers = 5
	c.LayerTypes = []string{"conv", "full_attention", "sliding_attention", "full_attention", "conv"}
	c.MaxPositions = 64
	return c
}

// convertPaged converts the model split at its attention with hwnpu-convert's code path.
func convertPaged(t *testing.T, cfg *Config, w llm.Weights, q, cache, chunk int, attn string) *llm.Manifest {
	t.Helper()
	opts := llm.Options{Seq: q, Cache: cache, Paged: chunk, Format: "fp16", Params: map[string]string{"post-scale": "1", "attn": attn}}
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
// internal/llm/chain does on a device: x of each part is y of the one before, an input named like
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

// The two-level attention (decode.Sparse) runs through the programs hwnpu-convert writes, the
// mass program among them, and past the length where it pays the logits stay near those of the
// whole cache: buckets of 4 over chunks of 8, three chunks read exactly. A manifest converted
// before the mass program gets it from AddMass.
func TestPagedSparse(t *testing.T) {
	cfg := pagedConfig()
	cfg.MaxPositions = 256
	w := llm.NewSynthetic(cfg)
	const n, cache, chunk = 200, 208, 8
	tokens := make([]int32, n)
	for i := range tokens {
		tokens[i] = int32((i*7 + i*i) % cfg.VocabSize)
	}
	emb, err := llm.Embed(w, cfg, tokens)
	if err != nil {
		t.Fatal(err)
	}
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	man := convertPaged(t, cfg, w, 1, cache, chunk, "grouped")
	if man.Mass == nil {
		t.Fatal("the conversion wrote no mass program")
	}
	allocated := 0
	dense, err := decode.New(cpuPaged(t, man, &allocated), cfg)
	if err != nil {
		t.Fatal(err)
	}
	sparse, err := decode.New(cpuPaged(t, man, &allocated), cfg)
	if err != nil {
		t.Fatal(err)
	}
	sp := decode.Sparse{Bucket: 4, Sinks: 1, Window: 4, Exact: 3, Refresh: 1}
	if err := sparse.SetSparse(sp); err != nil {
		t.Fatal(err)
	}
	worst := 0.0
	for pos := range n {
		want, err := dense.Step([][]float32{row(pos)})
		if err != nil {
			t.Fatal(err)
		}
		got, err := sparse.Step([][]float32{row(pos)})
		if err != nil {
			t.Fatal(err)
		}
		if sparse.AttentionCalls() == 0 && !slices.Equal(got, want) {
			t.Fatalf("at %d, before the two-level attention, the logits differ", pos)
		}
		worst = math.Max(worst, relErr(got, want))
	}
	if sparse.AttentionCalls() == 0 {
		t.Fatal("the two-level attention never ran")
	}
	if worst > 0.05 || math.IsNaN(worst) {
		t.Errorf("relative error %g against the whole cache", worst)
	}
	t.Logf("worst relative error %.2g, %d calls of the two-level attention", worst, sparse.AttentionCalls())

	// a manifest without the mass program gets it back
	path := strings.TrimSuffix(man.Attention.Path, ".attn.mlpackage") + ".parts.json"
	old := *man
	old.Mass = nil
	if err := llm.WriteManifest(path, &old); err != nil {
		t.Fatal(err)
	}
	if err := llm.AddMass(path, "grouped", 0); err != nil {
		t.Fatal(err)
	}
	back, err := llm.ReadManifest(path)
	if err != nil || back.Mass == nil || back.Mass.Path != man.Mass.Path {
		t.Fatalf("AddMass: %v, %+v", err, back.Mass)
	}
}

// The model split at its attention, stepped through a sequence with the host attending over its
// own cache in chunks, gives the logits of the program without a cache at every position: steps
// of one token and of several, a step over a chunk's boundary, chunks the cache's length is not a
// multiple of, both forms of the attention, and the chunks allocated only as the sequence
// reaches them.
func TestPagedMatchesFull(t *testing.T) {
	cfg := pagedConfig()
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5}
	n := len(tokens)
	emb, err := llm.Embed(w, cfg, tokens)
	if err != nil {
		t.Fatal(err)
	}
	full := runCPU(t, cfg, w, Options{Seq: n, Format: "fp16"}, emb)
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	for _, c := range []struct {
		q, cache, chunk int
		attn            string
	}{{1, 12, 4, "grouped"}, {3, 13, 4, "grouped"}, {4, 12, 4, "grouped"}, {2, 11, 3, "tile"}, {3, 64, 5, "tile"}} {
		t.Run(fmt.Sprintf("q%d cache%d chunk%d %s", c.q, c.cache, c.chunk, c.attn), func(t *testing.T) {
			man := convertPaged(t, cfg, w, c.q, c.cache, c.chunk, c.attn)
			if len(man.Parts) != 3 || man.Paged != c.chunk {
				t.Fatalf("%d parts, chunks of %d", len(man.Parts), man.Paged)
			}
			allocated := 0
			s, err := decode.New(cpuPaged(t, man, &allocated), cfg)
			if err != nil {
				t.Fatal(err)
			}
			if s.Cache() != c.cache {
				t.Fatalf("cache %d, want %d", s.Cache(), c.cache)
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
				worst = math.Max(worst, relErr(logits, full[(pos-1)*cfg.VocabSize:pos*cfg.VocabSize]))
				// 2 full attention layers × keys and values, a chunk each per chunk reached
				chunkBytes := 2 * cfg.NumKVHeads * c.chunk * cfg.HeadDim
				if reached := (min(pos-v+c.q, c.cache) + c.chunk - 1) / c.chunk; int(s.HeldBytes()) != 4*reached*chunkBytes {
					t.Errorf("at %d the session holds %d bytes, want %d chunks of %d", pos, s.HeldBytes(), 4*reached, chunkBytes)
				}
			}
			if worst > 2e-3 {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			t.Logf("worst relative error %.2g, %d bytes allocated", worst, allocated)
		})
	}
}

// Two programs split at their attention (a long step for prompts, a short one to generate)
// continue one sequence with their rings and windows shared and the long caches the session's;
// a mark and a saved sequence come back after a detour, and the rows a session exports start
// another (the prompt store's way), all with the logits of the program without a cache.
func TestPagedShareRewindResume(t *testing.T) {
	cfg := pagedConfig()
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	detour := []int32{7, 7, 0, 8, 2, 7, 1}
	n := len(tokens)
	emb, err := llm.Embed(w, cfg, append(append([]int32(nil), tokens...), detour...))
	if err != nil {
		t.Fatal(err)
	}
	full := runCPU(t, cfg, w, Options{Seq: n, Format: "fp16"}, emb[:n*cfg.HiddenSize])
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	const cache, chunk = 16, 4
	allocated := 0
	short := cpuPaged(t, convertPaged(t, cfg, w, 1, cache, chunk, "grouped"), &allocated)
	long := cpuPaged(t, convertPaged(t, cfg, w, 4, cache, chunk, "grouped"), &allocated)
	for name := range long.In {
		if strings.HasPrefix(name, "kv_") || strings.HasPrefix(name, "win_") {
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
	s := newSession()
	worst := 0.0
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
		if last := rows[len(rows)-1]; last < n {
			worst = math.Max(worst, relErr(logits, full[last*cfg.VocabSize:(last+1)*cfg.VocabSize]))
		}
	}
	step(s, 0, 1, 2, 3) // long
	step(s, 4)          // short
	m := s.Mark()
	step(s, n, n+1, n+2, n+3) // a detour over the chunk boundary
	step(s, n+4)
	s.Rewind(m)
	step(s, 5, 6, 7)
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
	step(s, 9)
	// a new session from the exported rows and the mark
	r := newSession()
	if err := r.Resume(rows, mark); err != nil {
		t.Fatal(err)
	}
	step(r, 8, 9)
	if worst > 2e-3 {
		t.Errorf("relative error %g against the program without a cache", worst)
	}
	if err := r.Limit(9); err == nil {
		t.Error("a limit under the sequence was taken")
	}
	if err := r.Limit(12); err != nil || r.Cache() != 12 {
		t.Errorf("limit 12: %v, cache %d", err, r.Cache())
	}
	r.Close()
	if r.HeldBytes() != 0 {
		t.Error("a closed session holds chunks")
	}
	t.Logf("worst relative error %.2g", worst)
}

// The parts are cut at the full attention layers, the part before one outputs what the host's
// attention takes, the one after takes its output, and neither has a cache input for it.
func TestPagedParts(t *testing.T) {
	cfg := pagedConfig()
	man := convertPaged(t, cfg, llm.NewSynthetic(cfg), 2, 32, 8, "grouped")
	var starts []int
	for _, p := range man.Parts {
		starts = append(starts, p.Start)
	}
	if !slices.Equal(starts, []int{0, 1, 3}) {
		t.Fatalf("parts start at %v, want [0 1 3]", starts)
	}
	names := func(fs []llm.FeatureRecord) []string {
		var out []string
		for _, f := range fs {
			out = append(out, f.Name)
		}
		return out
	}
	first, second := man.Parts[0], man.Parts[1]
	for _, want := range []string{"attn_q_1", "kv_k_1_new", "kv_v_1_new", "attn_gate_1"} {
		if !slices.Contains(names(first.State), want) {
			t.Errorf("part 1 does not output %s: %v", want, names(first.State))
		}
	}
	for _, want := range []string{"attn_o_1", "attn_gate_1"} {
		if !slices.Contains(names(second.Inputs), want) {
			t.Errorf("part 2 does not take %s: %v", want, names(second.Inputs))
		}
	}
	for _, p := range man.Parts {
		for _, in := range p.Inputs {
			if strings.HasPrefix(in.Name, "kv_k_1") || strings.HasPrefix(in.Name, "kv_k_3") || strings.HasPrefix(in.Name, "mask_full") {
				t.Errorf("%s takes %s", filepath.Base(p.Path), in.Name)
			}
		}
	}
	if man.Attention == nil || !slices.Equal(names(man.Attention.Inputs), []string{"q", "k", "v", "mask"}) {
		t.Errorf("attention program %+v", man.Attention)
	}
}

// A program that reads prompts with the output layer for the last row alone (LastRow) continues
// the sequence of the one that generates and gives the logits after the last token of each step,
// a step shorter than the program's too; it takes no drafts to check.
func TestPagedLastRow(t *testing.T) {
	cfg := pagedConfig()
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5}
	n := len(tokens)
	emb, err := llm.Embed(w, cfg, tokens)
	if err != nil {
		t.Fatal(err)
	}
	full := runCPU(t, cfg, w, Options{Seq: n, Format: "fp16"}, emb)
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	const cache, chunk = 16, 4
	allocated := 0
	short := cpuPaged(t, convertPaged(t, cfg, w, 1, cache, chunk, "grouped"), &allocated)
	opts := llm.Options{Seq: 4, Cache: cache, Paged: chunk, LastRow: true, Format: "fp16", Params: map[string]string{"post-scale": "1", "attn": "grouped"}}
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
