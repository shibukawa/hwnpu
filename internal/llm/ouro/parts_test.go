package ouro

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/llm/decode"
	"github.com/shibukawa/bonsaiai/internal/milcpu"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
)

// writeParts converts the model in n parts, as bonsaiai-dev convert does, and returns the manifest.
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
// earlier part's output takes that output (none in Ouro, whose parts pass x alone), and every
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
				bonsaiai.DecodeF16(v, b)
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
				bonsaiai.EncodeF16(p.Out[name], o.Data)
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
