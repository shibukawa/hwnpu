// Command moeprobe writes synthetic experts of gpt-oss-20b's shape (hidden 2880, intermediate
// 2880, lut4) as Core AI programs, and times them on the Neural Engine: what a mixture of experts
// costs there, one program an expert or a layer's experts all in one (docs/design.md 16.229).
//
//	go run ./tools/coreai/moeprobe -n 32 -out DIR             32 experts, a program each
//	go run ./tools/coreai/moeprobe -n 256 -fns 32 -out DIR    8 programs of 32 functions
//	go run ./tools/coreai/moeprobe -dense 24 -out DIR         24 layers, each one program of all 32
//	                                                          experts weighed by an input
//	go run ./tools/coreai/moeprobe -dyn -out DIR              a linear given its lut8 table as an input
//	go run ./tools/coreai/moeprobe -k 4 PROGRAM...            load all at once, call them 1 and k at a time
//
// The dense layers are timed in turn with aicheck cycle (built as a binary of its own, whose Core
// AI cache is its own): on an M1 7 layers (2.61 GiB) run at their own 9.3 ms, 8 (2.98 GiB) and
// more are loaded again at every call.
package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

const d, inter = 2880, 2880

type synth struct{}

func (synth) Get(name string) ([]float32, []int, error) {
	e, kind, _ := strings.Cut(name, ".")
	n, _ := strconv.Atoi(strings.TrimPrefix(e, "e"))
	rng := rand.New(rand.NewSource(int64(n)*7919 + int64(map[string]int{"gate": 1, "up": 2, "down": 3, "gate_b": 4, "up_b": 5, "down_b": 6}[kind])))
	var shape []int
	switch kind {
	case "gate", "up":
		shape = []int{inter, d}
	case "down":
		shape = []int{d, inter}
	case "gate_b", "up_b":
		shape = []int{inter}
	case "down_b":
		shape = []int{d}
	default:
		return nil, nil, fmt.Errorf("no tensor %s", name)
	}
	size := 1
	for _, s := range shape {
		size *= s
	}
	v := make([]float32, size)
	for i := range v {
		v[i] = float32(rng.NormFloat64() * 0.02)
	}
	return v, shape, nil
}
func (s synth) Rows(name string, b, e int) ([]float32, error) {
	v, sh, err := s.Get(name)
	if err != nil {
		return nil, err
	}
	return v[b*sh[1] : e*sh[1]], nil
}
func (synth) Source() string { return "synthetic" }
func (synth) Close() error   { return nil }

// expert is GptOssExperts for one expert: (up + 1) * gate * sigmoid(1.702 gate), gate clamped
// above at 7 and up to [-7, 7], then down.
func expert(g *blocks.Graph, x *mlpackage.Var, e int) *mlpackage.Var {
	p := fmt.Sprintf("e%d.", e)
	gate := g.LinearBias(x, p+"gate", p+"gate_b", "", inter)
	up := g.LinearBias(x, p+"up", p+"up_b", "", inter)
	gate = g.Op("minimum", map[string]any{"x": gate, "y": g.B.ScalarF16(7)}, gate.Shape...)
	up = g.Op("clip", map[string]any{"x": up, "alpha": g.B.ScalarF16(-7), "beta": g.B.ScalarF16(7)}, up.Shape...)
	neg := g.Op("mul", map[string]any{"x": gate, "y": g.B.ScalarF16(-1.702)}, gate.Shape...)
	den := g.Op("add", map[string]any{"x": g.Op("exp", map[string]any{"x": neg}, gate.Shape...), "y": g.B.ScalarF16(1)}, gate.Shape...)
	glu := g.Op("real_div", map[string]any{"x": gate, "y": den}, gate.Shape...)
	up1 := g.Op("add", map[string]any{"x": up, "y": g.B.ScalarF16(1)}, up.Shape...)
	h := g.Op("mul", map[string]any{"x": up1, "y": glu}, gate.Shape...)
	return g.LinearBias(h, p+"down", p+"down_b", "y", d)
}

func main() {
	n := flag.Int("n", 32, "experts")
	seq := flag.Int("seq", 8, "rows per call")
	out := flag.String("out", "", "directory")
	format := flag.String("format", "lut4", "weight format")
	k := flag.Int("k", 0, "time the programs given as arguments k at a time instead of writing")
	rounds := flag.Int("rounds", 5, "rounds of -k")
	hold := flag.Duration("hold", 0, "with -k: load the programs, call each once and hold them this long")
	dyn := flag.Bool("dyn", false, "write dyn.aimodel and const.aimodel: a linear given its lut8 tables as inputs, and as constants")
	dense := flag.Int("dense", 0, "write N layers, each one program of all -n experts weighed by an input (groups of -per)")
	per := flag.Int("per", 4, "experts a group of -dense")
	fns := flag.Int("fns", 0, "write the experts as programs of this many functions each")
	flag.Parse()
	if *k > 0 {
		if err := timeGroups(flag.Args(), *k, *rounds, *hold); err != nil {
			log.Fatal(err)
		}
		return
	}
	if *out == "" {
		log.Fatal("-out")
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		log.Fatal(err)
	}
	if *dense > 0 {
		for l := 0; l < *dense; l++ {
			if err := writeDense(filepath.Join(*out, fmt.Sprintf("dense%02d.aimodel", l)), *seq, *n, *per, l); err != nil {
				log.Fatal(err)
			}
		}
		return
	}
	if *dyn {
		if err := writeDyn(*out, *seq, d, inter); err != nil {
			log.Fatal(err)
		}
		return
	}
	if *fns > 0 {
		for first := 0; first < *n; first += *fns {
			var list []llm.AIFunction
			for e := first; e < min(first+*fns, *n); e++ {
				list = append(list, llm.AIFunction{Name: fmt.Sprintf("e%d", e), Model: probeModel{seq: *seq}, Options: llm.Options{Format: *format, Group: 16, Params: map[string]string{"expert": strconv.Itoa(e)}}})
			}
			path := filepath.Join(*out, fmt.Sprintf("layer%03d.aimodel", first / *fns))
			recs, err := llm.WriteAIFunctions(path, synth{}, list)
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("%s %d functions %.1f MiB\n", path, len(list), float64(recs[0].WeightBytes)/(1<<20))
		}
		return
	}
	for e := 0; e < *n; e++ {
		path := filepath.Join(*out, fmt.Sprintf("expert%03d.aimodel", e))
		size, err := llm.WriteAI(path, nil, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
			g := &blocks.Graph{Name: "moeprobe", B: b, W: synth{}, Opts: llm.Options{Format: *format, Group: 16}}
			var ins, outs []mlpackage.Feature
			err := blocks.Run(func() {
				x := b.Input("x", mlpackage.Float16, 1, *seq, d)
				y := expert(g, x, e)
				ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
				outs = []mlpackage.Feature{{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}}
			})
			return ins, outs, err
		})
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("%s %.1f MiB\n", path, float64(size)/(1<<20))
	}
}

// probeModel builds expert Params["expert"] as an llm.Model, for llm.WriteAIFunctions.
type probeModel struct{ seq int }

func (probeModel) Hidden() int               { return d }
func (probeModel) Layers() int               { return 1 }
func (probeModel) Output() (int, bool)       { return d, false }
func (probeModel) Embedding() string         { return "" }
func (probeModel) Tensors() map[string][]int { return nil }
func (p probeModel) Build(b *mlpackage.Builder, w llm.Weights, o llm.Options) ([]mlpackage.Feature, mlpackage.Feature, error) {
	e, _ := strconv.Atoi(o.Params["expert"])
	g := &blocks.Graph{Name: "moeprobe", B: b, W: w, Opts: o}
	var ins []mlpackage.Feature
	var out mlpackage.Feature
	err := blocks.Run(func() {
		x := b.Input("x", mlpackage.Float16, 1, p.seq, d)
		y := expert(g, x, e)
		ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
		out = mlpackage.Feature{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}
	})
	return ins, out, err
}
