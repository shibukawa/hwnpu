// hwnpu-dev irdiff checks what a Core ML conversion becomes on Intel's NPU, without one: it
// lowers each .mlpackage to OpenVINO IR as -backend intel does (internal/ovlower), runs the MIL
// program on the fp32 interpreter (internal/milcpu) and the IR on OpenVINO's CPU plugin
// (internal/ovcpu, HWNPU_OV_PYTHON) with the same random inputs, and prints each output's
// relative RMS error. A package whose operations ovlower cannot lower fails with the operation.
//
//	hwnpu-dev irdiff [-lower-only] [-keep DIR] [-in NAME=LO:HI] A.mlpackage B.mlpackage ...
//
// It is how a model that has only a Core ML path is checked before writing its Intel path: the
// model's own preparation writes the packages (on any machine; the converters are Go), and irdiff
// says which operations are missing and whether the lowered program computes the same.
package irdiff

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/milspec"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/mlspec"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
	"github.com/shibukawa/hwnpu/internal/ovlower"
)

type ranges map[string][2]float32

func (r ranges) String() string { return fmt.Sprint(map[string][2]float32(r)) }
func (r ranges) Set(s string) error {
	name, v, ok := strings.Cut(s, "=")
	lo, hi, ok2 := strings.Cut(v, ":")
	if !ok || !ok2 {
		return fmt.Errorf("want NAME=LO:HI, got %q", s)
	}
	a, err := strconv.ParseFloat(lo, 32)
	if err != nil {
		return err
	}
	b, err := strconv.ParseFloat(hi, 32)
	if err != nil {
		return err
	}
	r[name] = [2]float32{float32(a), float32(b)}
	return nil
}

func Main() {
	lowerOnly := flag.Bool("lower-only", false, "only lower to IR (no runs): checks the operations")
	keep := flag.String("keep", "", "write the IR into this directory and keep it (default: a temporary one)")
	tol := flag.Float64("tol", 0.01, "largest relative RMS error of an output")
	prec := flag.String("precision", "f32", "OpenVINO CPU's inference precision: f32 or f16")
	fnName := flag.String("function", "", "the function of a package with several (default: main, else each)")
	in := ranges{}
	flag.Var(in, "in", "the range of an input's random values, NAME=LO:HI (default -1:1); repeatable")
	flag.Parse()
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: hwnpu-dev irdiff [flags] A.mlpackage ...")
		os.Exit(2)
	}
	python := ovcpu.Python()
	if python == "" && !*lowerOnly {
		fmt.Fprintln(os.Stderr, "irdiff: HWNPU_OV_PYTHON is not set (a Python with openvino and numpy); -lower-only only lowers")
		os.Exit(2)
	}
	ok := true
	for _, pkg := range flag.Args() {
		if err := diff(pkg, *fnName, *keep, python, *prec, *lowerOnly, *tol, in); err != nil {
			fmt.Printf("%s: FAIL: %v\n", filepath.Base(pkg), err)
			ok = false
		}
	}
	if !ok {
		os.Exit(1)
	}
}

func diff(pkg, fnName, keep, python, prec string, lowerOnly bool, tol float64, in ranges) error {
	raw, err := os.ReadFile(filepath.Join(pkg, "Data", "com.apple.CoreML", "model.mlmodel"))
	if err != nil {
		return err
	}
	var m mlspec.Model
	if err := proto.Unmarshal(raw, &m); err != nil {
		return err
	}
	prog := m.GetMlProgram()
	if prog == nil {
		return fmt.Errorf("not an ML Program")
	}
	weights, err := os.Open(filepath.Join(pkg, "Data", "com.apple.CoreML", "weights", "weight.bin"))
	if err != nil {
		return err
	}
	defer weights.Close()
	var names []string
	switch {
	case fnName != "":
		names = []string{fnName}
	case prog.GetFunctions()["main"] != nil:
		names = []string{"main"}
	default:
		for n := range prog.GetFunctions() {
			names = append(names, n)
		}
		sort.Strings(names)
	}
	dir := keep
	if dir == "" {
		if dir, err = os.MkdirTemp("", "irdiff"); err != nil {
			return err
		}
		defer os.RemoveAll(dir)
	} else if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	var fails []string
	for _, name := range names {
		fn := prog.GetFunctions()[name]
		if fn == nil {
			return fmt.Errorf("no function %s", name)
		}
		label := filepath.Base(pkg)
		if len(names) > 1 || name != "main" {
			label += ":" + name
		}
		stem := strings.TrimSuffix(filepath.Base(pkg), ".mlpackage")
		if name != "main" {
			stem += "." + name
		}
		xml := filepath.Join(dir, stem+".xml")
		t0 := time.Now()
		size, err := ovlower.WriteFiles(xml, stem, fn, weights, nil)
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: lowering: %v", label, err))
			continue
		}
		fmt.Printf("%s: lowered %d ops to IR (%.1f MB) in %.1f s: %s\n", label, opCount(fn), float64(size)/1e6, time.Since(t0).Seconds(), opSummary(fn))
		if lowerOnly {
			continue
		}
		cpu, err := milcpu.LoadFunction(prog, name, weights)
		if err != nil {
			return err
		}
		rng := rand.New(rand.NewSource(1))
		feed := map[string]milcpu.Tensor{}
		ovFeed := map[string][]float32{}
		var inNames []string
		for n := range cpu.Inputs() {
			inNames = append(inNames, n)
		}
		sort.Strings(inNames)
		for _, n := range inNames {
			shape := cpu.Inputs()[n]
			r, ok := in[n]
			if !ok {
				r = [2]float32{-1, 1}
			}
			v := make([]float32, numel(shape))
			for i := range v {
				v[i] = mlpackage.Float16Value(mlpackage.Float16Bits(r[0] + (r[1]-r[0])*rng.Float32()))
			}
			feed[n] = milcpu.Tensor{Shape: shape, Data: v}
			ovFeed[n] = v
		}
		t0 = time.Now()
		want, err := cpu.Run(feed, milcpu.Options{})
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: milcpu: %v", label, err))
			continue
		}
		tc := time.Since(t0)
		t0 = time.Now()
		got, err := ovcpu.Run(python, xml, ovFeed, ovcpu.Options{Precision: prec})
		if err != nil {
			fails = append(fails, fmt.Sprintf("%s: OpenVINO: %v", label, err))
			continue
		}
		to := time.Since(t0)
		var outNames []string
		for n := range want {
			outNames = append(outNames, n)
		}
		sort.Strings(outNames)
		for _, n := range outNames {
			w := want[n].Data
			g, ok := got[n]
			if !ok {
				fails = append(fails, fmt.Sprintf("%s: OpenVINO has no output %s (has %v)", label, n, keys(got)))
				continue
			}
			if len(g) != len(w) {
				fails = append(fails, fmt.Sprintf("%s: output %s: %d values, want %d", label, n, len(g), len(w)))
				continue
			}
			e := relRMS(g, w)
			mark := "ok"
			if !(e <= tol) {
				mark = "FAIL"
				fails = append(fails, fmt.Sprintf("%s: output %s: relative RMS error %.3g", label, n, e))
			}
			fmt.Printf("  %-24s %v  rel RMS %.3g  max |want| %.3g  %s\n", n, want[n].Shape, e, maxAbs(w), mark)
		}
		fmt.Printf("  milcpu %.1f s, OpenVINO CPU %.1f s\n", tc.Seconds(), to.Seconds())
	}
	if len(fails) > 0 {
		return fmt.Errorf("%s", strings.Join(fails, "; "))
	}
	return nil
}

func numel(s []int) int {
	n := 1
	for _, d := range s {
		n *= d
	}
	return n
}

func keys(m map[string][]float32) []string {
	var k []string
	for n := range m {
		k = append(k, n)
	}
	sort.Strings(k)
	return k
}

func relRMS(got, want []float32) float64 {
	var num, den float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		num += d * d
		den += float64(want[i]) * float64(want[i])
	}
	if den == 0 {
		return math.Sqrt(num / float64(max(1, len(want))))
	}
	return math.Sqrt(num / den)
}

func maxAbs(v []float32) float64 {
	m := 0.0
	for _, x := range v {
		m = math.Max(m, math.Abs(float64(x)))
	}
	return m
}

func ops(fn *milspec.Function) []*milspec.Operation {
	return fn.GetBlockSpecializations()[fn.GetOpset()].GetOperations()
}

func opCount(fn *milspec.Function) int { return len(ops(fn)) }

func opSummary(fn *milspec.Function) string {
	n := map[string]int{}
	for _, op := range ops(fn) {
		if t := op.GetType(); t != "const" {
			n[t]++
		}
	}
	var s []string
	for t, c := range n {
		s = append(s, fmt.Sprintf("%s %d", t, c))
	}
	sort.Strings(s)
	return strings.Join(s, ", ")
}
