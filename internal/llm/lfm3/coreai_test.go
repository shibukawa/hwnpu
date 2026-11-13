package lfm3

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/coreai/ailower"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// A part written for Core AI (llm.WritePart with BackendCoreAI, from the weights without a Core
// ML package) is the program the Core AI backend lowers from the part's Core ML package, byte for
// byte: Core AI's cache, keyed by the bytecode's hash, serves both. lut4 takes the palettized
// weights' path too.
func TestCoreAIPartMatchesPackage(t *testing.T) {
	cfg := tinyConfig()
	w := llm.NewSynthetic(cfg)
	for _, format := range []string{"fp16", "lut4"} {
		dir := t.TempDir()
		opts := llm.Options{Seq: 8, Format: format}
		pkg := filepath.Join(dir, "m.mlpackage")
		opts.Backend = llm.BackendCoreML
		if _, err := llm.WritePart(pkg, cfg, w, opts, nil); err != nil {
			t.Fatal(err)
		}
		want, err := ailower.LowerPackage(pkg)
		if err != nil {
			t.Fatal(err)
		}
		asset := filepath.Join(dir, "m.aimodel")
		opts.Backend = llm.BackendCoreAI
		rec, err := llm.WritePart(asset, cfg, w, opts, func(p *mlpackage.Package) { p.Description = "tiny" })
		if err != nil {
			t.Fatal(err)
		}
		got, err := os.ReadFile(filepath.Join(asset, "main.mlirb"))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want.Bytecode) {
			t.Errorf("%s: the bytecode (%d bytes) is not the package's lowered (%d bytes)", format, len(got), len(want.Bytecode))
		}
		if rec.WeightBytes != int64(len(got)) {
			t.Errorf("%s: WeightBytes %d, bytecode %d", format, rec.WeightBytes, len(got))
		}
		meta, err := llm.ReadMeta(&llm.Manifest{Parts: []llm.PartRecord{{Path: asset}}})
		if err != nil || meta[llm.MetaDescription] != "tiny" {
			t.Errorf("%s: metadata %v (%v)", format, meta, err)
		}
		ins, outs, err := ailower.ReadIO(asset)
		if err != nil || len(ins) != len(want.Inputs) || len(outs) != len(want.Outputs) {
			t.Errorf("%s: inputs %v outputs %v (%v), want %v %v", format, ins, outs, err, want.Inputs, want.Outputs)
		}
	}
}

// Two functions of the same weights (WriteAIFunctions: here two sequence lengths, in Qwen-Image a
// DiT's text pass and its image blocks) keep them once: the program is the two programs apart
// less one copy of the weights, and each function's inputs and outputs are its own.
func TestCoreAIFunctionsShareWeights(t *testing.T) {
	cfg := tinyConfig()
	w := llm.NewSynthetic(cfg)
	dir := t.TempDir()
	opts := llm.Options{Seq: 8, Format: "lut4", Backend: llm.BackendCoreAI}
	long := opts
	long.Seq = 16
	size := func(asset string) int64 {
		st, err := os.Stat(filepath.Join(asset, "main.mlirb"))
		if err != nil {
			t.Fatal(err)
		}
		return st.Size()
	}
	var apart int64
	for i, o := range []llm.Options{opts, long} {
		asset := filepath.Join(dir, fmt.Sprintf("apart%d.aimodel", i))
		if _, err := llm.WritePart(asset, cfg, w, o, nil); err != nil {
			t.Fatal(err)
		}
		apart += size(asset)
	}
	pkg := filepath.Join(dir, "m.mlpackage")
	coreml := opts
	coreml.Backend = llm.BackendCoreML
	if _, err := llm.WritePart(pkg, cfg, w, coreml, nil); err != nil {
		t.Fatal(err)
	}
	wb, err := os.Stat(filepath.Join(pkg, "Data", "com.apple.CoreML", "weights", "weight.bin"))
	if err != nil {
		t.Fatal(err)
	}
	two := filepath.Join(dir, "two.aimodel")
	recs, err := llm.WriteAIFunctions(two, w, []llm.AIFunction{{Name: "f8", Model: cfg, Options: opts}, {Name: "f16", Model: cfg, Options: long}})
	if err != nil {
		t.Fatal(err)
	}
	if saved := apart - size(two); saved < wb.Size()*9/10 {
		t.Errorf("two functions take %d bytes, apart %d: %d saved of %d bytes of weights", size(two), apart, saved, wb.Size())
	}
	if len(recs) != 2 || recs[0].Path != two+"#f8" || recs[1].Path != two+"#f16" {
		t.Fatalf("records %+v", recs)
	}
	for i, fn := range []string{"f8", "f16"} {
		ins, _, err := ailower.ReadFunctionIO(two, fn)
		if err != nil || len(ins) != len(recs[i].Inputs) {
			t.Fatalf("%s: inputs %v (%v), want %v", fn, ins, err, recs[i].Inputs)
		}
		if p, f := llm.SplitFunction(recs[i].Path); p != two || f != fn {
			t.Errorf("SplitFunction(%s) = %s, %s", recs[i].Path, p, f)
		}
	}
	if _, err := llm.ReadMeta(&llm.Manifest{Parts: []llm.PartRecord{{Path: two + "#f8"}}}); err != nil {
		t.Errorf("ReadMeta of a function: %v", err)
	}
}
