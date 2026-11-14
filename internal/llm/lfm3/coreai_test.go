package lfm3

import (
	"bytes"
	"fmt"
	"io"
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

// The programs of one model converted for Core AI together (llm.ConvertAI: one that generates and
// one that reads prompts with the last row alone, split at the attention) share a Core AI program
// for each part, a function each, and each has its own manifest with its programs without
// weights; a manifest read back names files that are there.
func TestConvertAISharesParts(t *testing.T) {
	cfg := tinyConfig()
	w := llm.NewSynthetic(cfg)
	dir := t.TempDir()
	opts := llm.Options{Seq: 8, Cache: 256, Paged: 64, Format: "lut4", Params: map[string]string{}}
	pre := opts
	pre.Seq, pre.LastRow = 16, true
	base := filepath.Join(dir, "m-c256")
	paths, err := llm.ConvertAI(Arch, cfg, w, base+"-q8-q16-last", "none", []llm.AIProgram{
		{Out: base + "-q8", Name: "q8", Options: opts},
		{Out: base + "-q16-last", Name: "q16-last", Options: pre},
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if len(paths) != 2 {
		t.Fatalf("paths %v", paths)
	}
	var files [2][]string
	for i, p := range paths {
		man, err := llm.ReadManifest(p)
		if err != nil {
			t.Fatal(err)
		}
		if man.Backend != llm.BackendCoreAI || man.Paged != 64 || man.Attention == nil || man.Mass == nil || man.Meta[llm.MetaArch] != Arch.Name {
			t.Errorf("%s: %+v", p, man)
		}
		if len(man.Parts) != 2 {
			t.Fatalf("%s: %d parts, want 2 (split at layer 1)", p, len(man.Parts))
		}
		for _, part := range man.Parts {
			asset, fn := llm.SplitFunction(part.Path)
			if fn != []string{"q8", "q16-last"}[i] {
				t.Errorf("%s: function %q", part.Path, fn)
			}
			if _, err := os.Stat(filepath.Join(asset, "main.mlirb")); err != nil {
				t.Error(err)
			}
			files[i] = append(files[i], asset)
		}
		for _, r := range []*llm.PartRecord{man.Attention, man.Mass} {
			if _, err := os.Stat(filepath.Join(r.Path, "main.mlirb")); err != nil {
				t.Error(err)
			}
		}
		if meta, err := llm.ReadMeta(man); err != nil || meta[llm.MetaFormat] != "lut4" {
			t.Errorf("%s: metadata %v (%v)", p, meta, err)
		}
	}
	if fmt.Sprint(files[0]) != fmt.Sprint(files[1]) {
		t.Errorf("the programs' parts are in different files: %v and %v", files[0], files[1])
	}
}

// Convert for Core AI writes a program of one part as one .aimodel, not a manifest.
func TestConvertAIOnePart(t *testing.T) {
	cfg := tinyConfig()
	w := llm.NewSynthetic(cfg)
	out := filepath.Join(t.TempDir(), "m.aimodel")
	path, err := llm.Convert(Arch, cfg, w, out, "none", 0, llm.Options{Seq: 8, Format: "fp16", Backend: llm.BackendCoreAI, Params: map[string]string{}}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	if path != out {
		t.Fatalf("path %s, want %s", path, out)
	}
	if _, err := os.Stat(filepath.Join(out, "main.mlirb")); err != nil {
		t.Fatal(err)
	}
}
