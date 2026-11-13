package lfm3

import (
	"bytes"
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
