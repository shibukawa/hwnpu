package recipe

import (
	"slices"
	"strings"
	"testing"
)

// The built-in recipes make the exports for xdna2; a command takes the code's values for its
// placeholders, and the generation's extra arguments after its own.
func TestAMDCommand(t *testing.T) {
	f, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(f.AMD.Generations, []string{"xdna2"}) {
		t.Errorf("generations %v", f.AMD.Generations)
	}
	vars := map[string][]string{In: {"/repo"}, Sizes: {"1024", "512"}}
	c, err := f.AMDCommand("FLUX.2-klein-4B", 0, "xdna2", vars)
	if err != nil || c.Script != "iron_flux2klein.py" || strings.Join(c.Args, " ") != "/repo {out} --size 1024 512" ||
		strings.Join(c.Options, " ") != "--size 1024 512" {
		t.Errorf("FLUX.2 klein: %+v %v", c, err)
	}
	if c, _ := f.AMDCommand("FLUX.2-klein-4B", 0, "xdna1", vars); strings.Join(c.Args, " ") != "/repo {out} --size 1024 512 --device npu1" {
		t.Errorf("FLUX.2 klein on xdna1: %v", c.Args)
	}
	if _, err := f.AMDCommand("FLUX.2-klein-4B", 0, "xdna2", map[string][]string{In: {"/repo"}}); err == nil || !strings.Contains(err.Error(), Sizes) {
		t.Errorf("no sizes: %v", err)
	}
	if _, err := f.AMDCommand("Qwen3-0.6B", 0, "xdna2", vars); err == nil {
		t.Error("a model without a recipe for AMD's NPU")
	}
	// a variant's args from its GB up; the language models' extra arguments on xdna1
	for gb, want := range map[int]bool{0: true, 16: true, 24: false, 64: false} {
		c, err := f.AMDCommand("gpt-oss-20b", gb, "xdna2", vars)
		if err != nil || slices.Contains(c.Args, "--expert-places") != want {
			t.Errorf("gpt-oss-20b on %d GB: %v %v, want places %v", gb, c.Args, err, want)
		}
	}
	if got := f.AMDExtra("LFM3-1.7B", "xdna1"); strings.Join(got, " ") != "--device npu1 --cols 4" {
		t.Errorf("LFM3 on xdna1: %v", got)
	}
	if got := f.AMDExtra("LFM3-1.7B", "xdna2"); len(got) != 0 {
		t.Errorf("LFM3 on xdna2: %v", got)
	}
	if gens, skips := f.AMDGenerations("whisper-large-v3-turbo"); !slices.Equal(gens, []string{"xdna2"}) || len(skips) != 0 {
		t.Errorf("whisper: %v %v", gens, skips)
	}
	if GenerationOf("amd-xdna2") != "xdna2" || GenerationOf("coreai") != "" || AMDTarget("xdna1") != "amd-xdna1" {
		t.Error("targets")
	}
}

// A model's generations and skips: a skipped generation is not made.
func TestAMDGenerations(t *testing.T) {
	f, err := Parse("t", []byte(`version: 1
amd: {generations: [xdna2]}
models:
  m:
    amd:
      generations: [xdna1, xdna2]
      script: iron_whisper.py
      args: ["{in}", "{out}"]
      skip:
        - {generation: xdna1, why: "too large", checked: 2026-11-23}
`))
	if err != nil {
		t.Fatal(err)
	}
	if gens, skips := f.AMDGenerations("m"); !slices.Equal(gens, []string{"xdna2"}) || len(skips) != 1 {
		t.Errorf("%v %v", gens, skips)
	}
}

func TestParseRejectsAMD(t *testing.T) {
	head := "version: 1\nmodels:\n  m:\n    amd:\n"
	for _, c := range []struct{ amd, want string }{
		{"      script: iron_nothing.py\n      args: [\"{in}\", \"{out}\"]\n", "not a script"},
		{"      args: [\"{in}\", \"{out}\"]\n", "no script"},
		{"      script: iron_whisper.py\n      args: [\"{in}\"]\n", "no {out}"},
		{"      script: iron_whisper.py\n      args: [\"{in}\", \"{out}\", \"--size={size}\"]\n", "whole argument"},
		{"      script: iron_whisper.py\n      args: [\"{in}\", \"{out}\", \"{width}\"]\n", "whole argument"},
		{"      script: iron_whisper.py\n      args: [\"{in}\", \"{out}\"]\n      generations: [npu2]\n", "generation"},
		{"      script: iron_whisper.py\n      args: [\"{in}\", \"{out}\"]\n      extra: {xdna3: [a]}\n", "generation"},
		{"      script: iron_whisper.py\n      args: [\"{in}\", \"{out}\"]\n      skip: [{generation: xdna1, checked: 2026-11-23}]\n", "why"},
		{"      script: iron_whisper.py\n      args: [\"{in}\", \"{out}\"]\n      skip: [{generation: xdna1, why: x, checked: today}]\n", "checked"},
		{"      script: iron_whisper.py\n      args: [\"{in}\", \"{out}\"]\n      variants: {24: [a]}\n", "no {out}"},
		{"      script: iron_whisper.py\n      args: [\"{in}\", \"{out}\"]\n      device: npu1\n", "unknown"},
	} {
		_, err := Parse("t", []byte(head+c.amd))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error of %q", c.amd, err, c.want)
		}
	}
	if _, err := Parse("t", []byte("version: 1\namd: {generations: [xdna0]}\n")); err == nil {
		t.Error("generation xdna0")
	}
}
