package recipe

import (
	"strings"
	"testing"

	"github.com/goccy/go-yaml"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// The embedded recipes read, and name models of models.yaml.
func TestBuiltin(t *testing.T) {
	f, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	if f.Tools.CoreAIBuild == "" || len(f.Apple.Archs) == 0 {
		t.Errorf("tools %+v, apple %+v", f.Tools, f.Apple)
	}
	for name := range f.Models {
		if _, ok := chat.Lookup(name); !ok {
			t.Errorf("%s is not a model of models.yaml", name)
		}
	}
}

// FLUX.2 klein's single blocks go to the device on an M1 alone; Chroma's on an M1, and three of
// them on an M2.
func TestLeft(t *testing.T) {
	f, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	flux, chroma := f.AppleOf("FLUX.2-klein-4B"), f.AppleOf("Chroma1-Flash")
	for _, c := range []struct {
		a             Apple
		program, arch string
		want          bool
	}{
		{flux, "dit_single_0.aimodel", "h13g", true},
		{flux, "dit_single_15", "h13g", true},
		{flux, "dit_double_0.aimodel", "h13g", false},
		{flux, "dit_single_0.aimodel", "h14g", false},
		{flux, "dit_single_0.aimodel", "h15g", false},
		{chroma, "dit_single_31.aimodel", "h13g", true},
		{chroma, "dit_single_8.aimodel", "h14g", true},
		{chroma, "dit_single_24.aimodel", "h14g", false},
		{chroma, "dit_single_8.aimodel", "h15g", false},
	} {
		if _, got := c.a.Left("x.coreai", c.program, c.arch); got != c.want {
			t.Errorf("%s on %s: left %v, want %v", c.program, c.arch, got, c.want)
		}
	}
	if flux.Jobs != 3 || chroma.Jobs != 2 || strings.Join(chroma.Archs, " ") != "h13g h14g h15g" {
		t.Errorf("jobs %d %d, archs %v", flux.Jobs, chroma.Jobs, chroma.Archs)
	}
	// a model without a recipe is built as apple says
	if a := f.AppleOf("Qwen3-0.6B"); len(a.Prebuild) != 0 || a.Jobs != f.Apple.Jobs {
		t.Errorf("no recipe: %+v", a)
	}
}

// A rule must say what it does, why and when; names of architectures and patterns are checked.
func TestParseRejects(t *testing.T) {
	head := "version: 1\ntools: {coreai-build: 1}\napple:\n  archs: [h13g]\n  prebuild:\n"
	for _, c := range []struct{ rule, want string }{
		{"    - {programs: [a], do: rewrite, why: x, checked: 2026-11-23}", "do"},
		{"    - {programs: [a], do: device, checked: 2026-11-23}", "why"},
		{"    - {programs: [a], do: device, why: x, checked: yesterday}", "checked"},
		{"    - {arch: m1, programs: [a], do: device, why: x, checked: 2026-11-23}", "architecture"},
		{"    - {do: device, why: x, checked: 2026-11-23}", "no programs"},
		{"    - {programs: [\"[\"], do: device, why: x, checked: 2026-11-23}", "pattern"},
		{"    - {programs: [a], do: device, why: x, checked: 2026-11-23, at: 1}", "unknown"},
	} {
		_, err := Parse("t", []byte(head+c.rule+"\n"))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error of %q", c.rule, err, c.want)
		}
	}
	if _, err := Parse("t", []byte(head+"    - {arch: h13g, programs: a*, do: device, why: x, checked: 2026-11-23}\n")); err != nil {
		t.Errorf("one arch and one program as strings: %v", err)
	}
}

// What a failed build suggests reads as a recipe and leaves the program to the device.
func TestSuggest(t *testing.T) {
	s := Suggest("FLUX.2-klein-4B", "dit_single_5.aimodel", "h13g", `coreai-build left it off: "RegAlloc"
failed`, "2026-11-23")
	var f File
	if err := yaml.Unmarshal([]byte("version: 1\n"+s), &f); err != nil {
		t.Fatalf("%v\n%s", err, s)
	}
	f.Version = 1
	if _, ok := f.AppleOf("FLUX.2-klein-4B").Left("e.coreai", "dit_single_5", "h13g"); !ok {
		t.Errorf("suggested rule does not leave it:\n%s", s)
	}
}
