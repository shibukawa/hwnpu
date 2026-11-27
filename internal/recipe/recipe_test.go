package recipe

import (
	"strings"
	"testing"

	"github.com/goccy/go-yaml"
)

// The embedded recipes read (internal/chat checks that they name models of models.yaml).
func TestBuiltin(t *testing.T) {
	f, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	if f.Tools.CoreAIBuild == "" || len(f.Apple.Archs) == 0 {
		t.Errorf("tools %+v, apple %+v", f.Tools, f.Apple)
	}
}

// A recipe's prebuild rules leave programs to the device: MOSS's all of them; FLUX.2 klein's none
// since its single blocks are two a program (they went to the device on an M1 at five, until
// 2026-11-24).
func TestLeft(t *testing.T) {
	f, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	flux, moss := f.AppleOf("FLUX.2-klein-4B"), f.AppleOf("MOSS-SoundEffect-v2.0")
	for _, c := range []struct {
		a             Apple
		program, arch string
		want          bool
	}{
		{flux, "dit_single_0.aimodel", "h13g", false},
		{flux, "dit_double_0.aimodel", "h13g", false},
		{flux, "dit_single_0.aimodel", "h14g", false},
		{flux, "dit_single_0.aimodel", "h15g", false},
		{moss, "decoder_0.aimodel", "h13g", true},
		{moss, "decoder_0.aimodel", "h15g", true},
	} {
		if _, got := c.a.Left("x.coreai", c.program, c.arch); got != c.want {
			t.Errorf("%s on %s: left %v, want %v", c.program, c.arch, got, c.want)
		}
	}
	if flux.Jobs != 3 || strings.Join(flux.Archs, " ") != "h13g h14g h15g" {
		t.Errorf("jobs %d, archs %v", flux.Jobs, flux.Archs)
	}
	// the image generators alone are prebuilt when a build names no architecture
	for model, want := range map[string]bool{"FLUX.2-klein-4B": true, "flux-3-image-klein": true, "Anima-Turbo": true, "Anima-Aesthetic": true,
		"MOSS-SoundEffect-v2.0": false, "LFM3-1.7B": true, "LFM3-VL-2B": true, "LFM3-Omni-2B": true, "Qwen3.5-0.8B": true, "Qwen3.5-4B": true, "Qwen3.5-2B": true, "whisper-large-v3-turbo": false, "DA3-BASE": false, "no-such-model": false} {
		if got := f.AppleOf(model).Prebuilt; got != want {
			t.Errorf("%s: prebuilt %v, want %v", model, got, want)
		}
	}
	// Qwen3.5's chat models are prebuilt for the M1 alone
	if a := f.AppleOf("Qwen3.5-0.8B"); strings.Join(a.Archs, " ") != "h13g" {
		t.Errorf("Qwen3.5-0.8B: archs %v", a.Archs)
	}
	// a model without a recipe of its own is built as apple says, its rules too: the chat
	// models' attention programs are left to the device on the M1 alone (docs/known-issues.md 19)
	qwen := f.AppleOf("Qwen3-0.6B")
	if len(qwen.Prebuild) != len(f.Apple.Prebuild) || qwen.Jobs != f.Apple.Jobs {
		t.Errorf("no recipe: %+v", qwen)
	}
	for _, c := range []struct {
		program, arch string
		want          bool
	}{{"q8.attn1024.aimodel", "h13g", true}, {"model.attn.aimodel", "h13g", true}, {"q8.attn1024.aimodel", "h14g", false}, {"model.part1.aimodel", "h13g", false}} {
		if _, got := qwen.Left("x.coreai", c.program, c.arch); got != c.want {
			t.Errorf("Qwen3-0.6B %s on %s: left %v, want %v", c.program, c.arch, got, c.want)
		}
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

// TestVAELeftToDevice: FLUX.2 and FLUX.3 klein's VAE is left to the device on every chip, Anima's on M2 alone (prebuilt on M1 and on M3 and later), and their other programs are not left by these rules.
func TestVAELeftToDevice(t *testing.T) {
	f, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		model, program, arch string
		want                 bool
	}{
		{"FLUX.2-klein-4B", "vae_s0", "h13g", true},
		{"FLUX.2-klein-4B", "vae_enc_s3", "h13g", true},
		{"FLUX.2-klein-4B", "vae_s0", "h14g", true},
		{"FLUX.2-klein-4B", "vae_s0", "h15g", true},
		{"FLUX.2-klein-4B", "vae_enc_s3", "h15g", true},
		{"FLUX.2-klein-4B", "dit_double_0", "h15g", false},
		{"flux-3-image-klein", "vae_s0", "h13g", true},
		{"flux-3-image-klein", "vae_s9", "h14g", true},
		{"flux-3-image-klein", "vae_s9", "h15g", true},
		{"Anima-Turbo", "vae_s0", "h15g", false},
		{"Anima-Turbo", "vae_s0", "h13g", false},
		{"Anima-Aesthetic", "vae_s9", "h14g", true},
		{"Anima-Aesthetic", "vae_s9", "h16g", false},
		{"Anima-Turbo", "dit_blocks_0", "h15g", false},
	} {
		if _, got := f.AppleOf(c.model).Left("e.coreai", c.program, c.arch); got != c.want {
			t.Errorf("%s %s on %s: left %v", c.model, c.program, c.arch, got)
		}
	}
}

// TestRewrites: MOSS's decoder writes sin itself on M3 and later, by its series on M1 and M2,
// and a build converts an export of each.
func TestRewrites(t *testing.T) {
	f, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range []string{"MOSS-SoundEffect-v2.0", "MOSS-SoundEffect-v2.0-lut8"} {
		a := f.AppleOf(m)
		for arch, want := range map[string]string{"h13g": "", "h14g": "", "h15g": "-sin-native", "h17g": "-sin-native", "h15c": "-sin-native", "h17d": "-sin-native", "h14c": "", "h18g": ""} {
			if got := Suffix(a.Rewrites(arch), "sin"); got != want {
				t.Errorf("%s %s: %q, want %q", m, arch, got, want)
			}
		}
		if got := a.RewriteArchs(); len(got) != 2 || Suffix(a.Rewrites(got[0])) != "" || Suffix(a.Rewrites(got[1])) != "-sin-native" {
			t.Errorf("%s: RewriteArchs %v, want a default's then a native one's", m, got)
		}
		if _, ok := a.Left("hwnpu-dac-r3-fp16.coreai", "dac-0.aimodel", "h15g"); !ok {
			t.Errorf("%s: the decoder is prebuilt", m)
		}
	}
	if got := f.AppleOf("FLUX.2-klein-4B").RewriteArchs(); len(got) != 1 {
		t.Errorf("FLUX.2 klein: RewriteArchs %v, want one", got)
	}
	if Suffix(map[string]string{"sin": "native"}, "exp") != "" {
		t.Error("a suffix of an operation the programs do not have")
	}
	t.Setenv("HWNPU_DEV", "rewrite=sin:series")
	if rw, err := RewritesFor("MOSS-SoundEffect-v2.0", "h15g", true); err != nil || len(rw) != 0 {
		t.Errorf("rewrite=sin:series on h15g: %v %v, want none", rw, err)
	}
	if rw, _ := RewritesFor("MOSS-SoundEffect-v2.0", "h15g", false); rw["sin"] != "native" {
		t.Errorf("another chip's: %v, want the recipe's", rw)
	}
	t.Setenv("HWNPU_DEV", "rewrite=sin:table")
	if _, err := RewritesFor("MOSS-SoundEffect-v2.0", "h13g", true); err == nil {
		t.Error("rewrite=sin:table: no error")
	}
}

func TestParseRejectsRewrites(t *testing.T) {
	head := "version: 1\ntools: {coreai-build: 1}\napple:\n  rewrite:\n"
	for _, c := range []struct{ rule, want string }{
		{"    - {op: cos, as: native, why: x, checked: 2026-11-23}", "op"},
		{"    - {op: sin, as: table, why: x, checked: 2026-11-23}", "as"},
		{"    - {op: sin, as: native, checked: 2026-11-23}", "why"},
		{"    - {arch: m3, op: sin, as: native, why: x, checked: 2026-11-23}", "architecture"},
	} {
		_, err := Parse("t", []byte(head+c.rule+"\n"))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want an error of %q", c.rule, err, c.want)
		}
	}
}
