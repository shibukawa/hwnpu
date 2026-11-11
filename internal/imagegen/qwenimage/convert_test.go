package qwenimage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestConvertParts converts the parts HWNPU_QWENIMAGE_ONLY (text, dit_text, blocks, time, vae) of
// HWNPU_QWENIMAGE_ROOT into the export HWNPU_QWENIMAGE_OUT in the formats HWNPU_QWENIMAGE_DIT_TEXT
// and HWNPU_QWENIMAGE_DIT (format or format:quant): a way to try one part's format in a copy of an
// export.
func TestConvertParts(t *testing.T) {
	root, out, only := os.Getenv("HWNPU_QWENIMAGE_ROOT"), os.Getenv("HWNPU_QWENIMAGE_OUT"), os.Getenv("HWNPU_QWENIMAGE_ONLY")
	if root == "" || out == "" || only == "" {
		t.Skip("HWNPU_QWENIMAGE_ROOT, _OUT and _ONLY not set")
	}
	m, err := ReadProgramManifest(out)
	if err != nil {
		t.Fatal(err)
	}
	o := ConvertOptions{Width: m.Dit.Width, Height: m.Dit.Height, Group: m.Dit.Group, TextFormat: m.Formats["text"],
		DitFormat: m.Formats["dit"], DitTextFormat: m.Formats["dit_text"], TextQuant: m.Formats["text_quant"], DitQuant: m.Formats["dit_quant"],
		Only: map[string]bool{}}
	for _, p := range strings.Split(only, ",") {
		o.Only[p] = true
	}
	if f := os.Getenv("HWNPU_QWENIMAGE_DIT"); f != "" {
		o.DitFormat, o.DitQuant, _ = strings.Cut(f, ":")
	}
	if f := os.Getenv("HWNPU_QWENIMAGE_DIT_TEXT"); f != "" {
		o.DitTextFormat, _, _ = strings.Cut(f, ":")
	}
	if err := Convert(root, filepath.Join(filepath.Dir(root), TextDir), out, o, os.Stderr); err != nil {
		t.Fatal(err)
	}
}

// TestExportName checks the defaults that name an export: the text encoder lut8, the DiT's text
// pass lut8 and its image blocks lut4 (k-means with the extremes kept).
func TestExportName(t *testing.T) {
	got := ExportName("Qwen-Image-2.1-Turbo", ConvertOptions{Width: 512})
	if want := "qwen-image-2.1-turbo-512x512-lut8-lut8-kmm-lut4-kmm-g4-v1.coreml"; got != want {
		t.Errorf("%s, want %s", got, want)
	}
	if got := ExportName("Qwen-Image-2.1-Turbo", ConvertOptions{Width: 1024, DitFormat: "lut6"}); !strings.Contains(got, "1024x1024-lut8-lut8-kmm-lut6-kmm") {
		t.Errorf("%s", got)
	}
}

// TestShareText checks that an export of another size takes the text encoder's and the DiT text
// pass's programs of a whole export of the same formats as hard links, and not of other formats.
func TestShareText(t *testing.T) {
	dir := t.TempDir()
	o := ConvertOptions{Width: 512}
	o.defaults()
	from := filepath.Join(dir, ExportName("Qwen-Image-2.1-Turbo", o))
	m := ProgramManifest{Kind: ProgramKind, Version: ExportVersion, Formats: map[string]string{"text": o.TextFormat, "text_quant": o.TextQuant,
		"dit_text": o.DitTextFormat, "dit_quant": o.DitQuant}, Dit: DitConfig{Group: o.Group}, TextIn: []string{"text_0.mlpackage"},
		DitText: []string{"dit_text_0.mlpackage"}}
	for _, f := range []string{"text/embed.bin", "text_0.mlpackage/Data/weight.bin", "dit_text_0.mlpackage/Data/weight.bin"} {
		if err := os.MkdirAll(filepath.Dir(filepath.Join(from, f)), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(from, f), []byte(f), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	b, _ := json.Marshal(m)
	os.WriteFile(filepath.Join(from, "manifest.json"), b, 0o644)
	o2 := ConvertOptions{Width: 1024}
	o2.defaults()
	if got := sharedExport(dir, o2); got != "" {
		t.Fatalf("an export without .done shared: %s", got)
	}
	os.WriteFile(from+".done", nil, 0o644)
	if got := sharedExport(dir, o2); got != from {
		t.Fatalf("shared %q, want %q", got, from)
	}
	o3 := o2
	o3.DitTextFormat = "lut6"
	if got := sharedExport(dir, o3); got != "" {
		t.Fatalf("a text pass of another format shared: %s", got)
	}
	out := filepath.Join(dir, "out")
	if err := linkShared(from, out); err != nil {
		t.Fatal(err)
	}
	a, _ := os.Stat(filepath.Join(from, "dit_text_0.mlpackage/Data/weight.bin"))
	c, err := os.Stat(filepath.Join(out, "dit_text_0.mlpackage/Data/weight.bin"))
	if err != nil || !os.SameFile(a, c) {
		t.Fatalf("not linked: %v", err)
	}
}
