package chroma

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/imagegen/chroma/mil"
	"github.com/shibukawa/hwnpu/internal/llm"
)

func TestShiftedSigmas(t *testing.T) {
	// a linear grid through Chroma1-HD's shift 3: 3 s / (1 + 2 s)
	want := []float64{1, 0.9, 0.75, 0.5, 0}
	got := ShiftedSigmas(4, 3)
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("sigmas %v, want %v", got, want)
		}
	}
	if s := Sigmas(4); s[1] != 0.75 || s[4] != 0 {
		t.Errorf("unshifted sigmas %v", s)
	}
}

func TestModInput(t *testing.T) {
	p := &Pipeline{C: Config{Mods: 3, ModRows: 4, ApproxIn: 64}}
	x := p.modInput(500)
	if len(x) != 4*64 {
		t.Fatalf("%d values", len(x))
	}
	// each row: the timestep's 16, a zero guidance's 16 (cosines 1, sines 0), its index's 32
	if x[0] != float32(math.Cos(500)) || x[16] != 1 || x[24] != 0 || x[64+32] != float32(math.Cos(1000)) {
		t.Errorf("row 0 %v, row 1's index %v", x[:32], x[64+32])
	}
	for _, v := range x[3*64:] {
		if v != 0 {
			t.Fatal("the padding row is not zero")
		}
	}
}

func TestLinkSources(t *testing.T) {
	dir := t.TempDir()
	mk := func(name, model string, size, rows int, t5 string) string {
		d := filepath.Join(dir, name)
		m := &ProgramManifest{Version: ExportVersion, Formats: map[string]string{"t5": t5, "dit": "lut8"},
			Dit: Config{Width: size, Height: size, TextRows: rows}, Text: []string{"text_0.mlpackage"}, Vae: []string{"vae_s0.mlpackage"},
			Doubles: []Group{{Name: "dit_double_0.mlpackage"}}}
		if err := os.MkdirAll(filepath.Join(d, "text_0.mlpackage", "Data"), 0o755); err != nil {
			t.Fatal(err)
		}
		os.WriteFile(filepath.Join(d, "text_0.mlpackage", "Data", "w.bin"), []byte(name), 0o644)
		os.MkdirAll(filepath.Join(d, "vae_s0.mlpackage"), 0o755)
		os.WriteFile(filepath.Join(d, "model.json"), []byte(`{"name": "`+model+`"}`), 0o644)
		if err := writeProgramManifest(d, m); err != nil {
			t.Fatal(err)
		}
		return d
	}
	flash512 := mk("flash512", "Chroma1-Flash", 512, 512, "fp16")
	hd1024 := mk("hd1024", "Chroma1-HD", 1024, 512, "fp16")
	mk("lut8", "Chroma1-HD", 1024, 512, "lut8")
	o := ConvertOptions{Width: 1024, Height: 1024, TextFormat: "fp16", DitFormat: "lut8"}
	f := linkSources([]string{filepath.Join(dir, "lut8"), flash512, hd1024}, "Chroma1-Flash", o, mil.Dit{TextRows: 512})
	if f.text != flash512 || f.model != flash512 || f.vae != filepath.Join(dir, "lut8") || f.dit != "" {
		t.Errorf("sources %+v", f)
	}
	// another T5 format of the same model and size: its DiT
	o = ConvertOptions{Width: 1024, Height: 1024, TextFormat: "lut8", TextFormatFor: []string{"wo=fp16"}, DitFormat: "lut8"}
	if f := linkSources([]string{hd1024}, "Chroma1-HD", o, mil.Dit{TextRows: 512}); f.dit != hd1024 || f.text != "" {
		t.Errorf("sources %+v", f)
	}
	f = linkSources([]string{hd1024, flash512}, "Chroma1-HD", ConvertOptions{Width: 512, Height: 512, TextFormat: "fp16", DitFormat: "lut8"}, mil.Dit{TextRows: 512})
	if f.text != hd1024 || f.model != hd1024 || f.vae != flash512 {
		t.Errorf("sources %+v", f)
	}
	// nothing from an export of another program format (Core AI's)
	if f := linkSources([]string{hd1024}, "Chroma1-HD", ConvertOptions{Backend: llm.BackendCoreAI, Width: 1024, Height: 1024,
		TextFormat: "fp16", DitFormat: "lut8"}, mil.Dit{TextRows: 512}); f != (linkFrom{}) {
		t.Errorf("a Core AI export links from a Core ML one: %+v", f)
	}
	// the last T5 program only from the same model
	if src := (linkFrom{text: flash512}).part("text", "text_0.mlpackage", true); src != "" {
		t.Errorf("the last group linked from another model's export %s", src)
	}
	out := filepath.Join(dir, "out", "text_0.mlpackage")
	if err := linkTree(filepath.Join(flash512, "text_0.mlpackage"), out); err != nil {
		t.Fatal(err)
	}
	a, _ := os.Stat(filepath.Join(flash512, "text_0.mlpackage", "Data", "w.bin"))
	b, _ := os.Stat(filepath.Join(out, "Data", "w.bin"))
	if !os.SameFile(a, b) {
		t.Error("not a hard link")
	}
}
