package sana

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
)

// The files of the repository that are read: every configuration, and the weights of what is
// converted alone.
func TestNeeds(t *testing.T) {
	all := map[string]bool{"text": true, "caption": true, "time": true, "blocks": true, "final": true, "vae": true}
	for _, c := range []struct {
		name string
		only map[string]bool
		want []string
	}{
		{"all of it", all, []string{"transformer/config.json", "transformer/diffusion_pytorch_model.safetensors", "vae/config.json",
			"vae/diffusion_pytorch_model.safetensors", "scheduler/scheduler_config.json", "text_encoder/config.json",
			"text_encoder/model.safetensors.index.json", "text_encoder/model-00001-of-00002.safetensors",
			"text_encoder/model-00002-of-00002.safetensors", "tokenizer/tokenizer.json"}},
		{"Gemma 2 alone (the DiT from an older export)", map[string]bool{"text": true}, []string{"transformer/config.json", "vae/config.json",
			"scheduler/scheduler_config.json", "text_encoder/config.json", "text_encoder/model.safetensors.index.json",
			"text_encoder/model-00001-of-00002.safetensors", "text_encoder/model-00002-of-00002.safetensors", "tokenizer/tokenizer.json"}},
		{"the DiT alone (Gemma 2 from another Sana)", map[string]bool{"caption": true, "time": true, "blocks": true, "final": true, "vae": true},
			[]string{"transformer/config.json", "transformer/diffusion_pytorch_model.safetensors", "vae/config.json",
				"vae/diffusion_pytorch_model.safetensors", "scheduler/scheduler_config.json", "text_encoder/config.json"}},
		{"nothing (both from other exports)", map[string]bool{"text": false}, []string{"transformer/config.json", "vae/config.json",
			"scheduler/scheduler_config.json", "text_encoder/config.json"}},
	} {
		var got []string
		for _, f := range Files {
			if needs(f, c.only) {
				got = append(got, f.Name)
			}
		}
		if !slices.Equal(got, c.want) {
			t.Errorf("%s: reads %v, want %v", c.name, got, c.want)
		}
	}
}

// writeExport writes an export's manifest, its parts as files (a program as a directory with a
// file) and its .done.
func writeExport(t *testing.T, dir string, m *ProgramManifest, parts ...string) {
	t.Helper()
	for _, p := range parts {
		path := filepath.Join(dir, p)
		if filepath.Ext(p) == ".aimodel" {
			path = filepath.Join(path, "main.mlirb")
		}
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(p), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := writeProgramManifest(dir, m); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dir+".done", nil, 0o644); err != nil {
		t.Fatal(err)
	}
}

// A new export takes the DiT's and the VAE's programs from this Sana's export of the layout
// before (hard links, each program once whatever functions name it), and Gemma 2's from another
// Sana's export of this layout.
func TestLinkExports(t *testing.T) {
	dir := t.TempDir()
	o := ConvertOptions{Width: 512, Height: 512, Backend: llm.BackendCoreAI}
	o.defaults("Sana-600M")
	// an export of the layout before this one; when it holds other DiT programs (DitVersion is
	// this layout), ditExport takes none, and linkDit is checked on it alone
	older := DitVersion < ExportVersion
	old := filepath.Join(dir, exportNameAt("Sana-600M", o, ExportVersion-1))
	if want := filepath.Join(dir, fmt.Sprintf("sana-600m-512+1024-int8-int8-g4-v%d.coreai", ExportVersion-1)); old != want {
		t.Fatalf("the older export is %s, want %s", old, want)
	}
	if got := ditExport(dir, "Sana-600M", o); got != "" {
		t.Errorf("an older export found in an empty folder: %s", got)
	}
	m := &ProgramManifest{Backend: llm.BackendCoreAI, Version: ExportVersion - 1, Formats: map[string]string{"text": "int8", "dit": "int8"},
		TextParts: []string{"text_0.aimodel"}, Caption: "caption.aimodel", Time: TimeFile,
		Blocks: []string{"dit_blocks_0.aimodel#s512x512"}, Final: "dit_final.aimodel#s512x512", Vae: []string{"vae_s0.aimodel#s512x512"},
		Sizes: []SizeParts{
			{Width: 512, Height: 512, Blocks: []string{"dit_blocks_0.aimodel#s512x512"}, Final: "dit_final.aimodel#s512x512", Vae: []string{"vae_s0.aimodel#s512x512"}},
			{Width: 1024, Height: 1024, Blocks: []string{"dit_blocks_0.aimodel#s1024x1024"}, Final: "dit_final.aimodel#s1024x1024", Vae: []string{"vae_s0.aimodel#s1024x1024"}},
		}}
	writeExport(t, old, m, "text_0.aimodel", "caption.aimodel", TimeFile, "dit_blocks_0.aimodel", "dit_final.aimodel", "vae_s0.aimodel", "text/tokenizer.json")
	if got := ditExport(dir, "Sana-600M", o); older && got != old || !older && got != "" {
		t.Fatalf("the older export: %q (DitVersion %d, ExportVersion %d)", got, DitVersion, ExportVersion)
	}
	// not another Sana's, not another DiT format's, not one that is not whole
	if got := ditExport(dir, "SANA1.5-1.6B", o); got != "" {
		t.Errorf("another Sana's export taken: %s", got)
	}
	fp := o
	fp.DitFormat = "fp16"
	if got := ditExport(dir, "Sana-600M", fp); got != "" {
		t.Errorf("an export of another DiT format taken: %s", got)
	}
	out := filepath.Join(dir, ExportName("Sana-600M", o))
	if err := linkDit(old, out); err != nil {
		t.Fatal(err)
	}
	same := func(name string) bool {
		a, err1 := os.Stat(filepath.Join(old, name))
		b, err2 := os.Stat(filepath.Join(out, name))
		return err1 == nil && err2 == nil && os.SameFile(a, b)
	}
	for _, name := range []string{"caption.aimodel/main.mlirb", TimeFile, "dit_blocks_0.aimodel/main.mlirb", "dit_final.aimodel/main.mlirb", "vae_s0.aimodel/main.mlirb"} {
		if !same(name) {
			t.Errorf("%s is not linked", name)
		}
	}
	for _, name := range []string{"text_0.aimodel", "text", "manifest.json"} {
		if _, err := os.Stat(filepath.Join(out, name)); err == nil {
			t.Errorf("%s of the older layout was taken", name)
		}
	}
	// Gemma 2's programs of an export of this layout, whose functions name each program
	cur := filepath.Join(dir, fmt.Sprintf("sana1.5-1.6b-512+1024-int8-lut8-g4-v%d.coreai", ExportVersion))
	writeExport(t, cur, &ProgramManifest{Backend: llm.BackendCoreAI, Version: ExportVersion, Formats: map[string]string{"text": "int8", "dit": "lut8"},
		TextParts: []string{"text_0.aimodel#" + WholeFunction}, TextFrom: []TextFunc{{Rows: 64, Parts: []string{"text_0.aimodel#" + TextFunction(64)}}}},
		"text_0.aimodel", "text/tokenizer.json", HeldFile, HeldOutFile)
	if got := textExport(dir, o); got != cur {
		t.Fatalf("Gemma 2's export: %q, want %s (the older layout's is not taken)", got, cur)
	}
	if err := linkText(cur, out); err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"text_0.aimodel/main.mlirb", "text/tokenizer.json", HeldFile, HeldOutFile} {
		a, err1 := os.Stat(filepath.Join(cur, name))
		b, err2 := os.Stat(filepath.Join(out, name))
		if err1 != nil || err2 != nil || !os.SameFile(a, b) {
			t.Errorf("%s is not linked from the export of this layout", name)
		}
	}
	// an export that is not whole gives nothing
	os.Remove(old + ".done")
	if got := ditExport(dir, "Sana-600M", o); got != "" {
		t.Errorf("an export without its .done taken: %s", got)
	}
}

// The held rows' tokens and last hidden states are read back as written, and refused when they
// are not the programs'.
func TestHeldFiles(t *testing.T) {
	dir := t.TempDir()
	if err := os.MkdirAll(filepath.Join(dir, "text"), 0o755); err != nil {
		t.Fatal(err)
	}
	ids := []int32{2, 7, 9}
	last := []float32{1, -2, 0.5, 4, 5, 6}
	if err := writeHeld(dir, ids, last); err != nil {
		t.Fatal(err)
	}
	gotIDs, gotLast, err := readHeld(dir, TextConfig{Prefix: 3, Hidden: 2})
	if err != nil || !slices.Equal(gotIDs, ids) || !slices.Equal(gotLast, last) {
		t.Errorf("read %v, %v, %v", gotIDs, gotLast, err)
	}
	if _, _, err := readHeld(dir, TextConfig{Prefix: 4, Hidden: 2}); err == nil {
		t.Error("three tokens read for four held rows")
	}
	if _, _, err := readHeld(dir, TextConfig{Prefix: 3, Hidden: 4}); err == nil {
		t.Error("rows of two values read for a hidden width of four")
	}
}
