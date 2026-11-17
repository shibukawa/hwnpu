package flux2klein

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
)

// writeJSON writes v as path's JSON, making its directory.
func writeJSON(t *testing.T, path string, v any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		t.Fatal(err)
	}
}

// writeFile writes s as path, making its directory.
func writeFile(t *testing.T, path, s string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(s), 0o644); err != nil {
		t.Fatal(err)
	}
}

// graph writes a graph's manifest in dir/name: its banks (bank_scratch.bin and the groups'
// variants) as files of bankDir relative to it, its ELF of elf's bytes, and meta.
func graph(t *testing.T, dir, name, bankDir string, groups int, elf string, meta map[string]any) {
	t.Helper()
	rel := func(f string) string {
		if bankDir == "" {
			return f
		}
		return "../" + bankDir + "/" + f
	}
	var variants []map[string]string
	for g := 0; g < groups; g++ {
		f := "bank_scratch.bin"
		if g > 0 {
			f = fmt.Sprintf("bank_g%d_scratch.bin", g)
		}
		variants = append(variants, map[string]string{"scratch": rel(f)})
	}
	m := map[string]any{"version": 1, "kind": "graph", "name": name,
		"banks": map[string]any{"scratch": map[string]any{"bytes": 763656192, "image": rel("bank_scratch.bin")}},
		"graph": map[string]any{"elf": "graph.elf", "kernels": []map[string]any{{"name": "k"}}}, "meta": meta}
	if groups > 1 {
		m["variants"] = variants
	}
	writeJSON(t, filepath.Join(dir, name, "manifest.json"), m)
	writeFile(t, filepath.Join(dir, name, "graph.elf"), elf)
}

// fakeSized writes an IRON export of FLUX.2 klein of one size in dir, its graphs of the
// references' rows refRows too, with weights (the banks, the text's) of the bytes weights.
func fakeSized(t *testing.T, dir string, size int, refRows []int, weights string) {
	t.Helper()
	cfg := map[string]any{"width": size, "height": size, "text_rows": 512, "hidden": 3072, "head_dim": 128,
		"latent": 128, "text": 2560, "axes": []int{32, 32, 32, 32}, "theta": 2000.0, "doubles": 5, "singles": 20,
		"double_group": 1, "single_group": 2}
	meta := map[string]any{"config": cfg}
	for _, part := range []struct {
		name   string
		groups int
	}{{"doubles", 3}, {"singles", 2}} {
		graph(t, dir, part.name, "", part.groups, fmt.Sprintf("elf %s %d", part.name, size), meta)
		for g := 0; g < part.groups; g++ {
			f := "bank_scratch.bin"
			if g > 0 {
				f = fmt.Sprintf("bank_g%d_scratch.bin", g)
			}
			writeFile(t, filepath.Join(dir, part.name, f), weights+part.name+f)
		}
		for _, r := range refRows {
			graph(t, dir, fmt.Sprintf("%s_n%d", part.name, r), part.name, part.groups, fmt.Sprintf("elf %s rows %d", part.name, r), meta)
		}
	}
	for _, r := range append([]int{0}, refRows...) {
		name := "dit_in"
		if r > 0 {
			name = fmt.Sprintf("dit_in_n%d", r)
		}
		graph(t, dir, name, "", 1, fmt.Sprintf("elf dit_in %d %d", size, r), meta)
		writeFile(t, filepath.Join(dir, name, "bank_scratch.bin"), weights+"dit_in")
	}
	for _, name := range []string{"mods", "ctx", "text", "dit_out"} {
		groups := 1
		if name == "text" {
			groups = 2
		}
		graph(t, dir, name, "", groups, "elf "+name, map[string]any{"config": cfg, "taps": []int{0, 1, 2}})
		writeFile(t, filepath.Join(dir, name, "bank_scratch.bin"), weights+name)
		if groups > 1 {
			writeFile(t, filepath.Join(dir, name, "bank_g1_scratch.bin"), weights+name+"g1")
		}
	}
	writeFile(t, filepath.Join(dir, "dit_out", "bank_scratch.bin"), fmt.Sprintf("dit_out of %d", size))
	writeFile(t, filepath.Join(dir, "text", "embed.bin"), weights+"embed")
	writeFile(t, filepath.Join(dir, "text", "tokenizer.json"), "{}")
	for _, enc := range []struct {
		name string
		side int
	}{{"vae", size}, {"vae_enc", size}, {"vae_enc_d2", size / 2}} {
		writeJSON(t, filepath.Join(dir, enc.name, "manifest.json"), map[string]any{"stages": 1, "width": enc.side, "height": enc.side})
		writeFile(t, filepath.Join(dir, enc.name, "s0", "graph.elf"), fmt.Sprintf("elf %s %d", enc.name, enc.side))
	}
	writeJSON(t, filepath.Join(dir, "model.json"), map[string]any{"name": "FLUX.2-klein-4B", "steps": 4, "width": size, "height": size})
	writeFile(t, filepath.Join(dir, "entry.json"), "{}")
}

// readManifest reads the graph's manifest in dir.
func readManifest(t *testing.T, dir string) map[string]any {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestMergeAMD(t *testing.T) {
	root := t.TempDir()
	big, small, out := filepath.Join(root, "big"), filepath.Join(root, "small"), filepath.Join(root, "both")
	// 1024: the picture's 4608 rows; 512: 1536, and 4608 and 5632 with references (the same row
	// counts as 1024's picture and its own references')
	fakeSized(t, big, 1024, []int{5632, 6656}, "w")
	fakeSized(t, small, 512, []int{2048, 4608, 5632}, "w")
	if err := MergeAMD(out, big, small); err != nil {
		t.Fatal(err)
	}
	if got := IronSizes(out); !slices.Equal(got, [][2]int{{1024, 1024}, {512, 512}}) {
		t.Errorf("sizes %v", got)
	}
	if _, err := os.Stat(out + ".merging"); err == nil {
		t.Error("the merging folder stays")
	}
	for _, p := range []string{"doubles_n1536", "singles_n1536", "dit_in_n1536", "doubles_n2048", "dit_in_n2048",
		"doubles_n5632", "doubles_n6656", "sizes/512x512/dit_out", "sizes/512x512/vae", "sizes/256x256/vae_enc"} {
		if _, err := os.Stat(filepath.Join(out, p, "manifest.json")); err != nil {
			t.Errorf("%s: %v", p, err)
		}
	}
	for _, p := range []string{"doubles_n4608", "sizes/512x512/vae_enc", "entry.json"} {
		if _, err := os.Stat(filepath.Join(out, p)); err == nil {
			t.Errorf("%s is in the merged export", p)
		}
	}
	// 512's picture's blocks run on 1024's banks; their own files are not copied
	m := readManifest(t, filepath.Join(out, "doubles_n1536"))
	if m["name"] != "doubles_n1536" || m["banks"].(map[string]any)["scratch"].(map[string]any)["image"] != "../doubles/bank_scratch.bin" {
		t.Errorf("doubles_n1536's manifest: %v", m)
	}
	if v := fmt.Sprint(m["variants"]); v != "[map[scratch:../doubles/bank_scratch.bin] map[scratch:../doubles/bank_g1_scratch.bin] map[scratch:../doubles/bank_g2_scratch.bin]]" {
		t.Errorf("variants %s", v)
	}
	if n := m["banks"].(map[string]any)["scratch"].(map[string]any)["bytes"]; fmt.Sprint(n) != "7.63656192e+08" {
		t.Errorf("bytes %v", n) // the number as it was (json reads it as float64 here)
	}
	if b, err := os.ReadFile(filepath.Join(out, "doubles_n1536", "manifest.json")); err != nil || !strings.Contains(string(b), "763656192") {
		t.Errorf("the bank's bytes written otherwise: %s", b)
	}
	if ents, _ := os.ReadDir(filepath.Join(out, "doubles_n1536")); len(ents) != 2 {
		t.Errorf("doubles_n1536 holds %d files, want the graph and its manifest", len(ents))
	}
	if b, _ := os.ReadFile(filepath.Join(out, "doubles_n1536", "graph.elf")); string(b) != "elf doubles 512" {
		t.Errorf("doubles_n1536's ELF %q", b)
	}
	// dit_in keeps its own bank
	if m := readManifest(t, filepath.Join(out, "dit_in_n1536")); m["banks"].(map[string]any)["scratch"].(map[string]any)["image"] != "bank_scratch.bin" {
		t.Errorf("dit_in_n1536's manifest %v", m)
	}
	if _, err := os.Stat(filepath.Join(out, "dit_in_n1536", "bank_scratch.bin")); err != nil {
		t.Error(err)
	}
	// hard links: the merged export takes no room of its own
	a, _ := os.Stat(filepath.Join(out, "doubles", "bank_scratch.bin"))
	b, _ := os.Stat(filepath.Join(big, "doubles", "bank_scratch.bin"))
	if !os.SameFile(a, b) {
		t.Error("the banks are copies")
	}
	var mj struct {
		Width int      `json:"width"`
		Sizes [][2]int `json:"sizes"`
	}
	if bb, err := os.ReadFile(filepath.Join(out, "model.json")); err != nil || json.Unmarshal(bb, &mj) != nil || mj.Width != 1024 || len(mj.Sizes) != 2 {
		t.Errorf("model.json %+v %v", mj, err)
	}
	if bb, _ := os.ReadFile(filepath.Join(big, "model.json")); strings.Contains(string(bb), "sizes") {
		t.Error("the first export's model.json was written through its link")
	}

	// the pipeline's names of the parts at each size
	base, _ := ironConfig(out)
	p := &Pipeline{dir: out, base: base, C: base}
	if p.suffix(4608) != "" || p.suffix(1536) != "_n1536" || p.part("vae") != "vae" || p.encoder(512, 512) != "vae_enc_d2" || p.encoder(1024, 1024) != "vae_enc" {
		t.Errorf("1024: %q %q %q %q", p.suffix(1536), p.part("vae"), p.encoder(512, 512), p.encoder(1024, 1024))
	}
	if !p.HasReferences([]int{1, 2}) && !p.HasReferences(nil) {
		t.Error("1024 runs no references")
	}
	p.C.Width, p.C.Height, p.sized = 512, 512, sizeDir(512, 512)
	if p.part("dit_out") != "sizes/512x512/dit_out" || p.encoder(256, 256) != "sizes/256x256/vae_enc" {
		t.Errorf("512: %q %q", p.part("dit_out"), p.encoder(256, 256))
	}
	// 512 with one reference of its size and two of half: 1024 + 1024 + 2 x 256 + 512 = 3072 rows
	// (none here), nor one of its size (2560); with three of its size, 4608: 1024's picture's blocks; two of
	// half (2048), the encoder of 256 x 256
	if p.HasReferences([]int{1, 2, 2}) || !p.HasReferences([]int{1, 1, 1}) || !p.HasReferences(nil) || !p.HasReferences([]int{2, 2}) || p.HasReferences([]int{1}) {
		t.Errorf("512's references: %v", p.References())
	}
}

// TestMergeAMDRefuses: exports of other weights are not made one, and leave nothing.
func TestMergeAMDRefuses(t *testing.T) {
	root := t.TempDir()
	big, small, out := filepath.Join(root, "big"), filepath.Join(root, "small"), filepath.Join(root, "both")
	fakeSized(t, big, 1024, nil, "w")
	fakeSized(t, small, 512, nil, "x")
	err := MergeAMD(out, big, small)
	if err == nil || !strings.Contains(err.Error(), "differ") {
		t.Errorf("other weights: %v", err)
	}
	for _, p := range []string{out, out + ".merging"} {
		if _, err := os.Stat(p); err == nil {
			t.Errorf("%s made", p)
		}
	}
	// the text alike but the blocks' banks not
	fakeSized(t, small, 512, nil, "w")
	writeFile(t, filepath.Join(small, "doubles", "bank_g1_scratch.bin"), "other")
	if err := MergeAMD(out, big, small); err == nil {
		t.Error("other blocks' banks merged")
	}
}

// TestPrepareAMDMerges: the exports of 512 x 512 and 1024 x 1024 here are made one, and removed,
// when the other was not used lately; else they stay.
func TestPrepareAMDMerges(t *testing.T) {
	noSources(t)
	m := kleinModel(t)
	ctx := context.Background()
	big := filepath.Join(chat.AMDRoot(), AMDExportName(m.Name, 1024))
	small := filepath.Join(chat.AMDRoot(), AMDExportName(m.Name, 512))
	fakeSized(t, big, 1024, nil, "w")
	fakeSized(t, small, 512, nil, "w")
	// 1024's used now (by another process): 512 is drawn from its own
	if got, err := PrepareAMD(ctx, m, 512, 512, nil); err != nil || got != small {
		t.Fatalf("512 with 1024 in use: %s %v", got, err)
	}
	old := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(big, old, old); err != nil {
		t.Fatal(err)
	}
	got, err := PrepareAMD(ctx, m, 512, 512, nil)
	want := filepath.Join(chat.AMDRoot(), AMDMergedName(m.Name, amdSquares...))
	if err != nil || got != want {
		t.Fatalf("512: %s %v, want %s", got, err, want)
	}
	for _, d := range []string{big, small} {
		if _, err := os.Stat(d); err == nil {
			t.Errorf("%s stays", d)
		}
	}
	for _, size := range amdSquares {
		if got, err := PrepareAMD(ctx, m, size[0], size[1], nil); err != nil || got != want {
			t.Errorf("%v: %s %v", size, got, err)
		}
	}
	if n := AMDMergedName(m.Name, [2]int{1024, 1024}, [2]int{512, 512}); n != "flux.2-klein-4b-512+1024.amd-npu2" {
		t.Error(n)
	}
	if n := AMDMergedName(m.Name, AMDSizes...); n != "flux.2-klein-4b-aspects2.amd-npu2" {
		t.Error(n)
	}
	if r := AMDRecipe(m, AMDSizes...); r["args"] != "--size 1024 512 832x1248 880x1168 768x1344 1248x832 1168x880 1344x768" {
		t.Errorf("the recipe of every size: %v", r["args"])
	}

}

// TestPrepareAMDAspects: the export of every size here draws them all and replaces the exports of
// the squares (those not used lately).
func TestPrepareAMDAspects(t *testing.T) {
	noSources(t)
	m := kleinModel(t)
	ctx := context.Background()
	squares := filepath.Join(chat.AMDRoot(), AMDMergedName(m.Name, amdSquares...))
	fakeSized(t, squares, 1024, nil, "w")
	writeJSON(t, filepath.Join(squares, sizeDir(512, 512), "vae", "manifest.json"), map[string]any{})
	old := time.Now().Add(-2 * time.Hour)
	if err := os.Chtimes(squares, old, old); err != nil {
		t.Fatal(err)
	}
	// the export of the aspect ratios before (16:9 at 1360 x 752), which the new one replaces
	before := filepath.Join(chat.AMDRoot(), amdOldAspectsName(m.Name))
	fakeSized(t, before, 1024, nil, "w")
	for _, s := range aspect.Sizes(16)[1:] {
		writeJSON(t, filepath.Join(before, sizeDir(s[0], s[1]), "vae", "manifest.json"), map[string]any{})
	}
	if err := os.Chtimes(before, old, old); err != nil {
		t.Fatal(err)
	}
	all := filepath.Join(chat.AMDRoot(), AMDMergedName(m.Name, AMDSizes...))
	fakeSized(t, all, 1024, nil, "w")
	for _, s := range AMDSizes[1:] {
		writeJSON(t, filepath.Join(all, sizeDir(s[0], s[1]), "vae", "manifest.json"), map[string]any{})
	}
	for _, s := range AMDSizes {
		if got, err := PrepareAMD(ctx, m, s[0], s[1], nil); err != nil || got != all {
			t.Errorf("%v: %s %v, want %s", s, got, err, all)
		}
	}
	for _, d := range []string{squares, before} {
		if _, err := os.Stat(d); err == nil {
			t.Errorf("%s stays", d)
		}
	}
	if w, h, err := aspect.ParseOrSideIn("16:9", Aspects, 16); err != nil || w != 1344 || h != 768 {
		t.Errorf("16:9: %d x %d %v", w, h, err)
	}
}
