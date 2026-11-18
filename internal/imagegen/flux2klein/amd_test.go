package flux2klein

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

func kleinModel(t *testing.T) chat.Model {
	t.Helper()
	m, ok := chat.Lookup("FLUX.2-klein-4B")
	if !ok {
		t.Fatal("FLUX.2-klein-4B is not in the definitions")
	}
	return m
}

// fakeIronExport writes the manifests isIronExport and IronSizes look for into dir, an export
// that draws sizes (the first laid out as an export of its own, the others in SizesDir), with mark
// in model.json.
func fakeIronExport(t *testing.T, dir, mark string, sizes ...[2]int) {
	t.Helper()
	write := func(p string, b []byte) {
		t.Helper()
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for _, p := range []string{"singles", "mods", "ctx", "text"} {
		write(filepath.Join(dir, p, "manifest.json"), []byte("{}"))
	}
	doubles, err := json.Marshal(map[string]any{"version": 1, "kind": "graph", "name": "doubles",
		"graph": map[string]any{"elf": "graph.elf", "kernels": []map[string]any{{"name": "k"}}}, "meta": map[string]any{"config": map[string]any{"width": sizes[0][0], "height": sizes[0][1]}}})
	if err != nil {
		t.Fatal(err)
	}
	write(filepath.Join(dir, "doubles", "manifest.json"), doubles)
	for _, s := range sizes[1:] {
		write(filepath.Join(dir, sizeDir(s[0], s[1]), "vae", "manifest.json"), []byte("{}"))
	}
	write(filepath.Join(dir, "model.json"), []byte(mark))
}

// noSources leaves only the machine's cache (a new root) and no IRON, so that nothing is
// downloaded or exported.
func noSources(t *testing.T) {
	t.Helper()
	t.Setenv(cache.RootEnv, t.TempDir())
	t.Setenv("HOME", t.TempDir()) // no ~/models
	t.Setenv(chat.IRONEnv, filepath.Join(t.TempDir(), "no-iron"))
	ss, hf := shared.Sources(), shared.HuggingFace()
	shared.SetSources(nil, false)
	t.Cleanup(func() { shared.SetSources(ss, hf) })
}

func TestAMDExportName(t *testing.T) {
	if got := AMDExportName("FLUX.2-klein-4B"); got != "flux.2-klein-4b-aspects2.amd-npu2" {
		t.Errorf("%s, want flux.2-klein-4b-aspects2.amd-npu2", got)
	}
}

// TestAMDImportedPlace: the export made by hand of FLUX.2 klein that hwnpu-dev cache import put
// in a source is found where it put it, and the first run's is another entry, of every size.
func TestAMDImportedPlace(t *testing.T) {
	m := kleinModel(t)
	name := AMDExportName(m.Name)
	want := "converted/FLUX.2-klein-4B/amd-xdna2/flux.2-klein-4b-aspects2.amd-npu2/v0-89545ca2"
	if got := chat.PlaceEntry(chat.ImportKind, m, cache.AMDTarget, name, chat.ImportedRecipe(m, cache.AMDTarget, name)).Path; got != want {
		t.Errorf("%s, want %s", got, want)
	}
	r := AMDRecipe(m, amdTestSizes...)
	if r == nil || r["args"] != "--size 1024 512 832x1248 880x1168 768x1344 1248x832 1168x880 1344x768" || r["tool"] != AMDTool {
		t.Fatalf("recipe %v", r)
	}
	if r["sha256:transformer/diffusion_pytorch_model.safetensors"] == "" {
		t.Error("the recipe has not the DiT's SHA-256")
	}
}

// TestPrepareAMDHere runs the export here by its name without a source or IRON, and refuses a
// size it does not export.
func TestPrepareAMDHere(t *testing.T) {
	noSources(t)
	m := kleinModel(t)
	ctx := context.Background()
	if _, err := PrepareAMD(ctx, m, amdTestSizes, 768, 768, nil); err == nil || !strings.Contains(err.Error(), "--size 768x768") {
		t.Errorf("768: %v", err)
	}
	legacy := filepath.Join(chat.AMDRoot(), strings.ToLower(m.Name)+".amd-npu2")
	fakeIronExport(t, legacy, "{}", [2]int{1024, 1024}, [2]int{752, 1360}, [2]int{1360, 752})
	if _, err := PrepareAMD(ctx, m, amdTestSizes, 768, 1344, nil); err == nil || !strings.Contains(err.Error(), chat.IRONEnv) {
		t.Errorf("768 x 1344 reused the legacy export: %v", err)
	}
	os.RemoveAll(legacy)
	dir := filepath.Join(chat.AMDRoot(), AMDExportName(m.Name))
	fakeIronExport(t, dir, "{}", [2]int{1024, 1024})
	if got, err := PrepareAMD(ctx, m, amdTestSizes, 1024, 1024, nil); err != nil || got != dir {
		t.Errorf("1024 of the older export: %s %v, want %s", got, err, dir)
	}
	if _, err := PrepareAMD(ctx, m, amdTestSizes, 832, 1248, nil); err == nil || !strings.Contains(err.Error(), chat.IRONEnv) {
		t.Errorf("832 x 1248 with an export of 1024 x 1024 alone and no IRON: %v", err)
	}
	os.RemoveAll(dir)
	fakeIronExport(t, dir, "{}", amdTestSizes...)
	for _, s := range amdTestSizes {
		if got, err := PrepareAMD(ctx, m, amdTestSizes, s[0], s[1], nil); err != nil || got != dir {
			t.Errorf("%v: %s %v, want %s", s, got, err, dir)
		}
	}
}

// TestPrepareAMDImported takes an export made by hand that a source holds (hwnpu-dev cache
// import), without the checkpoint or IRON, into the cache's entries.
func TestPrepareAMDImported(t *testing.T) {
	noSources(t)
	m := kleinModel(t)
	ctx := context.Background()
	share := t.TempDir()
	s, err := shared.ParseSource(shared.SourceEnv, "file://"+filepath.ToSlash(share))
	if err != nil {
		t.Fatal(err)
	}
	name := AMDExportName(m.Name)
	hand := filepath.Join(t.TempDir(), name)
	fakeIronExport(t, hand, `{"by":"hand"}`, amdTestSizes...)
	e := chat.PlaceEntry(chat.ImportKind, m, cache.AMDTarget, name, chat.ImportedRecipe(m, cache.AMDTarget, name))
	if _, err := shared.Describe(hand, e); err != nil {
		t.Fatal(err)
	}
	if err := shared.PutEntry(ctx, s, hand); err != nil {
		t.Fatal(err)
	}
	shared.SetSources([]shared.Source{s}, false)
	got, err := PrepareAMD(ctx, m, amdTestSizes, 512, 512, nil)
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(chat.AMDRoot(), name); got != want {
		t.Errorf("%s, want %s", got, want)
	}
	if b, err := os.ReadFile(filepath.Join(got, "model.json")); err != nil || string(b) != `{"by":"hand"}` {
		t.Errorf("model.json %q %v", b, err)
	}
	if _, err := os.Stat(filepath.Join(chat.RepoDir(m.Repo), "transformer")); err == nil {
		t.Error("the checkpoint was downloaded")
	}
}

// amdTestSizes are the AMD export's sizes (models.yaml npu.amd), 1024 x 1024 first.
var amdTestSizes = [][2]int{{1024, 1024}, {512, 512}, {832, 1248}, {880, 1168}, {768, 1344}, {1248, 832}, {1168, 880}, {1344, 768}}
