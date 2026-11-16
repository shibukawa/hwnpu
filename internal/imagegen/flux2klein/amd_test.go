package flux2klein

import (
	"context"
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

// fakeIronExport writes the manifests isIronExport looks for into dir, with mark in model.json.
func fakeIronExport(t *testing.T, dir, mark string) {
	t.Helper()
	for _, p := range []string{"doubles", "singles", "mods", "ctx", "text"} {
		if err := os.MkdirAll(filepath.Join(dir, p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, p, "manifest.json"), []byte("{}"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "model.json"), []byte(mark), 0o644); err != nil {
		t.Fatal(err)
	}
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
	for size, want := range map[int]string{1024: "flux.2-klein-4b.amd-npu2", 512: "flux.2-klein-4b-512.amd-npu2"} {
		if got := AMDExportName("FLUX.2-klein-4B", size); got != want {
			t.Errorf("%d: %s, want %s", size, got, want)
		}
	}
}

// TestAMDImportedPlace: the exports made by hand of FLUX.2 klein that hwnpu-dev cache import put
// in a source are found where it put them (the folders of a source that holds both sizes).
func TestAMDImportedPlace(t *testing.T) {
	m := kleinModel(t)
	for size, want := range map[int]string{
		1024: "converted/FLUX.2-klein-4B/amd-xdna2/flux.2-klein-4b.amd-npu2/v0-2d2e30e3",
		512:  "converted/FLUX.2-klein-4B/amd-xdna2/flux.2-klein-4b-512.amd-npu2/v0-d9fa89fa",
	} {
		name := AMDExportName(m.Name, size)
		if got := chat.PlaceEntry(chat.ImportKind, m, cache.AMDTarget, name, chat.ImportedRecipe(m, cache.AMDTarget, name)).Path; got != want {
			t.Errorf("%d: %s, want %s", size, got, want)
		}
	}
	// the first run's export is another entry, of its size
	r512, r1024 := AMDRecipe(m, 512), AMDRecipe(m, 1024)
	if r512 == nil || r512["args"] != "--size 512" || r1024["args"] != "--size 1024" || r512["tool"] != AMDTool {
		t.Fatalf("recipes %v, %v", r512, r1024)
	}
	if r512["sha256:transformer/diffusion_pytorch_model.safetensors"] == "" {
		t.Error("the recipe has not the DiT's SHA-256")
	}
	if a, b := shared.Fingerprint(r512), shared.Fingerprint(r1024); a == b {
		t.Error("the sizes' recipes are the same")
	}
}

// TestPrepareAMDHere runs an export here by its name (the older NAME-1024 too) without a source or
// IRON, and refuses a size it does not export.
func TestPrepareAMDHere(t *testing.T) {
	noSources(t)
	m := kleinModel(t)
	ctx := context.Background()
	if _, err := PrepareAMD(ctx, m, 768, nil); err == nil || !strings.Contains(err.Error(), "--size 768") {
		t.Errorf("768: %v", err)
	}
	old := filepath.Join(chat.AMDRoot(), "flux.2-klein-4b-1024.amd-npu2")
	fakeIronExport(t, old, "{}")
	if got, err := PrepareAMD(ctx, m, 1024, nil); err != nil || got != old {
		t.Errorf("1024: %s %v, want %s", got, err, old)
	}
	dir := filepath.Join(chat.AMDRoot(), AMDExportName(m.Name, 1024))
	fakeIronExport(t, dir, "{}")
	if got, err := PrepareAMD(ctx, m, 1024, nil); err != nil || got != dir {
		t.Errorf("1024: %s %v, want %s", got, err, dir)
	}
	// none of 512 x 512 here, none in a source, no IRON: what to do
	_, err := PrepareAMD(ctx, m, 512, nil)
	if err == nil || !strings.Contains(err.Error(), chat.IRONEnv) {
		t.Errorf("512 with no export and no IRON: %v", err)
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
	name := AMDExportName(m.Name, 512)
	hand := filepath.Join(t.TempDir(), name)
	fakeIronExport(t, hand, `{"by":"hand"}`)
	e := chat.PlaceEntry(chat.ImportKind, m, cache.AMDTarget, name, chat.ImportedRecipe(m, cache.AMDTarget, name))
	if _, err := shared.Describe(hand, e); err != nil {
		t.Fatal(err)
	}
	if err := shared.PutEntry(ctx, s, hand); err != nil {
		t.Fatal(err)
	}
	shared.SetSources([]shared.Source{s}, false)
	got, err := PrepareAMD(ctx, m, 512, nil)
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
