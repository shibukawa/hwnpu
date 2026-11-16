package chat

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

func handExport(t *testing.T, dir, mark string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"manifest.json", "bank.bin"} {
		if err := os.WriteFile(filepath.Join(dir, f), []byte(mark+f), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// TestAMDNamed: an export made by hand in the cache's entries is put in the source written back to
// on its first use (once: described, then left), another machine takes it from there into its
// entries without making it, and its next run takes that copy; one in ~/models is not put.
func TestAMDNamed(t *testing.T) {
	ctx := context.Background()
	share := t.TempDir()
	if _, err := withSources(t, []string{"file://" + filepath.ToSlash(share) + "?writeback=true"}, "version: 1\nmodels:\n  LFM3-1.7B:\n    machine: 8\n"); err != nil {
		t.Fatal(err)
	}
	m := modelNamed("LFM3-1.7B")
	t.Setenv("HOME", t.TempDir()) // ~/models
	names := []string{"lfm3-1.7b.amd-npu2", "lfm3-1.7b-1024.amd-npu2"}
	isDir := func(d string) bool { fi, err := os.Stat(d); return err == nil && fi.IsDir() }

	// made by hand here, under the second name
	here := filepath.Join(AMDRoot(), names[1])
	handExport(t, here, "here:")
	if got := AMDNamed(ctx, m, names, isDir, nil); got != here {
		t.Fatalf("%s, want %s", got, here)
	}
	shared.Flush(ctx)
	es, _ := shared.ListEntries(ctx, shared.Sources()[0])
	if len(es) != 1 || !strings.Contains(es[0], "/"+names[1]+"/") {
		t.Fatalf("entries in the source: %v", es)
	}
	e, err := shared.ReadEntryAt(ctx, shared.Sources()[0], es[0])
	if err != nil || e.Kind != ImportKind || e.Recipe["export"] != names[1] {
		t.Fatalf("the entry: %+v %v", e, err)
	}
	// described once: the next run reads its entry.json and puts nothing
	before, _ := os.Stat(filepath.Join(here, shared.EntryFile))
	AMDNamed(ctx, m, names, isDir, nil)
	after, _ := os.Stat(filepath.Join(here, shared.EntryFile))
	if before == nil || after == nil || !before.ModTime().Equal(after.ModTime()) {
		t.Error("described again")
	}

	// another machine: from the source, into its entries, then that copy
	t.Setenv(cache.RootEnv, t.TempDir())
	got := AMDNamed(ctx, m, names, isDir, nil)
	if want := filepath.Join(AMDRoot(), names[1]); got != want {
		t.Fatalf("%s, want %s", got, want)
	}
	if b, err := os.ReadFile(filepath.Join(got, "bank.bin")); err != nil || string(b) != "here:bank.bin" {
		t.Fatalf("bank.bin %q %v", b, err)
	}
	os.RemoveAll(share) // the source is gone: the copy here runs
	if again := AMDNamed(ctx, m, names, isDir, nil); again != got {
		t.Errorf("the second run: %s, want %s", again, got)
	}

	// one in ~/models is run, and not put (~/models is read, not written)
	t.Setenv(cache.RootEnv, t.TempDir())
	os.MkdirAll(share, 0o755)
	legacy := filepath.Join(cache.LegacyExports(), names[0])
	handExport(t, legacy, "legacy:")
	if got := AMDNamed(ctx, m, names, isDir, nil); got != legacy {
		t.Fatalf("%s, want %s", got, legacy)
	}
	shared.Flush(ctx)
	if _, err := os.Stat(filepath.Join(legacy, shared.EntryFile)); err == nil {
		t.Error("an entry.json written in ~/models")
	}
	if es, _ := shared.ListEntries(ctx, shared.Sources()[0]); len(es) != 0 {
		t.Errorf("put from ~/models: %v", es)
	}
	// none here or there
	if got := AMDNamed(ctx, m, []string{"other.amd-npu2"}, isDir, nil); got != "" {
		t.Errorf("%s for none", got)
	}
}
