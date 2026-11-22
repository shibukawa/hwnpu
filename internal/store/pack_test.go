package store

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// TestPackedEntries puts an export for AMD's NPU in a source as one pack beside entry.json
// (replacing the loose files of one put before), makes it from there unpacked, goes on from a
// pack cut off, and refuses a pack that changed.
func TestPackedEntries(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	exp := filepath.Join(t.TempDir(), "m.amd-npu2")
	write(t, filepath.Join(exp, "manifest.json"), `{"version":1}`)
	write(t, filepath.Join(exp, "bank.bin"), strings.Repeat("weights", 1000))
	write(t, filepath.Join(exp, "check", "x.bin"), "check")
	recipe := map[string]string{"model": "m", "commit": "abc"}
	rel := EntryPath("m", cache.AMDTarget, "m.amd-npu2", VersionDir(0, false, Fingerprint(recipe)))
	if !Packs(rel) || !Packs(EntryPath("m", cache.CoreAITarget, "m.coreai", "v1")) ||
		Packs(EntryPath("m", "any", "tables.safetensors", "v1")) || Packs(EntryPath("m", "coreai-h14g", "m.coreai", "v1")) {
		t.Fatal("Packs")
	}
	e := describeAt(t, exp, rel, recipe)
	root := t.TempDir()
	far := Source{Store: Dir{Name: "far", Root: root}, Writeback: true}
	// loose files an hwnpu before put there
	write(t, filepath.Join(root, filepath.FromSlash(rel), "bank.bin"), "old")
	if err := PutEntry(ctx, far, exp); err != nil {
		t.Fatal(err)
	}
	keys, err := far.Store.List(ctx, rel+"/")
	if err != nil {
		t.Fatal(err)
	}
	slices.Sort(keys)
	if want := []string{rel + "/" + EntryFile, rel + "/" + PackFile}; !slices.Equal(keys, want) {
		t.Fatalf("in the source: %v, want %v", keys, want)
	}
	got, err := ReadEntryAt(ctx, far, rel)
	if err != nil || got.Pack == nil || got.Pack.Path != PackFile || len(got.Files) != 3 || got.Bytes() != e.Bytes() {
		t.Fatalf("its entry: %+v %v", got, err)
	}
	if got.Pack.Size >= e.Bytes() {
		t.Errorf("a pack of %d bytes for %d", got.Pack.Size, e.Bytes())
	}
	if e2, _ := ReadEntry(exp); e2.Pack != nil {
		t.Error("the entry here was given the pack")
	}
	out := filepath.Join(t.TempDir(), "m.amd-npu2")
	if _, err := FetchEntry(ctx, []Source{far}, rel, func(Entry) bool { return true }, out, nil); err != nil {
		t.Fatal(err)
	}
	if err := Verify(out, true); err != nil {
		t.Fatal(err)
	}
	if b, _ := os.ReadFile(filepath.Join(out, "check", "x.bin")); string(b) != "check" {
		t.Errorf("check/x.bin: %q", b)
	}
	if _, err := os.Stat(filepath.Join(out, packName)); err == nil {
		t.Error("the pack is kept here")
	}
	// a download cut off: half the pack in dst.part, and a file unpacked from another
	cut := filepath.Join(t.TempDir(), "m.amd-npu2")
	pack, _ := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel), PackFile))
	write(t, filepath.Join(cut+".part", packName+".part"), string(pack[:len(pack)/2]))
	write(t, filepath.Join(cut+".part", "stray.bin"), "x")
	if _, err := FetchEntry(ctx, []Source{far}, rel, func(Entry) bool { return true }, cut, nil); err != nil {
		t.Fatal(err)
	}
	if err := Verify(cut, true); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(cut, "stray.bin")); err == nil {
		t.Error("a file not of the entry was kept")
	}
	// the same key again: nothing is put
	if err := PutEntry(ctx, far, exp); err != nil {
		t.Fatal(err)
	}
	// a pack that changed in the source is found out
	pack[len(pack)/2] ^= 0xff
	write(t, filepath.Join(root, filepath.FromSlash(rel), PackFile), string(pack))
	if _, err := FetchEntry(ctx, []Source{far}, rel, func(Entry) bool { return true }, out+"2", nil); err == nil {
		t.Error("a changed pack was taken")
	}
}

// TestUnpackChecks refuses a pack that holds a file its entry does not list, or lacks one, or
// whose file differs.
func TestUnpackChecks(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	ctx := context.Background()
	dir := t.TempDir()
	write(t, filepath.Join(dir, "a.bin"), "aaaa")
	write(t, filepath.Join(dir, "b.bin"), "bbbb")
	e, err := Describe(dir, Entry{Kind: "iron-lm", Recipe: map[string]string{"m": "x"}})
	if err != nil {
		t.Fatal(err)
	}
	arch := filepath.Join(t.TempDir(), "p.tar.zst")
	if _, err := packTo(ctx, dir, e, arch); err != nil {
		t.Fatal(err)
	}
	if err := unpack(ctx, arch, e, t.TempDir(), nil); err != nil {
		t.Fatal(err)
	}
	fewer := e
	fewer.Files = e.Files[:1]
	if err := unpack(ctx, arch, fewer, t.TempDir(), nil); err == nil {
		t.Error("a file the entry does not list was taken")
	}
	more := e
	more.Files = append(slices.Clone(e.Files), File{Path: "c.bin", Size: 1, SHA256: strings.Repeat("0", 64)})
	if err := unpack(ctx, arch, more, t.TempDir(), nil); err == nil {
		t.Error("a pack without a file of the entry was taken")
	}
	other := e
	other.Files = slices.Clone(e.Files)
	other.Files[0].SHA256 = strings.Repeat("0", 64)
	if err := unpack(ctx, arch, other, t.TempDir(), nil); err == nil {
		t.Error("a file of another SHA-256 was taken")
	}
}
