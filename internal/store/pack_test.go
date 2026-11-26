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
	write(t, filepath.Join(exp, "sub", "x.bin"), "check")
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
	if b, _ := os.ReadFile(filepath.Join(out, "sub", "x.bin")); string(b) != "check" {
		t.Errorf("sub/x.bin: %q", b)
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

// TestDigest: an entry's digest is of its files alone, in any order, not of its version, path,
// pack or recipe.
func TestDigest(t *testing.T) {
	a := Entry{Key: "k", Path: "converted/m/t/e/v0-1", Recipe: map[string]string{"tools": "1"},
		Files: []File{{Path: "b", Size: 2, SHA256: "BB"}, {Path: "a", Size: 1, SHA256: "aa"}}}
	b := Entry{Key: "j", Version: 2, Path: "converted/m/t/e/v2", Pack: &File{Path: PackFile},
		Files: []File{{Path: "a", Size: 1, SHA256: "aa"}, {Path: "b", Size: 2, SHA256: "bb"}}}
	if a.Digest() != b.Digest() || !strings.HasPrefix(a.Digest(), DigestPrefix) {
		t.Errorf("%s != %s", a.Digest(), b.Digest())
	}
	b.Files[1].SHA256 = "bc"
	if a.Digest() == b.Digest() {
		t.Error("the digest of other bytes is the same")
	}
}

// putVersions puts the export exp as version 1, then as version 2 with the same files (a recipe
// raised), then as version 3 with a file changed, in s; it returns the three folders.
func putVersions(t *testing.T, s Source, exp string) [3]string {
	t.Helper()
	ctx := context.Background()
	var rels [3]string
	for i := range rels {
		v := i + 1
		if v == 3 {
			write(t, filepath.Join(exp, "bank.bin"), strings.Repeat("other", 1000))
		}
		rels[i] = EntryPath("m", cache.AMDTarget, "m.amd-npu2", VersionDir(v, false, ""))
		recipe := map[string]string{"model": "m", "version": string(rune('0' + v))}
		e := describeAt(t, exp, rels[i], recipe)
		e.Version = v
		if err := WriteEntry(exp, e); err != nil {
			t.Fatal(err)
		}
		if err := PutEntry(ctx, s, exp); err != nil {
			t.Fatal(err)
		}
		got, err := ReadEntryAt(ctx, s, rels[i])
		if err != nil || got.Pack == nil || got.Version != v {
			t.Fatalf("v%d: %+v %v", v, got, err)
		}
		out := filepath.Join(t.TempDir(), "m.amd-npu2")
		if _, err := FetchEntry(ctx, []Source{s}, rels[i], func(Entry) bool { return true }, out, nil); err != nil {
			t.Fatalf("v%d: %v", v, err)
		}
		if err := Verify(out, true); err != nil {
			t.Fatalf("v%d: %v", v, err)
		}
	}
	return rels
}

func testExport(t *testing.T) string {
	exp := filepath.Join(t.TempDir(), "m.amd-npu2")
	write(t, filepath.Join(exp, "manifest.json"), `{"version":1}`)
	write(t, filepath.Join(exp, "bank.bin"), strings.Repeat("weights", 1000))
	write(t, filepath.Join(exp, "sub", "graph.elf"), "elf")
	return exp
}

// TestCopyPack: a version with the same files as one in the source takes its pack by a copy in
// the source (a hard link in a folder), not packed and put again; one with other files is packed.
func TestCopyPack(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	root := t.TempDir()
	s := Source{Store: Dir{Name: "far", Root: root}, Writeback: true}
	rels := putVersions(t, s, testExport(t))
	ino := func(rel string) os.FileInfo {
		fi, err := os.Stat(filepath.Join(root, filepath.FromSlash(rel), PackFile))
		if err != nil {
			t.Fatal(err)
		}
		return fi
	}
	if !os.SameFile(ino(rels[0]), ino(rels[1])) {
		t.Error("v2 of the same files was packed again, not linked to v1's pack")
	}
	if os.SameFile(ino(rels[0]), ino(rels[2])) {
		t.Error("v3 of other files took v1's pack")
	}
}

// TestCopyPackWebDAV: over WebDAV the pack of the same files is copied on the server (COPY): no
// PUT of it.
func TestCopyPackWebDAV(t *testing.T) {
	t.Setenv(cache.RootEnv, t.TempDir())
	d := newDAV(t)
	s := Source{Store: openDAV(t, d), Writeback: true}
	exp := testExport(t)
	ctx := context.Background()
	rels := putVersions(t, s, exp)
	if d.methods["PUT"].Load() == 0 {
		t.Fatal("no PUT counted")
	}
	if d.methods["COPY"].Load() != 1 {
		t.Errorf("%d COPY, want 1 (v2)", d.methods["COPY"].Load())
	}
	// v2's folder holds its pack and entry.json alone
	keys, err := s.Store.List(ctx, rels[1]+"/")
	slices.Sort(keys)
	if want := []string{rels[1] + "/" + EntryFile, rels[1] + "/" + PackFile}; err != nil || !slices.Equal(keys, want) {
		t.Errorf("v2 holds %v (%v), want %v", keys, err, want)
	}
	// v2's pack is a copy of its own: v1's going leaves it
	if err := s.Store.Delete(ctx, rels[0]+"/"+PackFile); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "m.amd-npu2")
	if _, err := FetchEntry(ctx, []Source{s}, rels[1], func(Entry) bool { return true }, out, nil); err != nil {
		t.Fatalf("v2 after v1's pack was removed: %v", err)
	}
}
