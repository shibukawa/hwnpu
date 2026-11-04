package cache

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Clone replaces a file by a clone of another with the same bytes, and leaves alone one that
// differs, even only in its last byte; the clone keeps its bytes when the original is written.
func TestClone(t *testing.T) {
	dir := t.TempDir()
	data := bytes.Repeat([]byte("weights!"), 300_000) // over the 1 MiB read size
	write := func(name string, b []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	a, b := write("a.bin", data), write("b.bin", data)
	other := append(bytes.Clone(data[:len(data)-1]), '?')
	c := write("c.bin", other)
	ok, err := Clone(b, a)
	if err != nil {
		t.Fatal(err)
	}
	if ok != canClone {
		t.Fatalf("Clone of the same bytes: %v, want %v", ok, canClone)
	}
	if ok, err := Clone(c, a); ok || err != nil {
		t.Fatalf("Clone of other bytes: %v, %v", ok, err)
	}
	if got, _ := os.ReadFile(b); !bytes.Equal(got, data) {
		t.Fatal("the clone's bytes changed")
	}
	if fi, _ := os.Stat(b); fi.Mode().Perm() != 0o600 {
		t.Errorf("the clone's mode is %v, want 0600", fi.Mode().Perm())
	}
	write("a.bin", []byte("rewritten"))
	if got, _ := os.ReadFile(b); !bytes.Equal(got, data) {
		t.Fatal("writing the original changed the clone")
	}
	if got, _ := os.ReadFile(c); !bytes.Equal(got, other) {
		t.Fatal("the file that differs changed")
	}
}

// Prune removes the entries whose source is gone and those unused for longer than its age, and
// keeps the rest: a looked up entry counts as used, and what is not an entry stays.
func TestPrune(t *testing.T) {
	dir := t.TempDir()
	old := time.Now().Add(-48 * time.Hour)
	entry := func(name string, age time.Time) string {
		k := Key([]byte(name))
		p, err := Store(dir, k, ".mlmodelc", func(tmp string) error {
			if err := os.MkdirAll(filepath.Join(tmp, "weights"), 0o755); err != nil {
				return err
			}
			return os.WriteFile(filepath.Join(tmp, "weights", "weight.bin"), []byte(name), 0o644)
		})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.Chtimes(p, age, age); err != nil {
			t.Fatal(err)
		}
		return k
	}
	src := filepath.Join(t.TempDir(), "m.mlpackage")
	os.Mkdir(src, 0o755)
	gone := filepath.Join(t.TempDir(), "deleted.mlpackage")
	recent, stale, used, orphan, kept := entry("recent", time.Now()), entry("stale", old), entry("used", old), entry("orphan", time.Now()), entry("kept", time.Now())
	SetSource(dir, orphan, gone)
	SetSource(dir, kept, src)
	if _, ok := Lookup(dir, used, ".mlmodelc"); !ok {
		t.Fatal("no entry")
	}
	os.MkdirAll(filepath.Join(dir, "prompts"), 0o755)
	tmp := filepath.Join(dir, ".tmp-abandoned")
	os.Mkdir(tmp, 0o755)
	os.Chtimes(tmp, old, old)
	n, size, err := Prune(dir, 24*time.Hour)
	if err != nil {
		t.Fatal(err)
	}
	for k, want := range map[string]bool{recent: true, stale: false, used: true, orphan: false, kept: true} {
		if _, ok := Lookup(dir, k, ".mlmodelc"); ok != want {
			t.Errorf("entry %s… kept %v, want %v", k[:8], ok, want)
		}
	}
	if _, ok := Source(dir, orphan); ok {
		t.Error("the source record of a removed entry stays")
	}
	if _, err := os.Stat(filepath.Join(dir, "prompts")); err != nil {
		t.Error("a directory that is not an entry was removed")
	}
	if _, err := os.Stat(tmp); err == nil {
		t.Error("an abandoned temporary entry stays")
	}
	if n != 3 || size != int64(len("stale")+len("orphan")) {
		t.Errorf("removed %d entries of %d bytes, want 3 (with the temporary one) of %d", n, size, len("stale")+len("orphan"))
	}
}
