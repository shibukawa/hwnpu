package cache

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestKey(t *testing.T) {
	if Key([]byte("ab"), []byte("c")) == Key([]byte("a"), []byte("bc")) {
		t.Fatal("length prefix missing")
	}
	if Key([]byte("x")) != Key([]byte("x")) {
		t.Fatal("not deterministic")
	}
}

func TestStoreAndLookup(t *testing.T) {
	dir := t.TempDir()
	key := Key([]byte("model"))
	if _, ok := Lookup(dir, key, ".blob"); ok {
		t.Fatal("unexpected hit")
	}
	p, err := Store(dir, key, ".blob", func(tmp string) error { return os.WriteFile(tmp, []byte("data"), 0o644) })
	if err != nil {
		t.Fatal(err)
	}
	if got, ok := Lookup(dir, key, ".blob"); !ok || got != p {
		t.Fatalf("lookup %s %v", got, ok)
	}
	// a failing writer leaves nothing behind
	_, err = Store(dir, Key([]byte("other")), ".blob", func(tmp string) error {
		os.WriteFile(tmp, []byte("partial"), 0o644)
		return errors.New("boom")
	})
	if err == nil {
		t.Fatal("expected error")
	}
	entries, _ := os.ReadDir(dir)
	if len(entries) != 1 {
		t.Fatalf("left over files: %v", entries)
	}
}

func TestTreeDigest(t *testing.T) {
	a, b := t.TempDir(), t.TempDir()
	for _, d := range []string{a, b} {
		os.MkdirAll(filepath.Join(d, "sub"), 0o755)
		os.WriteFile(filepath.Join(d, "sub", "f"), []byte("1"), 0o644)
	}
	da, _ := TreeDigest(a)
	db, _ := TreeDigest(b)
	if string(da) != string(db) {
		t.Fatal("same trees differ")
	}
	os.WriteFile(filepath.Join(b, "sub", "f"), []byte("2"), 0o644)
	db, _ = TreeDigest(b)
	if string(da) == string(db) {
		t.Fatal("content change not detected")
	}
	c := t.TempDir()
	if err := CopyTree(a, c); err != nil {
		t.Fatal(err)
	}
	dc, _ := TreeDigest(c)
	if string(dc) != string(da) {
		t.Fatal("copy differs")
	}
}
