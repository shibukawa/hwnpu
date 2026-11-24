package mapped

import (
	"os"
	"path/filepath"
	"testing"
)

func TestOpen(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "f")
	os.WriteFile(p, []byte("hello"), 0o644)
	f, err := Open(p)
	if err != nil || string(f.Bytes()) != "hello" || f.Len() != 5 {
		t.Fatalf("%q, %v", f.Bytes(), err)
	}
	if err := f.Close(); err != nil || f.Bytes() != nil {
		t.Fatalf("close: %v", err)
	}
	f.Close() // twice is fine
	empty := filepath.Join(dir, "e")
	os.WriteFile(empty, nil, 0o644)
	if f, err := Open(empty); err != nil || f.Len() != 0 || f.Close() != nil {
		t.Fatalf("empty: %v", err)
	}
	if _, err := Open(filepath.Join(dir, "none")); err == nil {
		t.Fatal("a missing file opened")
	}
}

func TestAnon(t *testing.T) {
	f, err := Anon(1 << 20)
	if err != nil || f.Len() != 1<<20 {
		t.Fatalf("%d, %v", f.Len(), err)
	}
	b := f.Bytes()
	b[0], b[len(b)-1] = 1, 2
	if b[0] != 1 || b[len(b)-1] != 2 || f.Close() != nil {
		t.Fatal("not writable")
	}
	if f, err := Anon(0); err != nil || f.Len() != 0 || f.Close() != nil {
		t.Fatalf("empty: %v", err)
	}
}
