package cache

import (
	"bytes"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// ClonePrefix shares the longest common start of two files in whole blocks: of a file that is the
// start of a longer one, and of files of one size that differ past the middle. The bytes, mode and
// time of the file replaced stay, a second call finds the start shared, and files with too short a
// common start are left alone.
func TestClonePrefix(t *testing.T) {
	dir := t.TempDir()
	rng := rand.New(rand.NewPCG(1, 2))
	bank := make([]byte, 3<<20+123) // not whole blocks
	for i := range bank {
		bank[i] = byte(rng.Uint32())
	}
	put := func(name string, b []byte) string {
		p := filepath.Join(dir, name)
		if err := os.WriteFile(p, b, 0o640); err != nil {
			t.Fatal(err)
		}
		return p
	}
	longer := append(bytes.Clone(bank), bytes.Repeat([]byte("tower"), 1<<18)...)
	tail := bytes.Clone(bank)
	tail[2<<20+5000]++ // the same size, the same bytes up to past 2 MiB
	short := bytes.Clone(bank)
	short[1000]++ // the same bytes for less than a block
	text, mm, other, early := put("text.bin", bank), put("mm.bin", longer), put("tail.bin", tail), put("short.bin", short)
	old := time.Now().Add(-time.Hour).Truncate(time.Second)
	os.Chtimes(text, old, old)

	got, err := ClonePrefix(text, mm)
	if err != nil {
		t.Fatal(err)
	}
	if !CanCloneIn(dir) || rangeBlock(mustOpen(t, dir)) == 0 {
		if got != 0 {
			t.Fatalf("%d bytes shared where the system cannot clone ranges", got)
		}
		t.Skip("the file system cannot clone ranges")
	}
	bs := rangeBlock(mustOpen(t, dir))
	if want := int64(len(bank)) / bs * bs; got != want {
		t.Errorf("shared %d bytes of the text's start, want %d", got, want)
	}
	if b, _ := os.ReadFile(text); !bytes.Equal(b, bank) {
		t.Fatal("the text's bytes changed")
	}
	if fi, _ := os.Stat(text); fi.Mode().Perm() != 0o640 || !fi.ModTime().Equal(old) {
		t.Errorf("mode %v and time %v, want 0640 and %v", fi.Mode().Perm(), fi.ModTime(), old)
	}
	if again, _ := ClonePrefix(text, mm); again != 0 {
		t.Errorf("the shared start was shared again: %d bytes", again)
	}

	got, _ = ClonePrefix(other, mm)
	if want := int64(2<<20+5000) / bs * bs; got != want {
		t.Errorf("shared %d bytes of the file that differs past 2 MiB, want %d", got, want)
	}
	if b, _ := os.ReadFile(other); !bytes.Equal(b, tail) {
		t.Fatal("the bytes past the common start changed")
	}
	if got, _ := ClonePrefix(early, mm); got != 0 {
		t.Errorf("shared %d bytes of a file that differs in its first block", got)
	}
	if left, _ := filepath.Glob(filepath.Join(dir, ".*")); len(left) > 0 {
		t.Errorf("temporary files left: %v", left)
	}
}

func mustOpen(t *testing.T, p string) *os.File {
	f, err := os.Open(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.Close() })
	return f
}
