package llm

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// ShareWeights makes the weight files with the same bytes clones of the first of them, in packages
// and IR alike, and leaves the others: a file of the same size with other bytes, and a .bin that is not an IR's.
func TestShareWeights(t *testing.T) {
	dir := t.TempDir()
	w := bytes.Repeat([]byte{1, 2, 3, 4}, 1000)
	other := bytes.Repeat([]byte{4, 3, 2, 1}, 1000)
	put := func(path string, b []byte) {
		os.MkdirAll(filepath.Dir(path), 0o755)
		if err := os.WriteFile(path, b, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	q8, q64, q1 := filepath.Join(dir, "m-q8.mlpackage"), filepath.Join(dir, "m-q64.mlpackage"), filepath.Join(dir, "m-q1.mlpackage")
	put(WeightFile(q8), w)
	put(WeightFile(q64), w)
	put(WeightFile(q1), other)
	put(filepath.Join(dir, "m-q8.xml"), []byte("<net/>"))
	put(WeightFile(filepath.Join(dir, "m-q8.xml")), w)
	put(filepath.Join(dir, "stray.bin"), w)
	before := map[string]os.FileInfo{}
	for _, f := range []string{WeightFile(q8), WeightFile(q64), WeightFile(q1), filepath.Join(dir, "m-q8.bin"), filepath.Join(dir, "stray.bin")} {
		before[f], _ = os.Stat(f)
	}
	ShareWeights(dir)
	replaced := func(f string) bool {
		fi, err := os.Stat(f)
		if err != nil {
			t.Fatal(err)
		}
		return !os.SameFile(fi, before[f])
	}
	if !cache.CanCloneIn(dir) {
		t.Skip("the file system cannot clone")
	}
	// the first of each set (in name order) stays, the others become its clones
	if replaced(WeightFile(q64)) || !replaced(WeightFile(q8)) || !replaced(filepath.Join(dir, "m-q8.bin")) {
		t.Error("the files with the same bytes were not made clones of the first")
	}
	if replaced(WeightFile(q1)) || replaced(filepath.Join(dir, "stray.bin")) {
		t.Error("a file with other bytes, or one that is no program's, was replaced")
	}
	for _, f := range []string{WeightFile(q8), filepath.Join(dir, "m-q8.bin")} {
		if b, _ := os.ReadFile(f); !bytes.Equal(b, w) {
			t.Errorf("%s changed", f)
		}
	}
}
