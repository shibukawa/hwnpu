package coreai

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// fakePackage writes an .mlpackage of a model file and a weight file of the given blobs.
func fakePackage(t *testing.T, dir, name, model string, blobs ...[]byte) string {
	t.Helper()
	p := filepath.Join(dir, name+".mlpackage")
	w := weightFile(p)
	if err := os.MkdirAll(filepath.Dir(w), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(p, "model.mil"), []byte(model), 0o644); err != nil {
		t.Fatal(err)
	}
	bw, err := mlpackage.NewBlobWriter(w)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range blobs {
		if _, err := bw.Add(mlpackage.BlobUInt8, b); err != nil {
			t.Fatal(err)
		}
	}
	if err := bw.Close(); err != nil {
		t.Fatal(err)
	}
	return p
}

func fill(n int, b ...byte) []byte { return bytes.Repeat(b, n/len(b)) }

func TestResolveGroups(t *testing.T) {
	t.Setenv("HWNPU_CACHE", t.TempDir()) // the locks
	models, cdir := t.TempDir(), t.TempDir()
	w := [][]byte{fill(groupMin, 1, 2), fill(groupMin/2, 3), fill(groupMin/4, 4), fill(4096, 5)}
	other := [][]byte{fill(groupMin, 2, 1), fill(groupMin/2, 3), fill(groupMin/4, 4), fill(4096, 5)}
	q8 := fakePackage(t, models, "m-q8.part1", "q8", w...)
	q128 := fakePackage(t, models, "m-q128.part1", "q128", w...)
	part2 := fakePackage(t, models, "m-q8.part2", "q8 part2", other...) // the same sizes, the largest other bytes
	attn := fakePackage(t, models, "m-q8.attn", "attn", []byte{0})
	attn2 := fakePackage(t, models, "m-q128.attn", "attn 128", []byte{0})
	// an embedder's: its largest weights, and a table of the sequence's length
	e128 := fakePackage(t, models, "e-s128", "s128", fill(groupMin, 7, 8), fill(groupMin/2, 9), fill(groupMin/4, 10), fill(128*64, 11))
	e1024 := fakePackage(t, models, "e-s1024", "s1024", fill(groupMin, 7, 8), fill(groupMin/2, 9), fill(groupMin/4, 10), fill(1024*64, 12))

	var ones, manys [][]string
	var forgot []string
	saveOne, saveMany, saveForget := lowerOne, lowerMany, forget
	t.Cleanup(func() { lowerOne, lowerMany, forget = saveOne, saveMany, saveForget })
	lowerOne = func(pkg, dst string) error { ones = append(ones, []string{pkg}); return os.MkdirAll(dst, 0o755) }
	lowerMany = func(pkgs, names []string, dst string) error {
		manys = append(manys, pkgs)
		return os.MkdirAll(dst, 0o755)
	}
	forget = func(p string) { forgot = append(forgot, p) }

	// q128 was lowered alone by an earlier hwnpu
	s128, _ := cache.TreeDigest(q128)
	old := cache.Path(cdir, singleKey(s128), ".aimodel")
	os.MkdirAll(old, 0o755)

	a8, f8, cached, err := resolve(cdir, q8)
	if err != nil {
		t.Fatal(err)
	}
	if cached || len(manys) != 1 || len(manys[0]) != 2 || len(ones) != 0 {
		t.Fatalf("q8: cached %v, lowered together %v, alone %v; want q8 and q128 together", cached, manys, ones)
	}
	if exists(old) || len(forgot) != 1 || forgot[0] != old {
		t.Errorf("the old asset of q128: still there %v, forgotten %v", exists(old), forgot)
	}
	a128, f128, cached, err := resolve(cdir, q128)
	if err != nil {
		t.Fatal(err)
	}
	if a128 != a8 || f128 == f8 || !cached || len(manys) != 1 {
		t.Errorf("q128: %s#%s (cached %v), want %s with another function than %s, lowered once", a128, f128, cached, a8, f8)
	}
	if f128 != functionName(s128) {
		t.Errorf("q128's function %s, want %s", f128, functionName(s128))
	}

	// other bytes and programs without weights stay alone
	for _, p := range []string{part2, attn, attn2} {
		a, fn, _, err := resolve(cdir, p)
		if err != nil {
			t.Fatal(err)
		}
		if fn != "main" || a == a8 {
			t.Errorf("%s: %s#%s, want an asset of its own", filepath.Base(p), a, fn)
		}
	}
	if len(ones) != 3 || len(manys) != 1 {
		t.Errorf("lowered alone %d times, want 3", len(ones))
	}

	a, f, _, err := resolve(cdir, e1024)
	if err != nil {
		t.Fatal(err)
	}
	if a2, f2, cached, err := resolve(cdir, e128); err != nil || a2 != a || f2 == f || !cached {
		t.Errorf("the embedder's: %s#%s and %s#%s (cached %v, %v), want one asset", a, f, a2, f2, cached, err)
	}

	// a package converted later with the weights of one lowered alone: they go together
	pa, _, _, _ := resolve(cdir, part2)
	later := fakePackage(t, models, "m-q128.part2", "q128 part2", other...)
	if a, _, cached, err := resolve(cdir, later); err != nil || cached || a == pa || exists(pa) {
		t.Errorf("a later sibling: %s (cached %v, %v); the alone asset still there %v", a, cached, err, exists(pa))
	}
	if a, fn, cached, err := resolve(cdir, part2); err != nil || !cached || fn == "main" {
		t.Errorf("part2 after its sibling: %s#%s (cached %v, %v), want the group's", a, fn, cached, err)
	}
	manys = manys[:2]

	// the group's asset removed (cache.Prune): lowered together again
	os.RemoveAll(a8)
	if a, _, cached, err := resolve(cdir, q128); err != nil || a != a8 || cached || len(manys) != 3 {
		t.Errorf("after the group was removed: %s (cached %v, %v), lowered together %d times", a, cached, err, len(manys))
	}
}
