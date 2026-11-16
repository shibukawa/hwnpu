package coreai

import (
	"archive/tar"
	"os"
	"path/filepath"
	"testing"

	"github.com/klauspost/compress/zstd"
)

// TestBundle makes the packed programs again from their bundle, a compile's leftovers left out,
// and refuses a bundle with a path outside its folder.
func TestBundle(t *testing.T) {
	dir := t.TempDir()
	root := filepath.Join(dir, "h14g")
	files := map[string]string{
		"dit.h14g.aimodelc/main.hash":                        "h",
		"vae/decoder.h14g.aimodelc/resources.bin.map.json":   "{}",
		"vae/decoder.h14g.aimodelc/main-h14g-delegates/x.bc": "llir",
	}
	for f, body := range files {
		os.MkdirAll(filepath.Dir(filepath.Join(root, f)), 0o755)
		os.WriteFile(filepath.Join(root, f), []byte(body), 0o644)
	}
	os.MkdirAll(filepath.Join(root, ".full-1"), 0o755)
	os.WriteFile(filepath.Join(root, ".full-1", "resources.bin"), []byte("big"), 0o644)
	b := filepath.Join(dir, BundleFile)
	if err := Bundle(root, b); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "there", "h14g")
	if err := Unbundle(b, out); err != nil {
		t.Fatal(err)
	}
	for f, body := range files {
		if got, err := os.ReadFile(filepath.Join(out, f)); err != nil || string(got) != body {
			t.Errorf("%s: %q %v", f, got, err)
		}
	}
	if _, err := os.Stat(filepath.Join(out, ".full-1")); err == nil {
		t.Error("a leftover was bundled")
	}

	for _, name := range []string{"../escape", "nested/../escape"} {
		evil := filepath.Join(dir, "evil.tar.zst")
		f, _ := os.Create(evil)
		zw, _ := zstd.NewWriter(f)
		tw := tar.NewWriter(zw)
		tw.WriteHeader(&tar.Header{Typeflag: tar.TypeReg, Name: name, Size: 1})
		tw.Write([]byte("x"))
		tw.Close()
		zw.Close()
		f.Close()
		if err := Unbundle(evil, filepath.Join(dir, "evil")); err == nil {
			t.Errorf("%q was accepted", name)
		}
		if _, err := os.Stat(filepath.Join(dir, "escape")); err == nil {
			t.Errorf("%q was written outside", name)
		}
	}
}
