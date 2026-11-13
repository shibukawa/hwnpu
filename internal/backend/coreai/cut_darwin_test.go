package coreai

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

// TestCut: an asset's bytecode is cut only while Core AI's entry of it (with main-this.odix) is
// there for this executable, and isCut and hasEntry say so.
func TestCut(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	asset := filepath.Join(t.TempDir(), "x.aimodel")
	os.MkdirAll(asset, 0o755)
	hash := []byte{0xab, 0xcd}
	os.WriteFile(filepath.Join(asset, "main.hash"), hash, 0o644)
	os.WriteFile(filepath.Join(asset, "main.mlirb"), []byte("weights"), 0o644)

	build, err := syscall.Sysctl("kern.osversion")
	if err != nil {
		t.Skip(err)
	}
	exe, _ := os.Executable()
	entry := filepath.Join(home, "Library", "Caches", "coreai-cache", build, filepath.Base(exe), hex.EncodeToString(hash), "DEVICE", "model.aimodelx")
	os.MkdirAll(entry, 0o755)
	odix := filepath.Join(entry, "main-this.odix")

	for _, c := range []struct {
		name string
		odix []byte // nil: none
	}{{"no odix", nil}, {"empty odix", []byte{}}} {
		os.Remove(odix)
		if c.odix != nil {
			os.WriteFile(odix, c.odix, 0o644)
		}
		if hasEntry(asset) {
			t.Errorf("%s: hasEntry", c.name)
		}
		if err := cut(asset); err != nil || isCut(asset) {
			t.Errorf("%s: cut (%v)", c.name, err)
		}
	}
	os.WriteFile(odix, []byte("program"), 0o644)
	if !hasEntry(asset) {
		t.Fatal("no entry found")
	}
	if err := cut(asset); err != nil || !isCut(asset) {
		t.Fatalf("not cut (%v)", err)
	}
	if b, _ := os.ReadFile(filepath.Join(asset, "main.hash")); string(b) != string(hash) {
		t.Error("main.hash changed")
	}
}
