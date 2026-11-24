package cache

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Removing a unit removes Core AI's entries of its exports, of every OS build and executable,
// but not those of a hash an export left in the cache has too, nor those a unit of prebuilt
// programs is of.
func TestRemoveCoreAIEntries(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Core AI's cache is macOS's")
	}
	home, root := t.TempDir(), t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv(RootEnv, root)
	save := coreAICache
	t.Cleanup(func() { coreAICache = save })
	coreAICache = func() string { return filepath.Join(home, "Library", "Caches", "coreai-cache") }
	export := func(rel string, hash []byte) {
		d := filepath.Join(root, rel)
		os.MkdirAll(d, 0o755)
		os.WriteFile(filepath.Join(d, "main.hash"), hash, 0o644)
		os.WriteFile(filepath.Join(d, "main.mlirb"), []byte("weights"), 0o644)
	}
	entry := func(build, exe string, hash []byte) string {
		d := filepath.Join(home, "Library", "Caches", "coreai-cache", build, exe, hex.EncodeToString(hash))
		os.MkdirAll(filepath.Join(d, "DEVICE", "model.aimodelx"), 0o755)
		os.WriteFile(filepath.Join(d, "DEVICE", "model.aimodelx", "main-this.odix"), []byte("program"), 0o644)
		return d
	}
	alone, shared, pb := []byte{1, 2}, []byte{3, 4}, []byte{5, 6}
	export("models/A/a-r1-fp16.aimodel", alone)
	export("models/A/size/b-r1-fp16.coreai/b.aimodel", shared)
	export("models/B/b-r1-fp16.aimodel", shared)
	export("models/C/c-r1-fp16.aimodel", pb)
	os.MkdirAll(filepath.Join(root, "models/C/c-r1-fp16"+PrebuiltSuffix, "c.h13g.aimodelc"), 0o755)
	os.WriteFile(filepath.Join(root, "models/C/c-r1-fp16"+PrebuiltSuffix, "c.h13g.aimodelc", "main.hash"), pb, 0o644)
	gone := []string{entry("26B1", "bonsaiai-image", alone), entry("26A9", "bonsaiaid", alone)}
	kept := []string{entry("26B1", "bonsaiai-image", shared), entry("26B1", "bonsaiaid", pb)}

	u, err := Scan()
	if err != nil {
		t.Fatal(err)
	}
	unit := func(name string) Unit {
		for _, x := range u.Units {
			if x.Name == name {
				return x
			}
		}
		t.Fatalf("no unit %s in %v", name, u.Units)
		return Unit{}
	}
	if got := CoreAIEntries(unit("models/A")); len(got) != 2 {
		t.Errorf("entries of models/A: %v, want the 2 of its own hash", got)
	}
	for _, name := range []string{"models/A", "models/C/c-r1-fp16" + PrebuiltSuffix} {
		if err := unit(name).Remove(); err != nil {
			t.Fatal(err)
		}
	}
	for _, e := range gone {
		if _, err := os.Stat(e); err == nil {
			t.Errorf("%s: left", e)
		}
	}
	for _, e := range kept {
		if _, err := os.Stat(e); err != nil {
			t.Errorf("%s: removed", e)
		}
	}
	// the last export of the shared hash takes its entries along
	if err := unit("models/B").Remove(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(kept[0]); err == nil {
		t.Errorf("%s: left after the last export of its hash", kept[0])
	}
}
