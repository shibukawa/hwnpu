package coreai

import (
	"bytes"
	"encoding/hex"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/coreai/mlirbc"
)

// TestPackAssemble packs a prebuilt program whose weights are its asset's blobs in another order,
// with headers between them and a blob cut short, and assembles it again byte for byte.
func TestPackAssemble(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	blob := func(n int) []byte {
		b := make([]byte, n)
		for i := range b {
			b[i] = byte(r.Uint32())
		}
		return b
	}
	blobs := [][]byte{blob(3 << 20), blob(200 << 10), make([]byte, 1<<20), blob(1<<20 + 7)}
	w := mlirbc.NewWriter("test", map[string][]byte{"aicode": {0x05}})
	for i, b := range blobs {
		w.Blob("aicode", "b"+string(rune('0'+i)), 16, b)
	}
	top, _ := w.NewBlock()
	root := t.TempDir()
	asset := filepath.Join(root, "x.coreai", "p.aimodel")
	os.MkdirAll(asset, 0o755)
	if err := os.WriteFile(filepath.Join(asset, "main.mlirb"), w.Bytes(top), 0o644); err != nil {
		t.Fatal(err)
	}
	var res bytes.Buffer
	for _, i := range []int{3, 0, 2, 1} {
		res.Write([]byte{1, 2, 3, 4, 0, 0, 0, 9})
		b := blobs[i]
		if i == 0 {
			b = b[:len(b)-12345] // cut short
		}
		res.Write(b)
	}
	res.Write(blob(100)) // not in the bytecode
	full := filepath.Join(root, "p.h14g.aimodelc")
	pkg := filepath.Join(full, "main-h14g-delegates", "MPSGraph", "mpsExecutable.mpsgraphpackage")
	os.MkdirAll(pkg, 0o755)
	os.WriteFile(filepath.Join(pkg, "resources.bin"), res.Bytes(), 0o644)
	os.WriteFile(filepath.Join(pkg, "manifest.plist"), []byte("small"), 0o644)
	os.WriteFile(filepath.Join(pkg, "other.bin"), blob(2<<20), 0o644) // big, not in the bytecode

	packed := PackedPath(asset, "h14g")
	if err := Pack(full, asset, packed); err != nil {
		t.Fatal(err)
	}
	ppkg := filepath.Join(packed, "main-h14g-delegates", "MPSGraph", "mpsExecutable.mpsgraphpackage")
	if _, err := os.Stat(filepath.Join(ppkg, "resources.bin")); err == nil {
		t.Error("resources.bin was kept")
	}
	if fi, err := os.Stat(filepath.Join(ppkg, "resources.bin"+MapSuffix)); err != nil || fi.Size() > 4096 {
		t.Errorf("the map: %v %v", fi.Size(), err)
	}
	if _, err := os.Stat(filepath.Join(ppkg, "other.bin")); err != nil {
		t.Error("other.bin was not kept whole")
	}
	out := filepath.Join(root, "assembled.aimodelc")
	if err := Assemble(packed, asset, out); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"resources.bin", "manifest.plist", "other.bin"} {
		a, _ := os.ReadFile(filepath.Join(pkg, f))
		b, err := os.ReadFile(filepath.Join(out, "main-h14g-delegates", "MPSGraph", "mpsExecutable.mpsgraphpackage", f))
		if err != nil || !bytes.Equal(a, b) {
			t.Errorf("%s: not the same (%v)", f, err)
		}
	}
	// another bytecode: refused, nothing left
	b := w.Bytes(top)
	b[len(b)-100] ^= 0xff
	os.WriteFile(filepath.Join(asset, "main.mlirb"), b, 0o644)
	if err := Assemble(packed, asset, out); err == nil {
		t.Error("assembled from another bytecode")
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("a failed assembly left its output")
	}
}

// TestPackReal packs and assembles a prebuilt program of coreai-build's (BONSAIAI_TEST_PREBUILT: the
// X.ARCH.aimodelc beside its X.aimodel).
func TestPackReal(t *testing.T) {
	full := os.Getenv("BONSAIAI_TEST_PREBUILT")
	if full == "" {
		t.Skip("BONSAIAI_TEST_PREBUILT is not set")
	}
	asset := filepath.Join(filepath.Dir(full), filepath.Base(full)[:len(filepath.Base(full))-len(".h14g.aimodelc")]+".aimodel")
	dir := t.TempDir()
	packed := filepath.Join(dir, "packed")
	if err := Pack(full, asset, packed); err != nil {
		t.Fatal(err)
	}
	var size int64
	filepath.Walk(packed, func(p string, fi os.FileInfo, err error) error {
		if err == nil && !fi.IsDir() {
			size += fi.Size()
			t.Logf("%s %d", p[len(packed):], fi.Size())
		}
		return nil
	})
	t.Logf("packed: %d bytes", size)
	out := filepath.Join(dir, "out.aimodelc")
	if err := Assemble(packed, asset, out); err != nil {
		t.Fatal(err)
	}
}

// TestRestorableEntries lists Core AI's entries, of bonsaiai's executables, of the programs that
// have a prebuilt one for this device here: not another's, not a program's without one.
func TestRestorableEntries(t *testing.T) {
	arch := Arch()
	if arch == "" {
		t.Skip("Core AI is not available")
	}
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("BONSAIAI_CACHE", filepath.Join(home, "cache"))
	base := coreAICacheDir()
	export := filepath.Join(home, "cache", "models", "Sana", "x.coreai")
	asset := func(name, hash string, prebuilt bool) {
		a := filepath.Join(export, name+".aimodel")
		os.MkdirAll(a, 0o755)
		os.WriteFile(filepath.Join(a, "main.mlirb"), []byte("bytecode"), 0o644)
		os.WriteFile(filepath.Join(a, "main.hash"), []byte(hash), 0o644)
		if prebuilt {
			pb := PackedPath(a, arch)
			m := filepath.Join(pb, "main-"+arch+"-delegates", "MPSGraph", "mpsExecutable.mpsgraphpackage", "manifest.plist")
			os.MkdirAll(filepath.Dir(m), 0o755)
			os.WriteFile(filepath.Join(pb, "main.hash"), []byte(hash), 0o644)
			os.WriteFile(m, []byte("<key>mps.fullyPlacedOnANE</key>"), 0o644)
		}
		for _, exe := range []string{"bonsaiai-image", "other-app"} {
			os.MkdirAll(filepath.Join(base, exe, hex.EncodeToString([]byte(hash)), "ID", "model.aimodelx"), 0o755)
		}
	}
	asset("dit", "with", true)
	asset("vae", "none", false)
	got := restorableEntries()
	want := filepath.Join(base, "bonsaiai-image", hex.EncodeToString([]byte("with")))
	if len(got) != 1 || got[0] != want {
		t.Errorf("restorable %v, want %s", got, want)
	}
}
