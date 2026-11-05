package cache

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// ShareFiles replaces the files of a directory that have the bytes of a file next to it, or of an
// earlier file of its own, and leaves alone the other directories, small files and files that
// differ.
func TestShareFiles(t *testing.T) {
	root := t.TempDir()
	big := bytes.Repeat([]byte("ple!"), 1<<18) // 1 MiB, as ClonePrefix shares at least
	other := bytes.Repeat([]byte("emb!"), 1<<18)
	bank := bytes.Repeat([]byte("bnk!"), 1<<18)
	put := func(rel string, b []byte) string {
		p := filepath.Join(root, rel)
		os.MkdirAll(filepath.Dir(p), 0o755)
		if err := os.WriteFile(p, b, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	files := map[string]string{
		"old":  put("m.amd-npu2/ple.bin", big),
		"new":  put("m.mm.amd-npu2/ple.bin", big),            // the same bytes in the other export
		"s6":   put("m.mm.amd-npu2/vae/s6/graph.elf", other), // the same bytes twice in one export
		"s7":   put("m.mm.amd-npu2/vae/s7/graph.elf", other),
		"own":  put("m.mm.amd-npu2/embed.bin", append(bytes.Clone(big[:len(big)-1]), '?')),
		"tiny": put("m.mm.amd-npu2/config.json", []byte("{}")),
		"bank": put("m.amd-npu2/bank.bin", bank),
		"ext":  put("m.mm.amd-npu2/bank.bin", append(bytes.Clone(bank), big...)), // the bank, then a tower's
	}
	before := map[string]os.FileInfo{}
	for k, f := range files {
		before[k], _ = os.Stat(f)
	}
	n, size := ShareFiles(filepath.Join(root, "m.mm.amd-npu2"), []string{filepath.Join(root, "m.amd-npu2")}, 1024)
	if !CanCloneIn(root) {
		if n != 0 {
			t.Fatalf("%d files shared on a file system that does not clone", n)
		}
		t.Skip("the file system cannot clone")
	}
	ranges := rangeBlock(mustOpen(t, root)) > 0
	wantN, want := 2, int64(len(big)+len(other))
	if ranges {
		wantN, want = 3, want+int64(len(bank)) // the start of the longer bank
	}
	if n != wantN || size != want {
		t.Errorf("shared %d files of %d bytes, want %d of %d", n, size, wantN, want)
	}
	for k, want := range map[string]bool{"old": false, "new": true, "s6": false, "s7": true, "own": false, "tiny": false, "bank": false, "ext": ranges} {
		fi, err := os.Stat(files[k])
		if err != nil {
			t.Fatal(err)
		}
		if got := !os.SameFile(fi, before[k]); got != want {
			t.Errorf("%s replaced: %v, want %v", k, got, want)
		}
	}
	if b, _ := os.ReadFile(files["new"]); !bytes.Equal(b, big) {
		t.Error("the shared file's bytes changed")
	}
}
