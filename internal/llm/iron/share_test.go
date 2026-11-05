package iron

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// ShareExport shares an export's files with the exports next to it once, again after the export
// is written anew, and leaves alone the directories that are not exports.
func TestShareExport(t *testing.T) {
	root := t.TempDir()
	if !cache.CanCloneIn(root) {
		t.Skip("the file system cannot clone")
	}
	table := bytes.Repeat([]byte("ple!"), 1<<19) // 2 MiB, over ShareExport's 1 MiB
	put := func(rel string) string {
		p := filepath.Join(root, rel)
		os.MkdirAll(filepath.Dir(p), 0o755)
		if err := os.WriteFile(p, table, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	put("m.amd-npu2/ple.bin")
	checkpoint := put("gemma/model.bin") // a checkpoint, not an export
	mine := put("m.amd-npu2-q4/ple.bin")
	replaced := func(f string, before os.FileInfo) bool {
		fi, err := os.Stat(f)
		if err != nil {
			t.Fatal(err)
		}
		return !os.SameFile(fi, before)
	}
	cp, _ := os.Stat(checkpoint)
	var lines []string
	logf := func(f string, a ...any) { lines = append(lines, f) }
	before, _ := os.Stat(mine)
	ShareExport(filepath.Dir(mine), logf)
	if !replaced(mine, before) || len(lines) != 2 {
		t.Fatalf("the export's table was not shared (log %q)", lines)
	}
	before, _ = os.Stat(mine)
	ShareExport(filepath.Dir(mine), logf)
	if replaced(mine, before) || len(lines) != 2 {
		t.Fatal("ShareExport went over a shared export again")
	}
	time.Sleep(10 * time.Millisecond)
	put("m.amd-npu2-q4/ple.bin") // exported again
	before, _ = os.Stat(mine)
	ShareExport(filepath.Dir(mine), logf)
	if !replaced(mine, before) {
		t.Fatal("ShareExport left an export written again")
	}
	if replaced(checkpoint, cp) {
		t.Fatal("a file of a directory that is not an export was replaced")
	}
}
