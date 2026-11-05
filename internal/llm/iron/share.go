package iron

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// sharedMark is the file in an export that says ShareExport went over it. A file of the export
// written after it (an export again into the same directory) has ShareExport go over it again.
const sharedMark = ".files-shared"

// ShareExport makes the files of the export in dir that have the same bytes as a file of an export
// next to it (or as another of its own files) share that file's blocks on the disk (cache.Clone,
// on Btrfs, XFS with reflink, ReFS and the other file systems that clone), and a file that begins
// with the bytes of a file of the same name share that start (cache.ClonePrefix): the exports of
// one model with and without images and audio, 4-bit or not, again with another option, have the
// same per-layer embedding table (4.4 GiB for Gemma 4 E2B), the same host embedding, the same
// prompt program, and a bank of weights that the one with the towers extends. It runs once for
// each export, and does nothing where the file system cannot clone. Only dir's files are
// replaced, by clones; the exports next to it are only read. The exports next to it are the
// directories beside dir with a manifest.json or with ".amd-npu" in their name.
func ShareExport(dir string, logf func(string, ...any)) {
	dir, err := filepath.Abs(dir)
	if err != nil || shared(dir) || !cache.CanCloneIn(dir) {
		return
	}
	var others []string
	parent := filepath.Dir(dir)
	entries, _ := os.ReadDir(parent)
	for _, e := range entries {
		p := filepath.Join(parent, e.Name())
		if !e.IsDir() || p == dir {
			continue
		}
		if _, err := os.Stat(filepath.Join(p, "manifest.json")); err == nil || strings.Contains(e.Name(), ".amd-npu") {
			others = append(others, p)
		}
	}
	if logf != nil {
		logf("sharing the bytes that are the same as in the %d exports next to it (once)", len(others))
	}
	start := time.Now()
	n, size := cache.ShareFiles(dir, others, 1<<20)
	if logf != nil && n > 0 {
		logf("shared %d files, %.2f GiB, in %.1f s", n, float64(size)/(1<<30), time.Since(start).Seconds())
	}
	os.WriteFile(filepath.Join(dir, sharedMark), nil, 0o644)
}

// shared reports whether ShareExport went over dir after its files were last written.
func shared(dir string) bool {
	mi, err := os.Stat(filepath.Join(dir, sharedMark))
	if err != nil {
		return false
	}
	newer := false
	filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || newer || !d.Type().IsRegular() {
			return nil
		}
		if fi, err := d.Info(); err == nil && fi.ModTime().After(mi.ModTime()) {
			newer = true
			return filepath.SkipAll
		}
		return nil
	})
	return !newer
}
