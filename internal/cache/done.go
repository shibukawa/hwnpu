package cache

import (
	"os"
	"path/filepath"
)

// The marks of what is whole (NAME.done, written last): those that hold a path hold it relative
// to the mark's directory, so that a directory copied to another machine or mounted elsewhere
// still finds its programs (docs/cache-sharing.md 2).

// WriteDone writes the mark done that says path is whole.
func WriteDone(done, path string) error {
	rel := path
	if r, err := filepath.Rel(filepath.Dir(done), path); err == nil && filepath.IsLocal(r) {
		rel = filepath.ToSlash(r)
	}
	return os.WriteFile(done, []byte(rel), 0o644)
}

// ReadDone reads the mark done: the path it holds, relative to the mark's directory. A mark of
// an earlier hwnpu holds an absolute path; when that is not there (the directory was moved),
// the same name beside the mark is taken.
func ReadDone(done string) (string, bool) {
	b, err := os.ReadFile(done)
	if err != nil {
		return "", false
	}
	p := string(b)
	if p == "" {
		return "", false
	}
	if !filepath.IsAbs(p) {
		return filepath.Join(filepath.Dir(done), filepath.FromSlash(p)), true
	}
	if _, err := os.Stat(p); err == nil {
		return p, true
	}
	near := filepath.Join(filepath.Dir(done), filepath.Base(p))
	if _, err := os.Stat(near); err == nil {
		return near, true
	}
	return "", false
}
