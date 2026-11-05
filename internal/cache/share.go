package cache

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// ShareFiles makes each file of dir (and of its subdirectories) of least bytes or more a clone of
// a file with the same bytes (Clone): one of the directories in others, or an earlier one of dir
// itself. A file left as it is then shares its start with a file of the same name that begins
// with the same bytes (ClonePrefix): a bank of weights that another export extends. Exports of
// one model that differ in a few files (with images and audio or not, 4-bit or not) then take the
// room of what they have in common once. Only dir's files are replaced; the others are only
// read. It reports how many files were replaced and the bytes they now share. Best effort: a file
// that cannot be read or cloned stays as it is.
func ShareFiles(dir string, others []string, least int64) (n int, size int64) {
	bySize, byName := map[int64][]string{}, map[string][]string{}
	add := func(f string) {
		if fi, err := os.Stat(f); err == nil {
			bySize[fi.Size()] = append(bySize[fi.Size()], f)
			byName[filepath.Base(f)] = append(byName[filepath.Base(f)], f)
		}
	}
	for _, o := range others {
		for _, f := range regularFiles(o, least) {
			add(f)
		}
	}
	own := regularFiles(dir, least)
	sort.Strings(own)
	for _, f := range own {
		fi, err := os.Stat(f)
		if err != nil {
			continue
		}
		// a file that already shares its blocks with one of them (on Linux, where that is seen)
		// stays: a clone of another copy would give back nothing
		shared := false
		for _, c := range bySize[fi.Size()] {
			shared = shared || sharedPrefix(f, c, fi.Size()) >= fi.Size()
		}
		for _, c := range bySize[fi.Size()] {
			if shared {
				break
			}
			if ok, _ := Clone(f, c); ok {
				n, size, shared = n+1, size+fi.Size(), true
			}
		}
		for _, c := range byName[filepath.Base(f)] {
			shared = shared || sharedPrefix(f, c, min(fi.Size(), minPrefix)) >= minPrefix
		}
		for _, c := range byName[filepath.Base(f)] {
			if shared {
				break
			}
			if b, _ := ClonePrefix(f, c); b > 0 {
				n, size, shared = n+1, size+b, true
			}
		}
		add(f)
	}
	return n, size
}

// regularFiles lists the regular files under dir of at least min bytes, leaving out the
// temporary files of Clone and CanCloneIn.
func regularFiles(dir string, min int64) []string {
	var out []string
	filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || !d.Type().IsRegular() || strings.HasSuffix(p, ".clone") || strings.HasPrefix(d.Name(), ".clone-probe-") {
			return nil
		}
		if fi, err := d.Info(); err == nil && fi.Size() >= min {
			out = append(out, p)
		}
		return nil
	})
	return out
}
