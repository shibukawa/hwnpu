package store

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// EntryFile is the file in an entry's directory that describes it: written last, so that a
// directory with it is whole.
const EntryFile = "entry.json"

// EntryFormat is the version of EntryFile this hwnpu reads and writes.
const EntryFormat = 1

// The scopes of what is shared (docs/cache-sharing.md 6).
const (
	ScopeUpstream = "upstream" // a checkpoint's files, by their SHA-256
	ScopePortable = "portable" // what is converted: the same on every machine of the target
	ScopeHost     = "host"     // what is compiled for one OS, driver or chip
)

// Entry is a directory of what was made (an export, a converted program) and what it was made
// from (docs/cache-sharing.md 6). Its paths are relative to the directory.
type Entry struct {
	Format int    `json:"format"`
	Key    string `json:"key"` // the fingerprint of Recipe (Fingerprint)
	Kind   string `json:"kind"`
	Scope  string `json:"scope"`
	Model  string `json:"model,omitempty"`
	Target string `json:"target,omitempty"`
	// Version is the model's version for the target, for what is published (0: made here).
	Version int `json:"version,omitempty"`
	// Recipe is everything the entry was made from: what the key is the fingerprint of.
	Recipe map[string]string `json:"recipe"`
	Files  []File            `json:"files"`
	// MadeBy is the tools and the machine that made it, recorded only (not in the key).
	MadeBy map[string]string `json:"made_by,omitempty"`
	Made   time.Time         `json:"made"`
}

// File is a file of an entry.
type File struct {
	Path   string `json:"path"`
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

// Bytes is the size of the entry's files.
func (e Entry) Bytes() int64 {
	var n int64
	for _, f := range e.Files {
		n += f.Size
	}
	return n
}

// Fingerprint is the key of a recipe: the SHA-256 of its settings in order of their names.
func Fingerprint(recipe map[string]string) string {
	names := make([]string, 0, len(recipe))
	for k := range recipe {
		names = append(names, k)
	}
	sort.Strings(names)
	var parts [][]byte
	for _, k := range names {
		parts = append(parts, []byte(k), []byte(recipe[k]))
	}
	return cache.Key(parts...)
}

// Describe lists the files of dir (every regular file but EntryFile) with their sizes and
// SHA-256, into e, and writes e as dir's EntryFile. Symbolic links are not followed or listed.
func Describe(dir string, e Entry) (Entry, error) {
	e.Format, e.Files = EntryFormat, nil
	if e.Key == "" {
		e.Key = Fingerprint(e.Recipe)
	}
	if e.Made.IsZero() {
		e.Made = time.Now().UTC()
	}
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.Type().IsRegular() {
			return nil
		}
		rel, _ := filepath.Rel(dir, p)
		rel = filepath.ToSlash(rel)
		if rel == EntryFile {
			return nil
		}
		fi, err := d.Info()
		if err != nil {
			return err
		}
		sum, err := FileSHA256(p)
		if err != nil {
			return err
		}
		e.Files = append(e.Files, File{Path: rel, Size: fi.Size(), SHA256: sum})
		return nil
	})
	if err != nil {
		return e, err
	}
	return e, WriteEntry(dir, e)
}

// WriteEntry writes e as dir's EntryFile.
func WriteEntry(dir string, e Entry) error {
	b, err := json.MarshalIndent(e, "", "  ")
	if err != nil {
		return err
	}
	tmp := filepath.Join(dir, EntryFile+".tmp")
	if err := os.WriteFile(tmp, append(b, '\n'), 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(dir, EntryFile))
}

// ReadEntry reads dir's EntryFile.
func ReadEntry(dir string) (Entry, error) {
	b, err := os.ReadFile(filepath.Join(dir, EntryFile))
	if err != nil {
		return Entry{}, err
	}
	return parseEntry(b)
}

func parseEntry(b []byte) (Entry, error) {
	var e Entry
	if err := json.Unmarshal(b, &e); err != nil {
		return e, fmt.Errorf("store: %s: %w", EntryFile, err)
	}
	if e.Format != EntryFormat {
		return e, fmt.Errorf("store: %s of format %d, not the %d this hwnpu reads", EntryFile, e.Format, EntryFormat)
	}
	for _, f := range e.Files {
		if !SafeKey(f.Path) || len(f.SHA256) != 64 || strings.ToLower(f.SHA256) != f.SHA256 {
			return e, fmt.Errorf("store: %s: a file %q that is not a path inside the entry with its SHA-256", EntryFile, f.Path)
		}
	}
	return e, nil
}

// Verify checks dir's files against its EntryFile: the sizes, and with sums the SHA-256 too.
func Verify(dir string, sums bool) error {
	e, err := ReadEntry(dir)
	if err != nil {
		return err
	}
	for _, f := range e.Files {
		p := filepath.Join(dir, filepath.FromSlash(f.Path))
		fi, err := os.Stat(p)
		if err != nil {
			return err
		}
		if fi.Size() != f.Size {
			return fmt.Errorf("store: %s: %d bytes, not the %d of %s", p, fi.Size(), f.Size, EntryFile)
		}
		if sums {
			got, err := FileSHA256(p)
			if err != nil {
				return err
			}
			if got != f.SHA256 {
				return fmt.Errorf("store: %s: SHA-256 %s, not the %s of %s", p, got, f.SHA256, EntryFile)
			}
		}
	}
	return nil
}
