package store

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
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

// The scopes of what is shared (docs/cache-sharing-design.md 6).
const (
	ScopeUpstream = "upstream" // a checkpoint's files, by their SHA-256
	ScopePortable = "portable" // what is converted: the same on every machine of the target
	ScopeHost     = "host"     // what is compiled for one OS, driver or chip
)

// Entry is a directory of what was made (an export, a converted program) and what it was made
// from (docs/cache-sharing-design.md 6). Its paths are relative to the directory.
type Entry struct {
	Format int    `json:"format"`
	Key    string `json:"key"` // the fingerprint of Recipe (Fingerprint)
	Kind   string `json:"kind"`
	Scope  string `json:"scope"`
	Model  string `json:"model,omitempty"`
	Target string `json:"target,omitempty"`
	// Version is the model's version for the target (models.yaml's targets; 0: none).
	Version int `json:"version,omitempty"`
	// Develop is a version of a model in development (VersionDir).
	Develop bool `json:"develop,omitempty"`
	// Path is the entry's folder in a source (EntryPath).
	Path string `json:"path,omitempty"`
	// Recipe is everything the entry was made from: what the key is the fingerprint of.
	Recipe map[string]string `json:"recipe"`
	Files  []File            `json:"files"`
	// Pack is the one file in a source that holds Files, for an entry that goes there packed
	// (Packs: PackFile); none here, where the files are.
	Pack *File `json:"pack,omitempty"`
	// Holds names the parts an entry that grows holds (the prebuilt programs of a Core AI export
	// for a chip, which one machine may put before it has built them all): not in the key, so
	// that a machine finds the entry by what it would compile; PutEntry replaces an entry of the
	// same key that holds less (HoldsMore), and a machine that lacks a part fetches it again.
	Holds []string `json:"holds,omitempty"`
	// Revision is what of the entry changes without its readers fetching it again (the SHA-256
	// of the record of the build of prebuilt programs: coreai.RecordFile): PutEntry replaces an
	// entry of the same key and holdings whose revision is another, but not with one that holds
	// less.
	Revision string `json:"revision,omitempty"`
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

// HoldsMore reports whether e holds a part (Holds) that held lacks: an entry of the same key
// that is to replace an older one. An entry without Holds holds nothing more.
func HoldsMore(e Entry, held []string) bool {
	in := make(map[string]bool, len(held))
	for _, h := range held {
		in[h] = true
	}
	for _, h := range e.Holds {
		if !in[h] {
			return true
		}
	}
	return false
}

// Bytes is the size of the entry's files.
func (e Entry) Bytes() int64 {
	var n int64
	for _, f := range e.Files {
		n += f.Size
	}
	return n
}

// Digest is what the entry holds, whatever folder, version or pack it is kept in: "sha256:" and
// the SHA-256 of a line for each of its files (path, size and SHA-256, separated by NUL), in the
// order of their paths. What models.yaml's exports pin: every file of an entry fetched is
// checked against Files (unpacked ones too), so that an entry of the digest pinned holds those
// bytes and no others. Its version, path, recipe, pack and when it was made are not in it: the
// same export packed again, or written as a published version, has the same digest.
func (e Entry) Digest() string {
	fs := slices.Clone(e.Files)
	slices.SortFunc(fs, func(a, b File) int { return strings.Compare(a.Path, b.Path) })
	h := sha256.New()
	for _, f := range fs {
		fmt.Fprintf(h, "%s\x00%d\x00%s\n", f.Path, f.Size, strings.ToLower(f.SHA256))
	}
	return DigestPrefix + hex.EncodeToString(h.Sum(nil))
}

// DigestPrefix starts every Digest.
const DigestPrefix = "sha256:"

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

// LocalMarks are the names of the files a directory here holds about itself, never of what it is
// (iron.ShareExport's and chat.ShareWeights' marks that they went over it): not in an entry.
var LocalMarks = []string{".files-shared", ".weights-shared"}

// IsCheckData reports whether the folder at p is the check data of a tools/amd export: check/
// beside a graph's or a model's manifest (manifest.json, model.json), what iron_export_lm.py
// always writes (check/words.json) and --check adds (the inputs and outputs of a run). Only the
// tools' checks and tests read it, from the export they made: it is not in an entry.
func IsCheckData(p string) bool {
	if filepath.Base(p) != "check" {
		return false
	}
	for _, f := range []string{"manifest.json", "model.json"} {
		if _, err := os.Stat(filepath.Join(filepath.Dir(p), f)); err == nil {
			return true
		}
	}
	return false
}

// Describe lists the files of dir (every regular file but EntryFile, LocalMarks and those of
// check data: IsCheckData) with their sizes and SHA-256, into e, and writes e as dir's
// EntryFile. Symbolic links are not followed or listed.
func Describe(dir string, e Entry) (Entry, error) {
	e.Format, e.Files, e.Pack = EntryFormat, nil, nil
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
		if d.IsDir() && p != dir && IsCheckData(p) {
			return filepath.SkipDir
		}
		if !d.Type().IsRegular() {
			return nil
		}
		rel, _ := filepath.Rel(dir, p)
		rel = filepath.ToSlash(rel)
		if rel == EntryFile || slices.Contains(LocalMarks, d.Name()) {
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
