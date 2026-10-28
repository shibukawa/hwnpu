// Package cache stores compiled graphs on disk, keyed by everything that affects compilation.
// Entries are published with an atomic rename, so a crash leaves either the old state or a
// complete entry.
package cache

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
)

// Key hashes the parts with length prefixes, so ("ab","c") and ("a","bc") differ.
func Key(parts ...[]byte) string {
	h := sha256.New()
	for _, p := range parts {
		var n [8]byte
		binary.LittleEndian.PutUint64(n[:], uint64(len(p)))
		h.Write(n[:])
		h.Write(p)
	}
	return hex.EncodeToString(h.Sum(nil))
}

// TreeDigest hashes the relative paths and contents of every regular file under root.
func TreeDigest(root string) ([]byte, error) {
	var files []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.Type().IsRegular() {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	h := sha256.New()
	for _, f := range files {
		rel, _ := filepath.Rel(root, f)
		data, err := os.ReadFile(f)
		if err != nil {
			return nil, err
		}
		fmt.Fprintf(h, "%s\x00%d\x00", filepath.ToSlash(rel), len(data))
		h.Write(data)
	}
	return h.Sum(nil), nil
}

// Path returns where the entry for key with the given suffix lives.
func Path(dir, key, suffix string) string { return filepath.Join(dir, key+suffix) }

// Lookup reports whether the entry exists.
func Lookup(dir, key, suffix string) (string, bool) {
	p := Path(dir, key, suffix)
	_, err := os.Stat(p)
	return p, err == nil
}

// Store creates the entry. write receives a temporary path in dir (a file or a directory to
// create) and must fill it; the result is renamed into place. An existing entry is kept.
func Store(dir, key, suffix string, write func(tmp string) error) (string, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	final := Path(dir, key, suffix)
	tmpDir, err := os.MkdirTemp(dir, ".tmp-"+key[:12]+"-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tmpDir)
	tmp := filepath.Join(tmpDir, "entry"+suffix)
	if err := write(tmp); err != nil {
		return "", err
	}
	if err := os.Rename(tmp, final); err != nil {
		if _, statErr := os.Stat(final); statErr == nil {
			return final, nil // another process stored it first
		}
		return "", err
	}
	return final, nil
}

// CopyTree copies a directory tree (used when a rename across file systems is impossible).
func CopyTree(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, path)
		target := filepath.Join(dst, rel)
		switch {
		case d.IsDir():
			return os.MkdirAll(target, 0o755)
		case d.Type().IsRegular():
			data, err := os.ReadFile(path)
			if err != nil {
				return err
			}
			return os.WriteFile(target, data, 0o644)
		}
		return errors.New("cache: unsupported file type at " + path)
	})
}
