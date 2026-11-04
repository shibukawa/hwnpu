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
	"strings"
	"time"
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

// Lookup reports whether the entry exists, and marks it used now (Prune keeps what is used).
func Lookup(dir, key, suffix string) (string, bool) {
	p := Path(dir, key, suffix)
	if _, err := os.Stat(p); err != nil {
		return p, false
	}
	now := time.Now()
	os.Chtimes(p, now, now) // best effort
	return p, true
}

// sourceSuffix names the record of where an entry was compiled from (SetSource).
const sourceSuffix = ".src"

// SetSource records the file or directory an entry was made from, so that Prune removes the
// entry once that is gone.
func SetSource(dir, key, source string) error {
	if abs, err := filepath.Abs(source); err == nil {
		source = abs
	}
	return os.WriteFile(Path(dir, key, sourceSuffix), []byte(source), 0o644)
}

// Source returns what SetSource recorded for key.
func Source(dir, key string) (string, bool) {
	b, err := os.ReadFile(Path(dir, key, sourceSuffix))
	return string(b), err == nil
}

// Prune removes the entries of dir that are not used any more: those whose source (SetSource) is
// gone, and those not looked up or stored for maxAge. Temporary entries left by a process that
// stopped while storing go after a day. It returns how many entries it removed and their bytes
// (as the files' sizes: a clone's blocks are shared and free less).
func Prune(dir string, maxAge time.Duration) (removed int, bytes int64, err error) {
	des, err := os.ReadDir(dir)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return 0, 0, nil
		}
		return 0, 0, err
	}
	now := time.Now()
	remove := func(path string) {
		n := treeSize(path)
		if os.RemoveAll(path) == nil {
			removed++
			bytes += n
		}
	}
	for _, de := range des {
		name := de.Name()
		info, err := de.Info()
		if err != nil {
			continue
		}
		if strings.HasPrefix(name, ".tmp-") {
			if now.Sub(info.ModTime()) > 24*time.Hour {
				remove(filepath.Join(dir, name))
			}
			continue
		}
		key, suffix, ok := splitEntry(name)
		if !ok || suffix == sourceSuffix {
			continue
		}
		gone := false
		if src, ok := Source(dir, key); ok {
			_, err := os.Stat(src)
			gone = errors.Is(err, fs.ErrNotExist)
		}
		if gone || now.Sub(info.ModTime()) > maxAge {
			remove(filepath.Join(dir, name))
			os.Remove(Path(dir, key, sourceSuffix))
		}
	}
	return removed, bytes, nil
}

// splitEntry splits an entry's name into its key (64 hex digits, Key) and suffix.
func splitEntry(name string) (key, suffix string, ok bool) {
	if len(name) <= 64 {
		return "", "", false
	}
	if _, err := hex.DecodeString(name[:64]); err != nil {
		return "", "", false
	}
	return name[:64], name[64:], true
}

// treeSize is the bytes of the regular files under path.
func treeSize(path string) int64 {
	var n int64
	filepath.WalkDir(path, func(_ string, d fs.DirEntry, err error) error {
		if err == nil && d.Type().IsRegular() {
			if info, err := d.Info(); err == nil {
				n += info.Size()
			}
		}
		return nil
	})
	return n
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
