package tokenizer

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// openBeside opens the binary file that stands for the tokenizer file at path (BinaryPath:
// path itself when it is one). ok is false when there is none to take, and the caller reads
// path: no binary file, or one of another version beside a file to read instead.
func openBeside[T any](path string, open func(string) (T, error)) (t T, err error, ok bool) {
	b := BinaryPath(path)
	if _, err := os.Stat(b); err != nil {
		if b == path || !errors.Is(err, fs.ErrNotExist) {
			return t, err, true
		}
		return t, nil, false
	}
	t, err = open(b)
	if err != nil && errors.Is(err, errNotBinary) && b != path {
		if _, serr := os.Stat(path); serr == nil {
			return t, nil, false // an older layout: the text beside it is read
		}
	}
	return t, err, true
}

// Exists reports whether the tokenizer file at path is there to load: the file, or the binary
// file that stands for it (an export's, made by its recipe's finish).
func Exists(path string) bool {
	for _, p := range []string{path, BinaryPath(path)} {
		if _, err := os.Stat(p); err == nil {
			return true
		}
	}
	return false
}

// CacheDir is where a tokenizer read from its text file (no binary file beside it: a
// checkpoint's own, or an export made before exports held binary files) is kept as a binary
// file, so that the next process maps it rather than parse the text again: the cache's root,
// as the compile caches' entries (KEY.tok). "" keeps none (tests).
var CacheDir = func() string {
	if testing.Testing() {
		return ""
	}
	return cache.Root()
}

// cachedLoad is load(path), from the cache (CacheDir) when it holds the binary file of the same
// file (by its path, size and time, and PLaMo's tokenizer_config.json's), else read and stored
// there; a failure to store is not one to load.
func cachedLoad[T any](path string, load func(string) (T, error), write func(T, string) error, open func(string) (T, error)) (T, error) {
	dir := CacheDir()
	if dir == "" {
		return load(path)
	}
	key, err := sourceKey(path)
	if err != nil {
		return load(path)
	}
	if p, ok := cache.Lookup(dir, key, BinaryExt); ok {
		if t, err := open(p); err == nil {
			return t, nil
		}
	}
	t, err := load(path)
	if err != nil {
		return t, err
	}
	if _, err := cache.Store(dir, key, BinaryExt, func(tmp string) error { return write(t, tmp) }); err == nil {
		cache.SetSource(dir, key, path)
	}
	return t, nil
}

// sourceKey keys a tokenizer file by what it is: its path, size and modification time (and those
// of the tokenizer_config.json a tokenizer.jsonl is read with), and the binary layout's version.
func sourceKey(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	parts := [][]byte{[]byte(fmt.Sprintf("hwnpu-tokbin-v%d", tokbinVersion))}
	files := []string{abs}
	if strings.HasSuffix(abs, ".jsonl") {
		files = append(files, plamoConfig(abs))
	}
	for i, f := range files {
		fi, err := os.Stat(f)
		if err != nil {
			if i > 0 && errors.Is(err, fs.ErrNotExist) {
				parts = append(parts, []byte(f+" none"))
				continue
			}
			return "", err
		}
		parts = append(parts, []byte(fmt.Sprintf("%s %d %d", f, fi.Size(), fi.ModTime().UnixNano())))
	}
	return cache.Key(parts...), nil
}
