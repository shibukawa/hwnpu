// Package store reads and writes the places files are shared through (docs/cache-sharing.md 7):
// a folder (a shared one: NFS, SMB, an external disk) or S3 (any endpoint: AWS, RustFS, Cloudflare R2,
// through tinygodriver's storage/s3, without the AWS SDK or the aws command). A Store holds
// objects by key, a slash-separated path; the layers (layer.go) keep in it the blobs of files by
// their SHA-256, the entries of what was converted, and the index of the checkpoints' files.
package store

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
)

// ErrNotFound is a Store's error for a key it does not hold.
var ErrNotFound = errors.New("not there")

// Store is a place objects are read from and written to by key.
type Store interface {
	// String names the store for logs.
	String() string
	// Get writes the object to dst (through dst.part, which a Get cut off goes on from) and
	// returns its size. It reports progress to bar's function, if any, with the bytes so far and
	// the total.
	Get(ctx context.Context, key, dst string, bar func(done, total int64)) (int64, error)
	// Read returns a small object whole.
	Read(ctx context.Context, key string) ([]byte, error)
	// Has reports whether the object is there, and its size.
	Has(ctx context.Context, key string) (int64, bool, error)
	// Put writes the file src as the object, replacing one there.
	Put(ctx context.Context, key, src string) error
	// Write writes a small object.
	Write(ctx context.Context, key string, b []byte) error
	// List returns the keys under prefix ("a/b/"), all of them.
	List(ctx context.Context, prefix string) ([]string, error)
	// Delete removes the object; one not there is no error.
	Delete(ctx context.Context, key string) error
}

// SafeKey reports whether key is a relative slash path that stays inside the store.
func SafeKey(key string) bool {
	if key == "" || strings.HasPrefix(key, "/") || strings.Contains(key, "\\") || strings.Contains(key, ":") {
		return false
	}
	return !slices.Contains(strings.Split(key, "/"), "..") && !slices.Contains(strings.Split(key, "/"), "")
}

// Dir is a Store of a folder: the object of key is the file Root/key.
type Dir struct {
	Name, Root string
}

func (d Dir) String() string { return fmt.Sprintf("%s (folder %s)", d.Name, d.Root) }

func (d Dir) file(key string) (string, error) {
	if !SafeKey(key) {
		return "", fmt.Errorf("store: %q is not a key inside %s", key, d)
	}
	return filepath.Join(d.Root, filepath.FromSlash(key)), nil
}

func (d Dir) Get(ctx context.Context, key, dst string, bar func(done, total int64)) (int64, error) {
	src, err := d.file(key)
	if err != nil {
		return 0, err
	}
	fi, err := os.Stat(src)
	if err != nil || !fi.Mode().IsRegular() {
		return 0, fmt.Errorf("store: %s: %s: %w", d.Name, key, ErrNotFound)
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return 0, err
	}
	// the same file system: a hard link (the layer's files are never written in place)
	if os.Link(src, dst) == nil {
		return fi.Size(), nil
	}
	in, err := os.Open(src)
	if err != nil {
		return 0, err
	}
	defer in.Close()
	return copyTo(ctx, in, dst, 0, fi.Size(), bar)
}

func (d Dir) Read(_ context.Context, key string) ([]byte, error) {
	p, err := d.file(key)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("store: %s: %s: %w", d.Name, key, ErrNotFound)
	}
	return b, err
}

func (d Dir) Has(_ context.Context, key string) (int64, bool, error) {
	p, err := d.file(key)
	if err != nil {
		return 0, false, err
	}
	fi, err := os.Stat(p)
	if errors.Is(err, os.ErrNotExist) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, err
	}
	return fi.Size(), fi.Mode().IsRegular(), nil
}

// Put copies src through a temporary file of the folder and renames it into place, so that a
// reader never sees part of an object.
func (d Dir) Put(ctx context.Context, key, src string) error {
	p, err := d.file(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	tmp, err := os.CreateTemp(filepath.Dir(p), ".put-*")
	if err != nil {
		return err
	}
	tmp.Close()
	os.Remove(tmp.Name())
	if _, err := copyTo(ctx, in, tmp.Name(), 0, -1, nil); err != nil {
		os.Remove(tmp.Name())
		os.Remove(tmp.Name() + ".part")
		return err
	}
	return os.Rename(tmp.Name(), p)
}

func (d Dir) Write(_ context.Context, key string, b []byte) error {
	p, err := d.file(key)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(p), ".put-*")
	if err != nil {
		return err
	}
	_, err = tmp.Write(b)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp.Name())
		return err
	}
	return os.Rename(tmp.Name(), p)
}

func (d Dir) List(_ context.Context, prefix string) ([]string, error) {
	root := d.Root
	if prefix != "" {
		if !SafeKey(strings.TrimSuffix(prefix, "/")) {
			return nil, fmt.Errorf("store: %q is not a prefix inside %s", prefix, d)
		}
		root = filepath.Join(d.Root, filepath.FromSlash(prefix))
	}
	var keys []string
	err := filepath.WalkDir(root, func(p string, e os.DirEntry, err error) error {
		if err != nil {
			if errors.Is(err, os.ErrNotExist) {
				return nil
			}
			return err
		}
		if e.Type().IsRegular() && !strings.HasPrefix(e.Name(), ".put-") {
			rel, _ := filepath.Rel(d.Root, p)
			keys = append(keys, filepath.ToSlash(rel))
		}
		return nil
	})
	return keys, err
}

func (d Dir) Delete(_ context.Context, key string) error {
	p, err := d.file(key)
	if err != nil {
		return err
	}
	if err := os.Remove(p); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return nil
}

// copyTo writes r to dst through dst.part, from offset have (what dst.part holds already), and
// renames it into place. total < 0 is not known.
func copyTo(ctx context.Context, r io.Reader, dst string, have, total int64, bar func(done, total int64)) (int64, error) {
	part := dst + ".part"
	flags := os.O_CREATE | os.O_WRONLY | os.O_TRUNC
	if have > 0 {
		flags = os.O_CREATE | os.O_WRONLY | os.O_APPEND
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return 0, err
	}
	w := &counter{w: f, n: have, total: total, bar: bar, ctx: ctx}
	_, err = io.Copy(w, r)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return w.n, err
	}
	return w.n, os.Rename(part, dst)
}

type counter struct {
	w        io.Writer
	n, total int64
	bar      func(done, total int64)
	ctx      context.Context
}

func (c *counter) Write(b []byte) (int, error) {
	if err := c.ctx.Err(); err != nil {
		return 0, err
	}
	n, err := c.w.Write(b)
	c.n += int64(n)
	if c.bar != nil {
		c.bar(c.n, c.total)
	}
	return n, err
}

// FileSHA256 is the SHA-256 of a file, in lower-case hex.
func FileSHA256(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// Join joins key parts with slashes.
func Join(elem ...string) string { return path.Join(elem...) }
