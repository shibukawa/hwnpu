package coreai

import (
	"archive/tar"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/klauspost/compress/zstd"
)

// BundleFile is the one file the shared layers carry for the packed prebuilt programs of an
// export for an architecture: PackedRoot as a tar, compressed with zstd. One file rather than
// a hundred small ones (each a PUT and a MOVE on WebDAV), a third of their size (the compiled
// regions compress well).
const BundleFile = "prebuilt.tar.zst"

// Bundle writes the files under root (PackedRoot) as a tar compressed with zstd at out.
func Bundle(root, out string) error {
	tmp := out + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	err = func() error {
		zw, err := zstd.NewWriter(f, zstd.WithEncoderLevel(zstd.SpeedBestCompression))
		if err != nil {
			return err
		}
		tw := tar.NewWriter(zw)
		err = filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
			if err != nil || p == root {
				return err
			}
			rel, err := filepath.Rel(root, p)
			if err != nil {
				return err
			}
			name := filepath.ToSlash(rel)
			if strings.HasPrefix(d.Name(), ".") {
				// a compile's or a pack's leftovers (.full-*, .prebuild-*, *.part)
				if d.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
			if d.IsDir() {
				return tw.WriteHeader(&tar.Header{Typeflag: tar.TypeDir, Name: name + "/", Mode: 0o755})
			}
			if !d.Type().IsRegular() {
				return nil
			}
			fi, err := d.Info()
			if err != nil {
				return err
			}
			if err := tw.WriteHeader(&tar.Header{Typeflag: tar.TypeReg, Name: name, Mode: 0o644, Size: fi.Size()}); err != nil {
				return err
			}
			in, err := os.Open(p)
			if err != nil {
				return err
			}
			defer in.Close()
			_, err = io.Copy(tw, in)
			return err
		})
		if err != nil {
			return err
		}
		if err := tw.Close(); err != nil {
			return err
		}
		return zw.Close()
	}()
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, out)
}

// Unbundle makes dst (PackedRoot) from the bundle at file: through dst.part, renamed when
// whole. Only directories and regular files inside dst are taken.
func Unbundle(file, dst string) error {
	f, err := os.Open(file)
	if err != nil {
		return err
	}
	defer f.Close()
	zr, err := zstd.NewReader(f)
	if err != nil {
		return err
	}
	defer zr.Close()
	tmp := dst + ".part"
	os.RemoveAll(tmp)
	err = func() error {
		if err := os.MkdirAll(tmp, 0o755); err != nil {
			return err
		}
		tr := tar.NewReader(zr)
		for {
			h, err := tr.Next()
			if errors.Is(err, io.EOF) {
				return nil
			}
			if err != nil {
				return err
			}
			name := h.Name
			if h.Typeflag == tar.TypeDir {
				name = strings.TrimSuffix(name, "/")
			}
			if !fs.ValidPath(name) || name == "." || !filepath.IsLocal(filepath.FromSlash(name)) {
				return fmt.Errorf("%s: %q is not a path inside it", filepath.Base(file), h.Name)
			}
			p := filepath.Join(tmp, filepath.FromSlash(name))
			switch h.Typeflag {
			case tar.TypeDir:
				if err := os.MkdirAll(p, 0o755); err != nil {
					return err
				}
			case tar.TypeReg:
				if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
					return err
				}
				out, err := os.Create(p)
				if err != nil {
					return err
				}
				_, err = io.Copy(out, tr)
				if cerr := out.Close(); err == nil {
					err = cerr
				}
				if err != nil {
					return err
				}
			default:
				return fmt.Errorf("%s: %q is not a file or a directory", filepath.Base(file), h.Name)
			}
		}
	}()
	if err != nil {
		os.RemoveAll(tmp)
		return err
	}
	os.RemoveAll(dst)
	return os.Rename(tmp, dst)
}
