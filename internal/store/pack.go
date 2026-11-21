package store

import (
	"archive/tar"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"

	"github.com/klauspost/compress/zstd"

	"github.com/shibukawa/hwnpu/internal/cache"
)

// An export for AMD's NPU or of Core AI goes in a source as one file (docs/cache-sharing-design.md
// 7.10): its files as a tar compressed with zstd, PackFile, beside entry.json, which lists the
// files inside (Files) and the pack itself (Entry.Pack). One file rather than a folder of them
// (each a PUT and a MOVE on WebDAV; a Core AI export of a chat model is about a hundred), and
// smaller: AMD's programs (ELF) compress to a 60th, the weights' banks by a fifth, a Core AI
// export by a few percent. It is unpacked here into the export's directory as it is fetched; the
// directory here keeps no pack. Core AI's prebuilt programs are packed by their own
// (coreai.Bundle).

// PackFile is the file in a source that holds a packed entry's files.
const PackFile = "files.tar.zst"

// Packs reports whether the entry at rel (EntryPath) goes in a source packed: an export for
// AMD's NPU or of Core AI.
func Packs(rel string) bool {
	parts := strings.Split(rel, "/")
	return len(parts) >= 3 && parts[0] == "converted" && (parts[2] == cache.AMDTarget || parts[2] == cache.CoreAITarget)
}

// packTo writes e's files (in dir) as a tar compressed with zstd at out and returns it as a File
// of the entry (PackFile).
func packTo(ctx context.Context, dir string, e Entry, out string) (File, error) {
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		return File{}, err
	}
	f, err := os.Create(out)
	if err != nil {
		return File{}, err
	}
	h := sha256.New()
	cw := &counter{w: io.MultiWriter(f, h), ctx: ctx}
	err = func() error {
		zw, err := zstd.NewWriter(cw, zstd.WithEncoderLevel(zstd.SpeedDefault))
		if err != nil {
			return err
		}
		tw := tar.NewWriter(zw)
		for _, x := range e.Files {
			p := filepath.Join(dir, filepath.FromSlash(x.Path))
			in, err := os.Open(p)
			if err != nil {
				zw.Close()
				return err
			}
			err = tw.WriteHeader(&tar.Header{Typeflag: tar.TypeReg, Name: x.Path, Mode: 0o644, Size: x.Size})
			if err == nil {
				_, err = io.Copy(tw, in)
			}
			in.Close()
			if err != nil {
				zw.Close()
				return fmt.Errorf("store: packing %s: %w", p, err)
			}
		}
		if err := tw.Close(); err != nil {
			zw.Close()
			return err
		}
		return zw.Close()
	}()
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(out)
		return File{}, err
	}
	return File{Path: PackFile, Size: cw.n, SHA256: hex.EncodeToString(h.Sum(nil))}, nil
}

// unpack writes the files of the pack at file into dir: only e's files, each checked against its
// size and SHA-256, every one of them. progress, if any, is told the bytes written so far.
func unpack(ctx context.Context, file string, e Entry, dir string, progress func(int64)) error {
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
	want := map[string]File{}
	for _, x := range e.Files {
		want[x.Path] = x
	}
	var done int64
	tr := tar.NewReader(zr)
	for {
		hd, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("store: %s: %w", filepath.Base(file), err)
		}
		if hd.Typeflag == tar.TypeDir {
			continue
		}
		x, ok := want[hd.Name]
		if !ok || hd.Typeflag != tar.TypeReg || !fs.ValidPath(hd.Name) {
			return fmt.Errorf("store: %s: %q is not a file of its %s", filepath.Base(file), hd.Name, EntryFile)
		}
		delete(want, hd.Name)
		p := filepath.Join(dir, filepath.FromSlash(hd.Name))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		out, err := os.Create(p)
		if err != nil {
			return err
		}
		h := sha256.New()
		base := done
		w := &counter{w: io.MultiWriter(out, h), ctx: ctx, bar: func(n, _ int64) {
			if progress != nil {
				progress(base + n)
			}
		}}
		n, err := io.Copy(w, tr)
		if cerr := out.Close(); err == nil {
			err = cerr
		}
		if err != nil {
			return err
		}
		done += n
		if n != x.Size {
			return fmt.Errorf("store: %s: %s of %d bytes, not the %d of %s", filepath.Base(file), x.Path, n, x.Size, EntryFile)
		}
		if got := hex.EncodeToString(h.Sum(nil)); got != x.SHA256 {
			return fmt.Errorf("store: %s: %s has the SHA-256 %s, not %s", filepath.Base(file), x.Path, got, x.SHA256)
		}
	}
	for name := range want {
		return fmt.Errorf("store: %s holds no %s", filepath.Base(file), name)
	}
	return nil
}

// packName is where a pack fetched into part is kept until it is unpacked.
const packName = ".pack-" + PackFile

// materializePacked is materialize for a packed entry: its pack is downloaded into dst.part (a
// download cut off goes on), unpacked there, and removed; dst.part is renamed to dst.
func materializePacked(ctx context.Context, s Source, e Entry, rel, dst string, log io.Writer) error {
	if !SafeKey(e.Pack.Path) || strings.Contains(e.Pack.Path, "/") || len(e.Pack.SHA256) != 64 {
		return fmt.Errorf("store: %s: a pack %q that is not a file of the entry", EntryFile, e.Pack.Path)
	}
	part := dst + ".part"
	if err := os.MkdirAll(part, 0o755); err != nil {
		return err
	}
	// what was unpacked before goes (a pack cut off while it was unpacked; an unpacked download)
	if des, err := os.ReadDir(part); err == nil {
		for _, de := range des {
			if !strings.HasPrefix(de.Name(), packName) {
				os.RemoveAll(filepath.Join(part, de.Name()))
			}
		}
	}
	total := e.Bytes()
	arch := filepath.Join(part, packName)
	have := false
	if fi, err := os.Stat(arch); err == nil && fi.Size() == e.Pack.Size {
		sum, err := FileSHA256(arch)
		have = err == nil && sum == e.Pack.SHA256
	}
	need := total
	if !have {
		need += e.Pack.Size
	}
	if err := cache.Room(need, dst, logf(log)); err != nil {
		return err
	}
	if !have {
		br, set := bar(log, fmt.Sprintf("downloading %s from %s", filepath.Base(dst), s))
		if err := getChecked(ctx, s.Store, rel+"/"+e.Pack.Path, e.Pack.SHA256, arch, set); err != nil {
			br.Done("failed")
			return err
		}
		br.Done(rate(e.Pack.Size, br.Elapsed()))
	}
	br, set := bar(log, "unpacking "+filepath.Base(dst))
	if err := unpack(ctx, arch, e, part, func(n int64) { set(n, total) }); err != nil {
		br.Done("failed")
		os.Remove(arch) // a pack that does not unpack is fetched again
		return err
	}
	os.Remove(arch)
	if err := WriteEntry(part, e); err != nil {
		br.Done("failed")
		return err
	}
	os.RemoveAll(dst)
	if err := os.Rename(part, dst); err != nil {
		br.Done("failed")
		return err
	}
	br.Done(rate(total, br.Elapsed()))
	return nil
}

// putPacked packs the files of dir (e's) into the work folder, puts the pack in s at rel, and
// removes the rest of the folder there but entry.json and the markers (the files of an entry
// put before unpacked). It returns the pack as a File of the entry.
func putPacked(ctx context.Context, s Source, rel, dir string, e Entry) (File, error) {
	tmp := cache.Sub("work", "pack", cache.Key([]byte(s.String()), []byte(rel))+".tar.zst")
	defer os.Remove(tmp)
	pf, err := packTo(ctx, dir, e, tmp)
	if err != nil {
		return File{}, err
	}
	logQueue("%s: packed %.2f GB into %.2f GB", filepath.Base(dir), float64(e.Bytes())/1e9, float64(pf.Size)/1e9)
	if err := s.Store.Put(ctx, rel+"/"+pf.Path, tmp); err != nil {
		return File{}, err
	}
	keys, err := s.Store.List(ctx, rel+"/")
	if err != nil {
		return File{}, err
	}
	for _, k := range keys {
		if k == rel+"/"+pf.Path || k == rel+"/"+EntryFile || isMarker(k) {
			continue
		}
		if err := s.Store.Delete(ctx, k); err != nil {
			return File{}, err
		}
	}
	return pf, nil
}
