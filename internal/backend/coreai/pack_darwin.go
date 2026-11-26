package coreai

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"

	"github.com/shibukawa/hwnpu/internal/coreai/mlirbc"
)

// resMap stands for a file of a prebuilt program that is made of the bytes of its asset's
// bytecode (resources.bin: the weights, laid out in another order): the file is Ops in turn,
// each Len bytes of the bytecode from From, or, From -1, the next Len bytes of Literal (the
// few bytes between the weights).
type resMap struct {
	Format  int     `json:"format"`
	Source  string  `json:"source"` // the asset's file the bytes come from
	Size    int64   `json:"size"`
	SHA256  string  `json:"sha256"`
	Ops     []mapOp `json:"ops"`
	Literal []byte  `json:"literal,omitempty"`
}

type mapOp struct {
	From int64 `json:"from"`
	Len  int64 `json:"len"`
}

const resMapFormat = 1

// packMin is the smallest file Pack looks for in the bytecode.
const packMin = 1 << 20

// Pack writes the prebuilt program full (X.ARCH.aimodelc, of the asset at asset) to out without
// the files that are the asset's bytecode's bytes: each such file F is F.map.json (resMap). The
// other files are copied. A file whose bytes are mostly not the bytecode's is kept whole.
func Pack(full, asset, out string) error {
	src := filepath.Join(asset, "main.mlirb")
	sb, unmap, err := mmapFile(src)
	if err != nil {
		return err
	}
	defer unmap()
	var idx *blobIndex
	tmp := out + ".part"
	os.RemoveAll(tmp)
	err = filepath.WalkDir(full, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(full, p)
		dst := filepath.Join(tmp, rel)
		if d.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		if !d.Type().IsRegular() {
			return nil
		}
		fi, err := d.Info()
		if err != nil {
			return err
		}
		if fi.Size() >= packMin {
			if idx == nil {
				if idx, err = indexBlobs(sb); err != nil {
					return fmt.Errorf("%s: %w", src, err)
				}
			}
			m, err := mapFile(p, sb, idx)
			if err != nil {
				return err
			}
			if m != nil {
				m.Source = "main.mlirb"
				b, err := json.Marshal(m)
				if err != nil {
					return err
				}
				return os.WriteFile(dst+MapSuffix, b, 0o644)
			}
		}
		return copyFile(dst, p)
	})
	if err != nil {
		os.RemoveAll(tmp)
		return err
	}
	if err := os.MkdirAll(filepath.Dir(out), 0o755); err != nil {
		os.RemoveAll(tmp)
		return err
	}
	os.RemoveAll(out)
	return os.Rename(tmp, out)
}

// blobIndex finds where a run of bytes may start in the bytecode: the first 16 bytes of each
// resource blob and of every 64 KiB of one.
type blobIndex struct {
	at map[[16]byte][]int64
}

const (
	keyLen    = 16
	keyStride = 64 << 10
	minMatch  = 64
)

func indexBlobs(sb []byte) (*blobIndex, error) {
	f, err := mlirbc.Read(sb)
	if err != nil {
		return nil, err
	}
	base := uintptr(unsafe.Pointer(unsafe.SliceData(sb)))
	idx := &blobIndex{at: map[[16]byte][]int64{}}
	for _, rs := range f.Resources {
		for _, r := range rs {
			if len(r.Data) < minMatch {
				continue
			}
			off := int64(uintptr(unsafe.Pointer(unsafe.SliceData(r.Data))) - base)
			for k := int64(0); k+keyLen <= int64(len(r.Data)); k += keyStride {
				key := [16]byte(sb[off+k : off+k+keyLen])
				if l := idx.at[key]; len(l) < 8 { // a run of one value (zeros) starts anywhere
					idx.at[key] = append(l, off+k)
				}
			}
		}
	}
	return idx, nil
}

// mapFile maps the file at p onto the bytecode sb, nil when most of it is not there.
func mapFile(p string, sb []byte, idx *blobIndex) (*resMap, error) {
	db, unmap, err := mmapFile(p)
	if err != nil {
		return nil, err
	}
	defer unmap()
	m := &resMap{Format: resMapFormat, Size: int64(len(db))}
	sum := sha256.Sum256(db)
	m.SHA256 = hex.EncodeToString(sum[:])
	var lit []byte
	delta := int64(-1) // the last match's source minus file offset: a run cut short goes on there
	n := int64(len(db))
	for pos := int64(0); pos < n; {
		best, bestLen := int64(-1), int64(0)
		try := func(from int64) {
			if from < 0 || from >= int64(len(sb)) {
				return
			}
			if l := common(db[pos:], sb[from:]); l > bestLen {
				best, bestLen = from, l
			}
		}
		if delta >= 0 {
			try(pos + delta)
		}
		if bestLen < minMatch && pos+keyLen <= n {
			for _, from := range idx.at[[16]byte(db[pos:pos+keyLen])] {
				try(from)
			}
		}
		if bestLen < minMatch {
			lit = append(lit, db[pos])
			if k := len(m.Ops) - 1; k >= 0 && m.Ops[k].From == -1 {
				m.Ops[k].Len++
			} else {
				m.Ops = append(m.Ops, mapOp{From: -1, Len: 1})
			}
			pos++
			if int64(len(lit)) > n/16 {
				return nil, nil
			}
			continue
		}
		m.Ops = append(m.Ops, mapOp{From: best, Len: bestLen})
		delta = best - pos
		pos += bestLen
	}
	m.Literal = lit
	return m, nil
}

// common is the length of the common prefix of a and b.
func common(a, b []byte) int64 {
	n := min(len(a), len(b))
	i := 0
	for i+4096 <= n && bytes.Equal(a[i:i+4096], b[i:i+4096]) {
		i += 4096
	}
	for i+8 <= n && binary.LittleEndian.Uint64(a[i:]) == binary.LittleEndian.Uint64(b[i:]) {
		i += 8
	}
	for i < n && a[i] == b[i] {
		i++
	}
	return int64(i)
}

// Assemble makes the packed prebuilt program packed (of the asset at asset) whole in out: its
// files hard linked (else copied), each F.map.json made F again from the asset's bytecode, and
// checked.
func Assemble(packed, asset, out string) error {
	os.RemoveAll(out)
	err := filepath.WalkDir(packed, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(packed, p)
		dst := filepath.Join(out, rel)
		if d.IsDir() {
			return os.MkdirAll(dst, 0o755)
		}
		if !d.Type().IsRegular() {
			return nil
		}
		if name, ok := strings.CutSuffix(dst, MapSuffix); ok {
			return unmapFile(p, asset, name)
		}
		if os.Link(p, dst) == nil {
			return nil
		}
		return copyFile(dst, p)
	})
	if err != nil {
		os.RemoveAll(out)
	}
	return err
}

// unmapFile makes the file dst of the map at p from the asset's file the map names.
func unmapFile(p, asset, dst string) error {
	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	err = replayMap(p, asset, f)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(dst)
	}
	return err
}

// replayMap writes the file of the map at p, made from the asset's file the map names, to w,
// and checks it is the size and SHA-256 the map names.
func replayMap(p, asset string, w io.Writer) error {
	b, err := os.ReadFile(p)
	if err != nil {
		return err
	}
	var m resMap
	if err := json.Unmarshal(b, &m); err != nil {
		return fmt.Errorf("%s: %w", p, err)
	}
	if m.Format != resMapFormat || m.Source == "" || strings.ContainsAny(m.Source, `/\`) {
		return fmt.Errorf("%s: not a map of format %d", p, resMapFormat)
	}
	src, err := os.Open(filepath.Join(asset, m.Source))
	if err != nil {
		return err
	}
	defer src.Close()
	h := sha256.New()
	w = io.MultiWriter(w, h)
	buf := make([]byte, 4<<20)
	var lit, size int64
	for _, op := range m.Ops {
		if op.Len < 0 {
			err = errors.New("a negative length")
			break
		}
		if op.From < 0 {
			if lit+op.Len > int64(len(m.Literal)) {
				err = errors.New("past its literal bytes")
				break
			}
			_, err = w.Write(m.Literal[lit : lit+op.Len])
			lit += op.Len
		} else {
			var n int64
			n, err = io.CopyBuffer(w, io.NewSectionReader(src, op.From, op.Len), buf)
			if err == nil && n != op.Len {
				err = fmt.Errorf("%d bytes past the end of %s", op.Len-n, m.Source)
			}
		}
		if err != nil {
			break
		}
		size += op.Len
	}
	if err == nil && (size != m.Size || hex.EncodeToString(h.Sum(nil)) != m.SHA256) {
		err = fmt.Errorf("made %d bytes not of the SHA-256 it names (another bytecode?)", size)
	}
	if err != nil {
		return fmt.Errorf("%s: %w", filepath.Base(p), err)
	}
	return nil
}

// checkPacked checks that every map of the packed prebuilt program packed makes its file from
// the asset's bytecode at asset (what Assemble would make), reading the bytecode once a map.
func checkPacked(packed, asset string) error {
	return filepath.WalkDir(packed, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(p, MapSuffix) {
			return err
		}
		return replayMap(p, asset, io.Discard)
	})
}

// copyDir copies the directory src to dst (which must not be there), files cloned where the
// volume can (APFS), else copied.
func copyDir(dst, src string) error {
	return filepath.WalkDir(src, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, p)
		to := filepath.Join(dst, rel)
		if d.IsDir() {
			return os.MkdirAll(to, 0o755)
		}
		if !d.Type().IsRegular() {
			return nil
		}
		if unix.Clonefile(p, to, 0) == nil {
			return nil
		}
		return copyFile(to, p)
	})
}

func copyFile(dst, src string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

// mmapFile maps the file at p for reading.
func mmapFile(p string) ([]byte, func(), error) {
	f, err := os.Open(p)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, nil, err
	}
	if fi.Size() == 0 {
		return nil, func() {}, nil
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(fi.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", p, err)
	}
	return b, func() { syscall.Munmap(b) }, nil
}
