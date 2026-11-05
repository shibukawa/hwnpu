package cache

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

// Clone makes dst a clone of src when the two are regular files with the same bytes: dst is
// replaced by a copy that shares src's blocks on the disk (clonefile on macOS's APFS, FICLONE on
// Linux's Btrfs, XFS made with reflink, bcachefs and ZFS with block cloning, block cloning on
// Windows' ReFS and Dev Drive), so that it takes no
// room of its own. Both stay ordinary files, read and mapped as before; writing either one later
// gives it blocks of its own. dst keeps its mode and its modification time. It reports whether
// dst was replaced: not when the files differ, are already one file or clones of each other (seen
// on Linux only), or the file system (or the system: only macOS, Linux and Windows' ReFS so far)
// cannot clone, in which case nothing changes.
//
// The clone is made first and compared with dst, not src: a clone costs no more than its
// metadata, a file system that cannot clone says so before a byte is read, and a src written
// while the bytes are compared cannot slip into dst.
func Clone(dst, src string) (bool, error) {
	if !canClone {
		return false, nil
	}
	di, err := os.Stat(dst)
	if err != nil {
		return false, err
	}
	si, err := os.Stat(src)
	if err != nil {
		return false, err
	}
	if !di.Mode().IsRegular() || !si.Mode().IsRegular() || di.Size() != si.Size() || os.SameFile(di, si) || sharedPrefix(dst, src, di.Size()) >= di.Size() {
		return false, nil
	}
	// the process's own name: two processes sharing one directory at once do not swap clones
	tmp := filepath.Join(filepath.Dir(dst), fmt.Sprintf(".%s.%d.clone", filepath.Base(dst), os.Getpid()))
	os.Remove(tmp)
	if !clonefile(src, tmp) {
		return false, nil // another volume, or a file system without clones
	}
	same, err := sameBytes(dst, tmp)
	if err == nil && same {
		err = os.Chmod(tmp, di.Mode().Perm())
	}
	if err == nil && same {
		err = os.Chtimes(tmp, time.Time{}, di.ModTime())
	}
	if err == nil && same {
		err = os.Rename(tmp, dst)
	}
	if err != nil || !same {
		os.Remove(tmp)
		return false, err
	}
	return true, nil
}

// minPrefix is the least ClonePrefix shares: less is not worth a file replaced.
const minPrefix = 1 << 20

// ClonePrefix makes the start of dst share src's blocks on the disk where the two files begin
// with the same bytes: the longest common start, in whole blocks of the file system, of files
// that differ in their size or past it (the bank of weights of an export without the image and
// audio towers is the start of the one with them). dst is replaced by a clone of itself whose
// start is src's, as Clone replaces it, and keeps its bytes, mode and modification time. It
// reports the bytes newly shared: 0 when the files have no common start of minPrefix, already
// share it (seen on Linux only), or the system cannot clone ranges (FICLONERANGE on Linux,
// FSCTL_DUPLICATE_EXTENTS_TO_FILE on Windows' ReFS; not on macOS).
//
// As Clone, it makes the clone first and compares it with dst: a whole clone of dst, src's
// blocks over its start, the bytes compared with dst's to where they first differ, and dst's
// own blocks back from the block where they do.
func ClonePrefix(dst, src string) (int64, error) {
	if !canClone {
		return 0, nil
	}
	di, err := os.Stat(dst)
	if err != nil {
		return 0, err
	}
	si, err := os.Stat(src)
	if err != nil {
		return 0, err
	}
	if !di.Mode().IsRegular() || !si.Mode().IsRegular() || os.SameFile(di, si) {
		return 0, nil
	}
	d, err := os.Open(dst)
	if err != nil {
		return 0, err
	}
	bs := rangeBlock(d)
	d.Close()
	if bs <= 0 {
		return 0, nil
	}
	end := min(di.Size(), si.Size())
	end -= end % bs
	have := sharedPrefix(dst, src, end)
	have -= have % bs
	if end-have < minPrefix {
		return 0, nil
	}
	tmp := filepath.Join(filepath.Dir(dst), fmt.Sprintf(".%s.%d.clone", filepath.Base(dst), os.Getpid()))
	os.Remove(tmp)
	if !clonefile(dst, tmp) {
		return 0, nil
	}
	n, err := func() (int64, error) {
		files := make([]*os.File, 3)
		for i, p := range []string{dst, src, tmp} {
			flag := os.O_RDONLY
			if p == tmp {
				flag = os.O_RDWR
			}
			if files[i], err = os.OpenFile(p, flag, 0); err != nil {
				return 0, err
			}
			defer files[i].Close()
		}
		d, s, t := files[0], files[1], files[2]
		if !cloneRange(s, t, have, end-have) {
			return 0, nil
		}
		n, err := commonPrefix(t, d, have, end)
		if err != nil {
			return 0, err
		}
		n -= n % bs
		if n-have < minPrefix {
			return 0, nil
		}
		if n < end && !cloneRange(d, t, n, end-n) { // dst's own blocks back where they differ
			return 0, nil
		}
		return n, nil
	}()
	if err == nil && n > 0 {
		err = os.Chmod(tmp, di.Mode().Perm())
	}
	if err == nil && n > 0 {
		err = os.Chtimes(tmp, time.Time{}, di.ModTime())
	}
	if err == nil && n > 0 {
		err = os.Rename(tmp, dst)
	}
	if err != nil || n == 0 {
		os.Remove(tmp)
		return 0, err
	}
	return n - have, nil
}

// commonPrefix is the offset in [from, to) where the files a and b first differ, to if nowhere.
func commonPrefix(a, b *os.File, from, to int64) (int64, error) {
	ba, bb := make([]byte, 1<<20), make([]byte, 1<<20)
	for off := from; off < to; {
		n := int(min(int64(len(ba)), to-off))
		if _, err := a.ReadAt(ba[:n], off); err != nil {
			return off, err
		}
		if _, err := b.ReadAt(bb[:n], off); err != nil {
			return off, err
		}
		if !bytes.Equal(ba[:n], bb[:n]) {
			for i := range n {
				if ba[i] != bb[i] {
					return off + int64(i), nil
				}
			}
		}
		off += int64(n)
	}
	return to, nil
}

// sameBytes compares two files of the same size, stopping at the first difference.
func sameBytes(a, b string) (bool, error) {
	fa, err := os.Open(a)
	if err != nil {
		return false, err
	}
	defer fa.Close()
	fb, err := os.Open(b)
	if err != nil {
		return false, err
	}
	defer fb.Close()
	ba, bb := make([]byte, 1<<20), make([]byte, 1<<20)
	for {
		na, ea := io.ReadFull(fa, ba)
		nb, eb := io.ReadFull(fb, bb)
		if na != nb || !bytes.Equal(ba[:na], bb[:nb]) {
			return false, nil
		}
		if ea == io.EOF || ea == io.ErrUnexpectedEOF {
			return eb == ea, nil
		}
		if ea != nil {
			return false, ea
		}
		if eb != nil {
			return false, eb
		}
	}
}

// CanClone reports whether this system can clone files at all (Clone; the file system decides
// for each pair, CanCloneIn).
func CanClone() bool { return canClone }

// CanCloneIn reports whether the file system of dir clones files, by cloning a small file in it.
func CanCloneIn(dir string) bool {
	if !canClone {
		return false
	}
	f, err := os.CreateTemp(dir, ".clone-probe-*")
	if err != nil {
		return false
	}
	_, err = f.WriteString("hwnpu")
	f.Close()
	defer os.Remove(f.Name())
	if err != nil {
		return false
	}
	dst := f.Name() + ".clone"
	os.Remove(dst)
	defer os.Remove(dst)
	return clonefile(f.Name(), dst)
}
