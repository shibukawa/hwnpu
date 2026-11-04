package cache

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
)

// Clone makes dst a clone of src when the two are regular files with the same bytes: dst is
// replaced by a copy that shares src's blocks on the disk (APFS's clonefile), so that it takes no
// room of its own. Both stay ordinary files, read and mapped as before; writing either one later
// gives it blocks of its own. It reports whether dst was replaced: not when the files differ, are
// already one file, or the file system (or the system: only macOS so far) cannot clone, in which
// case nothing changes.
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
	if !di.Mode().IsRegular() || !si.Mode().IsRegular() || di.Size() != si.Size() || os.SameFile(di, si) {
		return false, nil
	}
	if same, err := sameBytes(dst, src); err != nil || !same {
		return false, err
	}
	tmp := filepath.Join(filepath.Dir(dst), "."+filepath.Base(dst)+".clone")
	os.Remove(tmp)
	if !clonefile(src, tmp) {
		return false, nil // another volume, or a file system without clones
	}
	if err := os.Chmod(tmp, di.Mode().Perm()); err != nil {
		os.Remove(tmp)
		return false, err
	}
	if err := os.Rename(tmp, dst); err != nil {
		os.Remove(tmp)
		return false, err
	}
	return true, nil
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
// for each pair).
func CanClone() bool { return canClone }
