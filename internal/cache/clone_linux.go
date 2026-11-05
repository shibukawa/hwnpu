package cache

import (
	"encoding/binary"
	"os"
	"runtime"
	"syscall"
	"unsafe"
)

const canClone = true

// ioctlW is _IOW(0x94, nr, size): the architectures with their own encoding of ioctls put the
// direction in other bits.
func ioctlW(nr, size uintptr) uintptr {
	write := uintptr(1) << 30 // _IOC_WRITE << _IOC_DIRSHIFT
	switch runtime.GOARCH {
	case "ppc64", "ppc64le", "mips", "mipsle", "mips64", "mips64le", "sparc64":
		write = 1 << 31 // 4 << 29
	}
	return write | size<<16 | 0x94<<8 | nr
}

var (
	ficlone      = ioctlW(9, 4)   // FICLONE, the whole file (an int: the source's descriptor)
	ficlonerange = ioctlW(13, 32) // FICLONERANGE (struct file_clone_range)
)

// clonefile makes dst, which must not exist, a clone of src with the ioctl FICLONE: the file
// system shares src's blocks (Btrfs, XFS with reflink, bcachefs, ZFS with block cloning) or
// refuses (ext4, tmpfs, another file system than src's), in which case dst is removed again.
func clonefile(src, dst string) bool {
	s, err := os.Open(src)
	if err != nil {
		return false
	}
	defer s.Close()
	d, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return false
	}
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, d.Fd(), ficlone, s.Fd())
	if err := d.Close(); errno != 0 || err != nil {
		os.Remove(dst)
		return false
	}
	return true
}

// cloneRange makes the n bytes of dst at off share the blocks of the same bytes of src
// (FICLONERANGE). off and n are whole blocks of the file system (rangeBlock).
func cloneRange(src, dst *os.File, off, n int64) bool {
	// struct file_clone_range: src_fd (s64), src_offset, src_length, dest_offset (u64)
	var arg [32]byte
	binary.NativeEndian.PutUint64(arg[0:], uint64(src.Fd()))
	binary.NativeEndian.PutUint64(arg[8:], uint64(off))
	binary.NativeEndian.PutUint64(arg[16:], uint64(n))
	binary.NativeEndian.PutUint64(arg[24:], uint64(off))
	_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, dst.Fd(), ficlonerange, uintptr(unsafe.Pointer(&arg[0])))
	runtime.KeepAlive(src)
	return errno == 0
}

// rangeBlock is the block of f's file system that ranges to clone are made of.
func rangeBlock(f *os.File) int64 {
	var st syscall.Statfs_t
	if err := syscall.Fstatfs(int(f.Fd()), &st); err != nil || st.Bsize <= 0 {
		return 0
	}
	return int64(st.Bsize)
}

// fsIocFiemap is the ioctl FS_IOC_FIEMAP, _IOWR('f', 11, struct fiemap), the same number on every
// architecture.
const fsIocFiemap = 0xC020660B

// fiemapExtents is how many extents one FS_IOC_FIEMAP call asks for.
const fiemapExtents = 64

// extent is where a run of a file's bytes lies on the disk (struct fiemap_extent).
type extent struct {
	logical, physical, length uint64
	flags                     uint32
}

// extents lists where the first limit bytes of the file at path lie on the disk (FS_IOC_FIEMAP).
func extents(path string, limit int64) ([]extent, bool) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false
	}
	defer f.Close()
	// struct fiemap is 32 bytes, each struct fiemap_extent 56: fe_logical, fe_physical and
	// fe_length (u64), two reserved u64, fe_flags (u32) and three reserved u32
	const head, size, last = 32, 56, 0x1 // FIEMAP_EXTENT_LAST
	buf := make([]byte, head+fiemapExtents*size)
	var out []extent
	for start := uint64(0); start < uint64(limit); {
		clear(buf)
		binary.NativeEndian.PutUint64(buf[0:], start)
		binary.NativeEndian.PutUint64(buf[8:], uint64(limit)-start)
		binary.NativeEndian.PutUint32(buf[24:], fiemapExtents)
		_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, f.Fd(), fsIocFiemap, uintptr(unsafe.Pointer(&buf[0])))
		if errno != 0 {
			return nil, false
		}
		n := int(binary.NativeEndian.Uint32(buf[20:]))
		if n == 0 {
			break
		}
		for i := range n {
			e := buf[head+i*size:]
			out = append(out, extent{binary.NativeEndian.Uint64(e[0:]), binary.NativeEndian.Uint64(e[8:]), binary.NativeEndian.Uint64(e[16:]), binary.NativeEndian.Uint32(e[40:])})
		}
		if l := out[len(out)-1]; l.flags&last != 0 {
			break
		} else {
			start = l.logical + l.length
		}
	}
	return out, true
}

// sharedPrefix is how many of the first limit bytes of the files a and b lie on the same blocks of
// the disk: a start that one shares from the other, which Clone and ClonePrefix need not read
// again. It reads where their extents lie (FS_IOC_FIEMAP), not their bytes; it counts only extents
// whose place is exact (not compressed, inline or still to be written), so it may say less than
// is shared, never more.
func sharedPrefix(a, b string, limit int64) int64 {
	if limit <= 0 {
		return 0
	}
	ea, okA := extents(a, limit)
	eb, okB := extents(b, limit)
	if !okA || !okB {
		return 0
	}
	// FIEMAP_EXTENT_UNKNOWN, _DELALLOC, _ENCODED, _DATA_ENCRYPTED, _DATA_INLINE, _DATA_TAIL
	const inexact = 0x2 | 0x4 | 0x8 | 0x80 | 0x200 | 0x400
	at := func(es []extent, i *int, cur uint64) (extent, bool) {
		for *i < len(es) && es[*i].logical+es[*i].length <= cur {
			*i++
		}
		if *i == len(es) || es[*i].logical > cur || es[*i].flags&inexact != 0 {
			return extent{}, false // a hole, the end, or an extent without an exact place
		}
		return es[*i], true
	}
	var cur uint64
	for i, j := 0, 0; cur < uint64(limit); {
		x, okX := at(ea, &i, cur)
		y, okY := at(eb, &j, cur)
		if !okX || !okY || x.physical+(cur-x.logical) != y.physical+(cur-y.logical) {
			break
		}
		cur = min(x.logical+x.length, y.logical+y.length)
	}
	return min(int64(cur), limit)
}
