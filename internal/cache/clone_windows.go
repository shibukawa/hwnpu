package cache

import (
	"encoding/binary"
	"os"
	"syscall"
)

const canClone = true

// The file system controls of ReFS's block cloning (Windows Server 2016 and later, and the Dev
// Drive of Windows 11): CTL_CODE(FILE_DEVICE_FILE_SYSTEM, function, METHOD_BUFFERED, access).
const (
	fsctlGetIntegrityInformation = 0x9027C // 159, any access
	fsctlSetIntegrityInformation = 0x9C280 // 160, read and write data
	fsctlSetSparse               = 0x900C4 // 49, any access
	fsctlDuplicateExtentsToFile  = 0x98344 // 209, write data
	fileAttributeSparseFile      = 0x200
	// duplicateChunk is the most bytes one FSCTL_DUPLICATE_EXTENTS_TO_FILE is asked to share:
	// one call must stay under 4 GiB.
	duplicateChunk = 1 << 30
)

// integrity reads FSCTL_GET_INTEGRITY_INFORMATION_BUFFER (ChecksumAlgorithm u16, Reserved u16,
// Flags u32, ChecksumChunkSizeInBytes u32, ClusterSizeInBytes u32) of f: only ReFS answers.
func integrity(f *os.File) ([16]byte, bool) {
	var out [16]byte
	var n uint32
	err := syscall.DeviceIoControl(syscall.Handle(f.Fd()), fsctlGetIntegrityInformation, nil, 0, &out[0], uint32(len(out)), &n, nil)
	return out, err == nil && n >= 16
}

// rangeBlock is the cluster of f's volume, which ranges to clone are made of; 0 where the volume
// cannot clone (NTFS, FAT).
func rangeBlock(f *os.File) int64 {
	info, ok := integrity(f)
	if !ok {
		return 0
	}
	return int64(binary.LittleEndian.Uint32(info[12:]))
}

// duplicate makes n bytes of dst at off share src's clusters at the same offset
// (FSCTL_DUPLICATE_EXTENTS_TO_FILE), in calls of at most duplicateChunk.
func duplicate(src, dst *os.File, off, n int64) bool {
	for n > 0 {
		c := min(n, duplicateChunk)
		// DUPLICATE_EXTENTS_DATA: FileHandle (a HANDLE in 8 bytes, 4 and padding on 386),
		// SourceFileOffset, TargetFileOffset, ByteCount (LARGE_INTEGER)
		var arg [32]byte
		binary.LittleEndian.PutUint64(arg[0:], uint64(src.Fd()))
		binary.LittleEndian.PutUint64(arg[8:], uint64(off))
		binary.LittleEndian.PutUint64(arg[16:], uint64(off))
		binary.LittleEndian.PutUint64(arg[24:], uint64(c))
		var ret uint32
		if syscall.DeviceIoControl(syscall.Handle(dst.Fd()), fsctlDuplicateExtentsToFile, &arg[0], uint32(len(arg)), nil, 0, &ret, nil) != nil {
			return false
		}
		off, n = off+c, n-c
	}
	return true
}

// cloneRange makes the n bytes of dst at off share the clusters of the same bytes of src. off and
// n are whole clusters (rangeBlock).
func cloneRange(src, dst *os.File, off, n int64) bool { return duplicate(src, dst, off, n) }

// clonefile makes dst, which must not exist, a clone of src with ReFS's block cloning: dst gets
// src's integrity streams and sparseness (the two must agree), its length, and then src's
// clusters, the last one whole past the end as the call asks. Any other volume refuses, and dst
// is removed again.
func clonefile(src, dst string) bool {
	s, err := os.Open(src)
	if err != nil {
		return false
	}
	defer s.Close()
	info, ok := integrity(s)
	if !ok {
		return false
	}
	cluster := int64(binary.LittleEndian.Uint32(info[12:]))
	var fi syscall.ByHandleFileInformation
	if cluster <= 0 || syscall.GetFileInformationByHandle(syscall.Handle(s.Fd()), &fi) != nil {
		return false
	}
	size := int64(fi.FileSizeHigh)<<32 | int64(fi.FileSizeLow)
	d, err := os.OpenFile(dst, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return false
	}
	ok = func() bool {
		var ret uint32
		h := syscall.Handle(d.Fd())
		if fi.FileAttributes&fileAttributeSparseFile != 0 && syscall.DeviceIoControl(h, fsctlSetSparse, nil, 0, nil, 0, &ret, nil) != nil {
			return false
		}
		// FSCTL_SET_INTEGRITY_INFORMATION_BUFFER: ChecksumAlgorithm, Reserved, Flags
		var set [8]byte
		copy(set[0:2], info[0:2])
		copy(set[4:8], info[4:8])
		syscall.DeviceIoControl(h, fsctlSetIntegrityInformation, &set[0], uint32(len(set)), nil, 0, &ret, nil)
		if d.Truncate(size) != nil {
			return false
		}
		return duplicate(s, d, 0, (size+cluster-1)/cluster*cluster)
	}()
	if err := d.Close(); !ok || err != nil {
		os.Remove(dst)
		return false
	}
	return true
}

// sharedPrefix would say how much of two files' start lies on the same clusters; Windows has no
// cheap way to ask, so ClonePrefix and Clone compare their bytes again (once for each export).
func sharedPrefix(a, b string, limit int64) int64 { return 0 }
