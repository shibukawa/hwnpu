package mlpackage

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
)

// BlobDType is the data type tag of a weight blob (MILBlob/Blob/BlobDataType.hpp).
type BlobDType uint32

const (
	BlobFloat16 BlobDType = 1
	BlobFloat32 BlobDType = 2
	BlobUInt8   BlobDType = 3
	BlobInt8    BlobDType = 4
	BlobInt4    BlobDType = 8
	BlobUInt4   BlobDType = 11
	BlobUInt6   BlobDType = 13
	BlobInt32   BlobDType = 14
)

const blobAlign = 64

// BlobWriter writes weights/weight.bin in the MILBlob storage format: a 64 byte header, then for
// every blob a 64 byte metadata record followed by the data, each 64 byte aligned. The offset a
// MIL const refers to is that of the metadata record.
type BlobWriter struct {
	w     *os.File
	pos   int64
	count uint32
	err   error
}

// NewBlobWriter creates the file. The header is written on Close, when the count is known.
func NewBlobWriter(path string) (*BlobWriter, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	b := &BlobWriter{w: f}
	b.write(make([]byte, blobAlign))
	return b, b.err
}

// Size is the number of bytes written so far.
func (b *BlobWriter) Size() int64 { return b.pos }

func (b *BlobWriter) write(p []byte) {
	if b.err != nil {
		return
	}
	n, err := b.w.Write(p)
	b.pos += int64(n)
	b.err = err
}

func (b *BlobWriter) pad() {
	if rem := b.pos % blobAlign; rem != 0 {
		b.write(make([]byte, blobAlign-rem))
	}
}

// Add appends one blob and returns the offset of its metadata record. For sub byte types the data
// is already packed (two uint4 values per byte, low nibble first).
func (b *BlobWriter) Add(dtype BlobDType, data []byte) (uint64, error) {
	if b.err != nil {
		return 0, b.err
	}
	b.pad()
	off := uint64(b.pos)
	meta := make([]byte, blobAlign)
	binary.LittleEndian.PutUint32(meta[0:], 0xDEADBEEF)
	binary.LittleEndian.PutUint32(meta[4:], uint32(dtype))
	binary.LittleEndian.PutUint64(meta[8:], uint64(len(data)))
	binary.LittleEndian.PutUint64(meta[16:], off+blobAlign)
	b.write(meta)
	b.write(data)
	b.count++
	return off, b.err
}

// Close writes the header and closes the file.
func (b *BlobWriter) Close() error {
	if b.err == nil {
		b.pad()
		hdr := make([]byte, blobAlign)
		binary.LittleEndian.PutUint32(hdr[0:], b.count)
		binary.LittleEndian.PutUint32(hdr[4:], 2)
		if _, err := b.w.WriteAt(hdr, 0); err != nil {
			b.err = err
		}
	}
	if err := b.w.Close(); err != nil && b.err == nil {
		b.err = err
	}
	return b.err
}

// ReadBlob reads one blob back: the data type and the data, given the metadata offset.
func ReadBlob(r io.ReaderAt, off uint64) (BlobDType, []byte, error) {
	meta := make([]byte, blobAlign)
	if _, err := r.ReadAt(meta, int64(off)); err != nil {
		return 0, nil, err
	}
	if binary.LittleEndian.Uint32(meta) != 0xDEADBEEF {
		return 0, nil, fmt.Errorf("mlpackage: no blob at offset %d", off)
	}
	size := binary.LittleEndian.Uint64(meta[8:])
	data := make([]byte, size)
	if _, err := r.ReadAt(data, int64(binary.LittleEndian.Uint64(meta[16:]))); err != nil {
		return 0, nil, err
	}
	return BlobDType(binary.LittleEndian.Uint32(meta[4:])), data, nil
}
