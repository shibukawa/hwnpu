// Package binview provides generic helpers for little-endian byte slices, zero-copy aligned views,
// and numeric slice element conversions.
package binview

import (
	"encoding/binary"
	"math"
	"unsafe"
)

// FixedLE is the constraint of fixed-width multi-byte scalar types stored in little-endian layout.
type FixedLE interface {
	~uint16 | ~int16 | ~uint32 | ~int32 | ~uint64 | ~int64 | ~float32 | ~float64
}

// Number is the constraint of built-in integer and floating-point scalar types.
type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~float32 | ~float64
}

// NativeLE reports whether the host CPU stores multi-byte integers in little-endian byte order.
var NativeLE = binary.NativeEndian.Uint16([]byte{1, 0}) == 1

// NativeLEView returns a zero-copy []T view backed by b when the host is little-endian, len(b) is
// a multiple of sizeof(T), and &b[0] is aligned to sizeof(T). When len(b) == 0, it returns an
// empty non-nil slice and true on little-endian hosts.
func NativeLEView[T FixedLE](b []byte) ([]T, bool) {
	if !NativeLE {
		return nil, false
	}
	var z T
	size := int(unsafe.Sizeof(z))
	if len(b)%size != 0 {
		return nil, false
	}
	if len(b) == 0 {
		return []T{}, true
	}
	if uintptr(unsafe.Pointer(&b[0]))%uintptr(size) != 0 {
		return nil, false
	}
	return unsafe.Slice((*T)(unsafe.Pointer(&b[0])), len(b)/size), true
}

// EncodeLE encodes v into a newly allocated little-endian byte slice.
func EncodeLE[T FixedLE](v []T) []byte {
	var z T
	size := int(unsafe.Sizeof(z))
	b := make([]byte, size*len(v))
	if len(v) == 0 {
		return b
	}
	if NativeLE {
		src := unsafe.Slice((*byte)(unsafe.Pointer(&v[0])), len(b))
		copy(b, src)
		return b
	}
	switch any(z).(type) {
	case uint16, int16:
		for i, x := range v {
			u := *(*uint16)(unsafe.Pointer(&x))
			binary.LittleEndian.PutUint16(b[2*i:], u)
		}
	case uint32, int32, float32:
		for i, x := range v {
			u := *(*uint32)(unsafe.Pointer(&x))
			binary.LittleEndian.PutUint32(b[4*i:], u)
		}
	case uint64, int64, float64:
		for i, x := range v {
			u := *(*uint64)(unsafe.Pointer(&x))
			binary.LittleEndian.PutUint64(b[8*i:], u)
		}
	}
	return b
}

// DecodeLE decodes a little-endian byte slice into a newly allocated []T. It panics if len(b) is
// not a multiple of sizeof(T).
func DecodeLE[T FixedLE](b []byte) []T {
	var z T
	size := int(unsafe.Sizeof(z))
	out := make([]T, len(b)/size)
	if len(out) == 0 {
		return out
	}
	if NativeLE {
		dst := unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), len(out)*size)
		copy(dst, b[:len(dst)])
		return out
	}
	switch any(z).(type) {
	case uint16, int16:
		for i := range out {
			u := binary.LittleEndian.Uint16(b[2*i:])
			out[i] = *(*T)(unsafe.Pointer(&u))
		}
	case uint32, int32:
		for i := range out {
			u := binary.LittleEndian.Uint32(b[4*i:])
			out[i] = *(*T)(unsafe.Pointer(&u))
		}
	case float32:
		for i := range out {
			f := math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
			out[i] = *(*T)(unsafe.Pointer(&f))
		}
	case uint64, int64:
		for i := range out {
			u := binary.LittleEndian.Uint64(b[8*i:])
			out[i] = *(*T)(unsafe.Pointer(&u))
		}
	case float64:
		for i := range out {
			f := math.Float64frombits(binary.LittleEndian.Uint64(b[8*i:]))
			out[i] = *(*T)(unsafe.Pointer(&f))
		}
	}
	return out
}

// Convert converts each element of src from Src to Dst.
func Convert[Dst, Src Number](src []Src) []Dst {
	out := make([]Dst, len(src))
	for i, v := range src {
		out[i] = Dst(v)
	}
	return out
}
