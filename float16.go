package hwnpu

import (
	"encoding/binary"
	"math"
)

// EncodeF16 writes src as IEEE 754 half precision, little endian, into dst.
// dst must hold 2*len(src) bytes. Values are rounded to nearest even.
func EncodeF16(dst []byte, src []float32) {
	_ = dst[2*len(src)-1]
	for i, v := range src {
		binary.LittleEndian.PutUint16(dst[2*i:], f32ToF16(v))
	}
}

// DecodeF16 reads little endian half precision values from src into dst.
// src must hold 2*len(dst) bytes.
func DecodeF16(dst []float32, src []byte) {
	_ = src[2*len(dst)-1]
	for i := range dst {
		dst[i] = f16ToF32(binary.LittleEndian.Uint16(src[2*i:]))
	}
}

func f32ToF16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23) & 0xff
	mant := b & 0x7fffff
	switch {
	case exp == 0xff: // Inf or NaN
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp-127+15 >= 0x1f: // overflow
		return sign | 0x7c00
	case exp-127+15 <= 0: // subnormal or zero
		e := exp - 127 + 15
		if e < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint32(14 - e)
		h := mant >> shift
		rem := mant & (1<<shift - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && h&1 == 1) {
			h++
		}
		return sign | uint16(h)
	}
	h := uint32(exp-127+15)<<10 | mant>>13
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && h&1 == 1) {
		h++ // may carry into the exponent, which is correct rounding
	}
	return sign | uint16(h)
}

func f16ToF32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch {
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | mant<<13)
	case exp == 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (mant&0x3ff)<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}
