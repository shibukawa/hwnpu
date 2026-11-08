package hwnpu

import "github.com/shibukawa/hwnpu/internal/vecops"

// EncodeF16 writes src as IEEE 754 half precision, little endian, into dst.
// dst must hold 2*len(src) bytes. Values are rounded to nearest even.
func EncodeF16(dst []byte, src []float32) {
	_ = dst[2*len(src)-1]
	vecops.F32ToF16(dst, src)
}

// DecodeF16 reads little endian half precision values from src into dst.
// src must hold 2*len(dst) bytes.
func DecodeF16(dst []float32, src []byte) {
	_ = src[2*len(dst)-1]
	vecops.F16ToF32(dst, src)
}

func f32ToF16(f float32) uint16 { return vecops.F16Bits(f) }

func f16ToF32(h uint16) float32 { return vecops.F16Value(h) }
