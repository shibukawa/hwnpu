package mxfp4

import (
	"encoding/binary"
	"math"
	"testing"
)

func f16ToF32(u uint16) float32 {
	sign := uint32(u&0x8000) << 16
	exp := int((u >> 10) & 0x1F)
	mant := uint32(u & 0x03FF)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		for (mant & 0x0400) == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x03FF
		return math.Float32frombits(sign | uint32(exp+127-15)<<23 | (mant << 13))
	case 31:
		return math.Float32frombits(sign | 0x7F800000 | (mant << 13))
	default:
		return math.Float32frombits(sign | uint32(exp+127-15)<<23 | (mant << 13))
	}
}

func TestDequantizeF16AllExponents(t *testing.T) {
	// Test every scale exponent e in 0..254 and every byte value 0..255 across 16 blocks (256 bytes = all byte values).
	w := make([]byte, 16*16)
	for i := range w {
		w[i] = byte(i)
	}
	e := make([]byte, 16)
	dst := make([]byte, 16*64)
	wantF32 := make([]float32, 16*32)

	for baseE := 0; baseE < 255; baseE += 16 {
		for b := range 16 {
			eb := min(baseE+b, 254)
			e[b] = byte(eb)
		}
		DequantizeF16(dst, w, e)
		Dequantize(wantF32, w, e)
		for i, wf := range wantF32 {
			gotU16 := binary.LittleEndian.Uint16(dst[2*i : 2*i+2])
			wantU16 := f32ToF16Bits(wf)
			if wf == 0 {
				wantU16 = 0
			}
			if gotU16 != wantU16 {
				t.Fatalf("e=%d idx=%d: got 0x%04x (%g), want 0x%04x (%g)",
					e[i/32], i, gotU16, f16ToF32(gotU16), wantU16, wf)
			}
		}
	}
}

func BenchmarkDequantizeF16(b *testing.B) {
	// 1 Qwen3.6 gate_up matrix: 1024 x 2048 = 2M weights (1 MB MXFP4 -> 4 MB FP16)
	const rows, cols = 1024, 2048
	nb := rows * (cols / 32)
	w := make([]byte, nb*16)
	e := make([]byte, nb)
	for i := range e {
		e[i] = 121
	}
	dst := make([]byte, rows*cols*2)
	b.SetBytes(int64(len(dst)))
	b.ResetTimer()
	for range b.N {
		DequantizeF16(dst, w, e)
	}
}
