// Package mxfp4 multiplies by weights in OCP's MXFP4 on the CPU: blocks of 32 values of 4 bits
// (e2m1: ±0, 0.5, 1, 1.5, 2, 3, 4, 6), two a byte (the even value in the low nibble), with a
// scale of the block a power of two (e8m0: 2^(e-127)). gpt-oss's experts are in it as released
// (the checkpoint's *_blocks, (rows, blocks, 16) bytes, and *_scales, (rows, blocks)).
//
// A row's dot product takes the vector quantized to int8 by blocks of 32 (Quantize): the
// values of the nibbles, doubled to integers, times the int8 values summed in int32 (NEON's
// SDOT on arm64), each block scaled in float32.
package mxfp4

import "math"

// Block is the values of a block.
const Block = 32

// Values are e2m1's values by their 4 bits.
var Values = [16]float32{0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6}

// doubled are Values times 2, integers.
var doubled = [16]int8{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}

// Vec is a vector quantized for Dot: int8 values by blocks, each block's 16 even values then its
// 16 odd ones (the order of a block's low and high nibbles), and each block's scale.
type Vec struct {
	Q []int8
	S []float32
}

// NewVec makes a Vec of n values (a multiple of Block).
func NewVec(n int) *Vec { return &Vec{Q: make([]int8, n), S: make([]float32, n/Block)} }

// Quantize writes x into v, each block's values over its largest magnitude times 127.
func (v *Vec) Quantize(x []float32) {
	for b := range len(x) / Block {
		blk := x[b*Block : (b+1)*Block]
		var m float32
		for _, f := range blk {
			m = max(m, float32(math.Abs(float64(f))))
		}
		q := v.Q[b*Block : (b+1)*Block]
		if m == 0 {
			clear(q)
			v.S[b] = 0
			continue
		}
		inv := 127 / m
		for j := range Block / 2 {
			q[j] = int8(math.RoundToEven(float64(blk[2*j] * inv)))
			q[Block/2+j] = int8(math.RoundToEven(float64(blk[2*j+1] * inv)))
		}
		v.S[b] = m / 127
	}
}

// pairVec is two Vecs a and b interleaved for SMMLA (mm24Kernel): for each block its 4 segments
// of 8 values (the evens 0-7, 8-15, the odds 0-7, 8-15), a's then b's, and the scales a, b, a, b.
type pairVec struct {
	Q []int8
	S []float32
}

func interleave(a, b *Vec) pairVec {
	nb := len(a.S)
	p := pairVec{Q: make([]int8, 2*len(a.Q)), S: make([]float32, 4*nb)}
	for blk := range nb {
		aq, bq, o := a.Q[blk*Block:], b.Q[blk*Block:], p.Q[2*blk*Block:]
		for j := range 4 {
			copy(o[16*j:16*j+8], aq[8*j:8*j+8])
			copy(o[16*j+8:16*j+16], bq[8*j:8*j+8])
		}
		p.S[4*blk], p.S[4*blk+1], p.S[4*blk+2], p.S[4*blk+3] = a.S[blk], b.S[blk], a.S[blk], b.S[blk]
	}
	return p
}

// scale is 2^(e-127), and 2^-127 for e of 0 (a value under float32's normal ones, as zero).
func scale(e byte) float32 { return math.Float32frombits(uint32(e) << 23) }

// dotGo is Dot without the kernel.
func dotGo(w, e []byte, v *Vec) float32 {
	var sum float32
	for b := range len(e) {
		var acc int32
		blk, q := w[16*b:16*(b+1)], v.Q[b*Block:(b+1)*Block]
		for j, by := range blk {
			acc += int32(doubled[by&15])*int32(q[j]) + int32(doubled[by>>4])*int32(q[16+j])
		}
		sum += float32(acc) * scale(e[b]) * v.S[b]
	}
	return sum / 2
}

// Dot is a row's dot product with v: the row's blocks w (16 bytes a block) and their scales e.
func Dot(w, e []byte, v *Vec) float32 {
	if len(e) == 0 {
		return 0
	}
	if haveKernel {
		return dotKernel(w, e, v)
	}
	return dotGo(w, e, v)
}

// Rows writes into out each row's dot product with v, on the caller's core: the rows' blocks w
// and their scales e, one row after another. The values are Dot's.
func Rows(out []float32, w, e []byte, v *Vec) {
	if len(out) == 0 {
		return
	}
	if haveKernel && rowsKernel(out, w, e, v) {
		return
	}
	nb := len(e) / len(out)
	for r := range out {
		out[r] = Dot(w[r*nb*16:(r+1)*nb*16], e[r*nb:(r+1)*nb], v)
	}
}

// Dequantize writes a row's values.
func Dequantize(dst []float32, w, e []byte) {
	for b := range len(e) {
		s := scale(e[b])
		for j, by := range w[16*b : 16*(b+1)] {
			dst[b*Block+2*j] = Values[by&15] * s
			dst[b*Block+2*j+1] = Values[by>>4] * s
		}
	}
}

// f16Hi holds the high bytes of IEEE-754 binary16 for Values at exponent bias 127 (scale = 1.0),
// with -0 canonicalized to +0 (0x0000). Because every e2m1 value has at most 1 mantissa bit (bit 9
// of binary16), the low byte of every normal scaled binary16 value (114 <= e <= 140) is 0x00.
var f16Hi = [16]byte{0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46, 0x00, 0xB8, 0xBC, 0xBE, 0xC0, 0xC2, 0xC4, 0xC6}

// DequantizeF16 writes a row's MXFP4 blocks w (16 bytes/block) and scales e (1 byte/block) as
// little-endian IEEE-754 fp16 (64 bytes/block) into dst.
func DequantizeF16(dst, w, e []byte) {
	nb := len(e)
	if nb == 0 {
		return
	}
	b := 0
	for b < nb {
		if haveKernel {
			n := dequantF16Kernel(dst[b*64:], w[b*16:], e[b:])
			b += n
			if b >= nb {
				return
			}
		}
		dequantF16BlockGo(dst[b*64:(b+1)*64], w[b*16:(b+1)*16], e[b])
		b++
	}
}

func dequantF16BlockGo(dst, blk []byte, eb byte) {
	if eb >= 114 && eb <= 140 {
		delta := byte(int8(int(eb)-127) << 2)
		for j, by := range blk[:16] {
			h0, h1 := f16Hi[by&15], f16Hi[by>>4]
			if h0 != 0 {
				h0 += delta
			}
			if h1 != 0 {
				h1 += delta
			}
			dst[4*j] = 0
			dst[4*j+1] = h0
			dst[4*j+2] = 0
			dst[4*j+3] = h1
		}
		return
	}
	s := scale(eb)
	for j, by := range blk[:16] {
		u0 := f32ToF16Bits(Values[by&15] * s)
		u1 := f32ToF16Bits(Values[by>>4] * s)
		dst[4*j] = byte(u0)
		dst[4*j+1] = byte(u0 >> 8)
		dst[4*j+2] = byte(u1)
		dst[4*j+3] = byte(u1 >> 8)
	}
}

func f32ToF16Bits(f float32) uint16 {
	if f == 0 {
		return 0
	}
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xFF) - 127 + 15
	mant := bits & 0x7FFFFF
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		half := mant >> shift
		if (mant>>(shift-1))&1 != 0 && ((mant&((1<<(shift-1))-1)) != 0 || (half&1) != 0) {
			half++
		}
		return sign | uint16(half)
	}
	if exp >= 31 {
		return sign | 0x7C00
	}
	half := uint16(exp<<10) | uint16(mant>>13)
	if (mant&0x1000) != 0 && ((mant&0x0FFF) != 0 || (half&1) != 0) {
		half++
	}
	return sign | half
}
