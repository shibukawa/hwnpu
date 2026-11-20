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
