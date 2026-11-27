package mxfp4

import "math"

// Encode writes a row of values (a multiple of Block) in MXFP4: its blocks w (16 bytes a
// block, the even value in the low nibble) and their scales e, as a checkpoint holds them. A
// block's scale is the power of two of OCP's rule (its largest magnitude's exponent less e2m1's
// largest, 2), or the one above or below it when that rounds the block with a smaller squared
// error: the rule's scale clips magnitudes from 6 to 8 times it to 6, the next one up keeps them
// at the cost of the small values' steps. Each value rounds to the nearest e2m1 value.
func Encode(w, e []byte, row []float32) {
	var codes [Block]byte
	for b := range len(row) / Block {
		blk := row[b*Block : (b+1)*Block]
		var m float32
		for _, v := range blk {
			m = max(m, math.Float32frombits(math.Float32bits(v)&0x7fffffff))
		}
		best, bestErr := byte(0), math.Inf(1)
		if m > 0 {
			// IEEE-754 exponent extraction for floor(log2(m)) without Frexp on normal floats.
			bits := math.Float32bits(m)
			var floorLog2 int
			if exp := int((bits >> 23) & 0xff); exp > 0 && exp < 0xff {
				floorLog2 = exp - 127
			} else {
				_, e2 := math.Frexp(float64(m))
				floorLog2 = e2 - 1
			}
			rule := floorLog2 - 2
			for _, x := range [3]int{rule - 1, rule, rule + 1} {
				eb := x + 127
				if eb < 1 || eb > 254 {
					continue
				}
				s := float64(scale(byte(eb)))
				invS := 1 / s // exact power of two in float64: v * invS == v / s bit-for-bit
				var se float64
				for _, v := range blk {
					fv := float64(v)
					d := fv - float64(Values[nearest(fv*invS)])*s
					se += d * d
				}
				if se < bestErr {
					best, bestErr = byte(eb), se
				}
			}
		}
		if best == 0 { // a block of zeros, or out of e8m0's range: zeros (codes 0) at the least scale
			clear(w[16*b : 16*(b+1)])
			e[b] = 0
			continue
		}
		invS := 1 / float64(scale(best))
		for j, v := range blk {
			codes[j] = nearest(float64(v) * invS)
		}
		for j := range Block / 2 {
			w[16*b+j] = codes[2*j] | codes[2*j+1]<<4
		}
		e[b] = best
	}
}

// nearest is the 4 bits of the e2m1 value nearest x, clipped to ±6; halfway between two values
// it takes the one of the even code (round to nearest even), and 0 is always +0.
func nearest(x float64) byte {
	var sign byte
	if x < 0 {
		sign, x = 8, -x
	}
	var c byte
	if x <= 1.75 {
		if x < 0.75 {
			if x <= 0.25 {
				return 0
			}
			c = 1
		} else if x <= 1.25 {
			c = 2
		} else if x < 1.75 {
			c = 3
		} else {
			c = 4
		}
	} else if x <= 3.5 {
		if x <= 2.5 {
			c = 4
		} else if x < 3.5 {
			c = 5
		} else {
			c = 6
		}
	} else if x <= 5.0 {
		c = 6
	} else {
		c = 7
	}
	return sign | c
}
