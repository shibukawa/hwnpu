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
			m = max(m, float32(math.Abs(float64(v))))
		}
		best, bestErr := byte(0), math.Inf(1)
		if m > 0 {
			_, exp := math.Frexp(float64(m)) // m = f 2^exp, f in [0.5, 1): floor(log2 m) = exp-1
			rule := exp - 1 - 2
			for _, x := range []int{rule - 1, rule, rule + 1} {
				eb := x + 127
				if eb < 1 || eb > 254 {
					continue
				}
				s := float64(scale(byte(eb)))
				var se float64
				for _, v := range blk {
					d := float64(v) - float64(Values[nearest(float64(v)/s)])*s
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
		s := float64(scale(best))
		for j, v := range blk {
			codes[j] = nearest(float64(v) / s)
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
	c := byte(7)
	for k := range byte(7) {
		mid := float64(Values[k]+Values[k+1]) / 2
		if x < mid || x == mid && k%2 == 0 {
			c = k
			break
		}
		if x == mid {
			c = k + 1
			break
		}
	}
	if c == 0 {
		return 0
	}
	return sign | c
}
