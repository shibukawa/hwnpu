package milcpu

import (
	"fmt"
	"math"
)

// q4nxWeight is a weight in the q4nx format: w = scale * code + min, one bf16 scale and minimum
// per row and 32 columns.
type q4nxWeight struct {
	rows, cols   int
	codes        []int32
	scales, mins []float32 // bf16 values, (row, column / 32)
}

// asQ4NX recognizes the q4nx form of constexpr_blockwise_shift_scale (palette.Q4NX.ShiftScale):
// a rows x cols matrix of codes 0..15 with a float offset per 32 columns. The minimum is
// -offset * scale rounded to bf16, which recovers the bf16 minimum the fp16 offset came from.
func asQ4NX(data, scale, offset *value) *q4nxWeight {
	if offset == nil || offset.f == nil || len(data.shape) != 2 || len(scale.shape) != 2 {
		return nil
	}
	rows, cols := data.shape[0], data.shape[1]
	if cols%32 != 0 || scale.shape[0] != rows || scale.shape[1] != cols/32 {
		return nil
	}
	for _, c := range data.i {
		if c < 0 || c > 15 {
			return nil
		}
	}
	q := &q4nxWeight{rows: rows, cols: cols, codes: data.i, scales: make([]float32, len(scale.f)), mins: make([]float32, len(scale.f))}
	for i, s := range scale.f {
		q.scales[i] = bf16Round(s, false)
		q.mins[i] = bf16Round(-offset.f[i]*s, false)
	}
	return q
}

// bf16Round rounds f to bfloat16, to the nearest (ties to even) or toward minus infinity.
func bf16Round(f float32, floor bool) float32 {
	b := math.Float32bits(f)
	if b&0x7f800000 == 0x7f800000 {
		return f
	}
	if floor {
		if b>>31 != 0 && b&0xffff != 0 {
			b += 0x10000
		}
	} else {
		b += 0x7fff + (b>>16)&1
	}
	return math.Float32frombits(b &^ 0xffff)
}

// linearQ4NX is linear with the arithmetic of the AMD NPU's q4nx kernel (q4nx_accumulate in
// mlir-aie's aie_kernels/flm_gemma4/q4nx.h). Per output and group of 32 columns g:
//
//	acc += bf16(sum_c code[c] * xb[c]) * scale[g] + min[g] * bf16(sum_c xb[c])
//
// where xb is x rounded to bf16, both sums run in fp32 in column order and acc is fp32.
func linearQ4NX(x, w, bias *value, floor bool) (*value, error) {
	q := w.q4
	if len(x.shape) == 0 || x.shape[len(x.shape)-1] != q.cols {
		return nil, fmt.Errorf("x %v and weight %v do not match", x.shape, w.shape)
	}
	n, k := q.rows, q.cols
	m := len(x.f) / k
	ng := k / 32
	xb := make([]float32, len(x.f))
	for i, v := range x.f {
		xb[i] = bf16Round(v, floor)
	}
	sums := make([]float32, m*ng)
	for i := range sums {
		var s float32
		for _, v := range xb[i*32 : (i+1)*32] {
			s += v
		}
		sums[i] = bf16Round(s, floor)
	}
	shape := append(append([]int{}, x.shape[:len(x.shape)-1]...), n)
	out := &value{shape: shape, f: make([]float32, m*n)}
	parallel(n, func(lo, hi int) {
		for j := lo; j < hi; j++ {
			codes := q.codes[j*k : (j+1)*k]
			for i := 0; i < m; i++ {
				xr := xb[i*k : (i+1)*k]
				var acc float32
				for g := 0; g < ng; g++ {
					var t float32
					for c := g * 32; c < (g+1)*32; c++ {
						t += float32(codes[c]) * xr[c]
					}
					acc += bf16Round(t, floor) * q.scales[j*ng+g]
					acc += q.mins[j*ng+g] * sums[i*ng+g]
				}
				if bias != nil {
					acc += bias.f[j]
				}
				out.f[i*n+j] = acc
			}
		}
	})
	return out, nil
}
