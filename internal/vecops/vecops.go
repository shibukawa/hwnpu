// Package vecops holds the inner loops of weight conversion, written in assembly for amd64
// (AVX2) and arm64 (NEON) with portable Go fallbacks: assigning values to a sorted table
// (palettization), the clipping search and rounding of symmetric linear quantization, and BF16
// decoding.
//
// Every kernel returns exactly what its Go fallback returns, bit for bit, on every input without
// NaNs; the functions that could differ on NaNs report them instead, and their callers take the
// Go path. The Go fallbacks spell out their roundings with float32 conversions, so that no
// compiler fuses a multiply and an add: the results are the same on every architecture.
package vecops

import (
	"math"
)

// Thresholds is a sorted list of 2^L-1 thresholds (L from 1 to 8), prepared for Assign.
type Thresholds struct {
	mids []float32
	// tree holds mids in breadth first order (an Eytzinger layout): level l (from 0) is
	// tree[2^l-1 : 2^(l+1)-1], node j of a level has children 2j and 2j+1 of the next.
	tree   []float32
	levels int
	sorted bool // non-decreasing and without NaNs: the kernels may run
}

// NewThresholds prepares mids, which must have 2^L-1 entries for L from 1 to 8. mids is kept,
// not copied.
func NewThresholds(mids []float32) *Thresholds {
	t := &Thresholds{mids: mids}
	t.Reset(mids)
	return t
}

// Reset prepares t for new thresholds of the same count, reusing its memory.
func (t *Thresholds) Reset(mids []float32) {
	n := len(mids)
	levels := 0
	for 1<<levels-1 < n {
		levels++
	}
	if 1<<levels-1 != n || levels > 8 {
		panic("vecops: thresholds must number 2^L-1 for L from 1 to 8")
	}
	t.mids, t.levels = mids, levels
	t.sorted = true
	for i := 1; i < n; i++ {
		if !(mids[i-1] <= mids[i]) {
			t.sorted = false
		}
	}
	for _, m := range mids {
		if m != m {
			t.sorted = false
		}
	}
	if cap(t.tree) < n {
		t.tree = make([]float32, n)
	}
	t.tree = t.tree[:n]
	for l := range levels {
		h := 1 << (levels - l - 1) // half the span of a node of level l
		for j := range 1 << l {
			t.tree[1<<l-1+j] = mids[j*2*h+h-1]
		}
	}
}

// Assign writes to dst[i] the number of thresholds below vals[i] (strictly): for sorted
// thresholds that are the midpoints between sorted centers, the index of the nearest center,
// ties to the lower one. A NaN counts as below no threshold. dst must hold len(vals) entries.
func (t *Thresholds) Assign(dst []uint8, vals []float32) {
	dst = dst[:len(vals)]
	done := 0
	if t.sorted {
		done = assignKernel(t, dst, vals)
	}
	assignGo(t.mids, dst[done:], vals[done:])
}

// assignGo is the binary search over mids. The kernels count instead, which gives the same
// result only for sorted mids without NaNs; for others this search decides.
func assignGo(mids []float32, dst []uint8, vals []float32) {
	for i, v := range vals {
		lo, hi := 0, len(mids)
		for lo < hi {
			mid := (lo + hi) / 2
			if v > mids[mid] {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		dst[i] = uint8(lo)
	}
}

// MinMax returns the smallest and largest of vals, starting from math.MaxFloat32 and
// -math.MaxFloat32 (so an empty slice or one of infinities only gives those), and false when
// vals holds a NaN. The sign of a zero result is not specified.
func MinMax(vals []float32) (lo, hi float32, ok bool) {
	lo, hi = math.MaxFloat32, -math.MaxFloat32
	done := 0
	if haveKernels {
		var nan bool
		done, lo, hi, nan = minMaxKernel(vals)
		if nan {
			return lo, hi, false
		}
	}
	for _, v := range vals[done:] {
		if v != v {
			return lo, hi, false
		}
		lo, hi = min(lo, v), max(hi, v)
	}
	return lo, hi, true
}

// AbsMax returns the largest magnitude in vals (0 for none), and false when vals holds a NaN.
func AbsMax(vals []float32) (float32, bool) {
	var top float32
	done := 0
	if haveKernels {
		var nan bool
		done, top, nan = absMaxKernel(vals)
		if nan {
			return top, false
		}
	}
	for _, v := range vals[done:] {
		if v != v {
			return top, false
		}
		top = max(top, float32(math.Abs(float64(v))))
	}
	return top, true
}

// MaxScales is the number of scales ClipErrors takes at most.
const MaxScales = 16

// ClipErrors writes to errs[c] the squared error of quantizing vals with scales[c] to integers
// from qmin to qmax: the sum, in order, of (v - q s)^2 for q = v/s rounded to even and clamped (0
// when s is 0), each step rounded to float32. len(scales) is at most MaxScales, errs at least as
// long. vals must hold no NaN (AbsMax tells).
func ClipErrors(errs, vals, scales []float32, qmin, qmax float32) {
	if len(scales) > MaxScales {
		panic("vecops: too many scales")
	}
	errs = errs[:len(scales)]
	if haveKernels && len(scales) > 0 && len(scales) <= kernelScales {
		var s [MaxScales]float32
		var nz [MaxScales]uint32
		var e [MaxScales]float32
		for i := range s {
			s[i] = 1
			if i < len(scales) {
				s[i] = scales[i]
			}
			if s[i] != 0 {
				nz[i] = math.MaxUint32
			}
		}
		clipErrorsKernel(&e, vals, &s, &nz, qmin, qmax)
		copy(errs, e[:])
		return
	}
	for c, s := range scales {
		var e float32
		for _, v := range vals {
			d := v - float32(quantize(v, s, qmin, qmax)*s)
			e = float32(e + float32(d*d))
		}
		errs[c] = e
	}
}

// quantize is v/s rounded to even and clamped to [qmin, qmax], or 0 when s is 0.
func quantize(v, s, qmin, qmax float32) float32 {
	if s == 0 {
		return 0
	}
	q := float32(math.RoundToEven(float64(v / s)))
	return min(max(q, qmin), qmax)
}

// QuantizeSym writes to dst[i] vals[i]/s rounded to even and clamped to [qmin, qmax], which must
// lie within [-128, 127]; 0 when s is 0. vals must hold no NaN.
func QuantizeSym(dst []int8, vals []float32, s, qmin, qmax float32) {
	dst = dst[:len(vals)]
	if s == 0 {
		clear(dst)
		return
	}
	done := 0
	if haveKernels {
		done = quantizeSymKernel(dst, vals, s, qmin, qmax)
	}
	for i, v := range vals[done:] {
		dst[done+i] = int8(quantize(v, s, qmin, qmax))
	}
}

// BF16ToF32 decodes little endian BF16 values from src into dst; src must hold 2*len(dst) bytes.
func BF16ToF32(dst []float32, src []byte) {
	_ = src[:2*len(dst)]
	done := 0
	if haveKernels {
		done = bf16Kernel(dst, src)
	}
	for i := done; i < len(dst); i++ {
		dst[i] = math.Float32frombits(uint32(src[2*i])<<16 | uint32(src[2*i+1])<<24)
	}
}
