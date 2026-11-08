package vecops

// haveKernels: every arm64 processor has NEON.
var haveKernels = true

//go:noescape
func assign15NEON(dst *uint8, vals *float32, n int, mids *float32)

//go:noescape
func assignTreeARM64(dst *uint8, vals *float32, n int, tree *float32, levels int)

//go:noescape
func minMaxNEON(vals *float32, n int) (lo, hi float32)

//go:noescape
func absMaxNEON(vals *float32, n int) float32

//go:noescape
func clipErrorsNEON(errs *[MaxScales]float32, vals *float32, n int, scales *[MaxScales]float32, nz *[MaxScales]uint32, qmin, qmax float32)

//go:noescape
func quantizeSymNEON(dst *int8, vals *float32, n int, s, qmin, qmax float32)

//go:noescape
func bf16NEON(dst *float32, src *byte, n int)

// assignKernel counts with NEON compares for 15 thresholds (16 values at a time), and runs the
// binary search without branches on 8 values at a time for more.
func assignKernel(t *Thresholds, dst []uint8, vals []float32) int {
	if !haveKernels {
		return 0
	}
	if t.levels == 4 {
		n := len(vals) &^ 15
		if n > 0 {
			assign15NEON(&dst[0], &vals[0], n, &t.mids[0])
		}
		return n
	}
	n := len(vals) &^ 7
	if n > 0 {
		assignTreeARM64(&dst[0], &vals[0], n, &t.tree[0], t.levels)
	}
	return n
}

// The NEON minimum and maximum give NaN for a NaN: that is how they are told.
func minMaxKernel(vals []float32) (int, float32, float32, bool) {
	n := len(vals) &^ 15
	if n == 0 {
		return 0, 3.4028234663852886e+38, -3.4028234663852886e+38, false
	}
	lo, hi := minMaxNEON(&vals[0], n)
	return n, lo, hi, lo != lo || hi != hi
}

func absMaxKernel(vals []float32) (int, float32, bool) {
	n := len(vals) &^ 15
	if n == 0 {
		return 0, 0, false
	}
	top := absMaxNEON(&vals[0], n)
	return n, top, top != top
}

// clipErrorsKernel takes 12 scales in three registers; ClipErrors calls it only for that many.
func clipErrorsKernel(e *[MaxScales]float32, vals []float32, s *[MaxScales]float32, nz *[MaxScales]uint32, qmin, qmax float32) {
	if len(vals) == 0 {
		return
	}
	clipErrorsNEON(e, &vals[0], len(vals), s, nz, qmin, qmax)
}

const kernelScales = 12

func quantizeSymKernel(dst []int8, vals []float32, s, qmin, qmax float32) int {
	n := len(vals) &^ 15
	if n == 0 {
		return 0
	}
	quantizeSymNEON(&dst[0], &vals[0], n, s, qmin, qmax)
	return n
}

func bf16Kernel(dst []float32, src []byte) int {
	n := len(dst) &^ 15
	if n == 0 {
		return 0
	}
	bf16NEON(&dst[0], &src[0], n)
	return n
}

//go:noescape
func pack4NEON(dst *byte, idx *uint8, n int)

//go:noescape
func pack6NEON(dst *byte, idx *uint8, n int)

// packKernel packs 4 bit entries 32 at a time (LD2) and 6 bit ones 64 at a time (LD4, ST3); it
// returns how many entries it packed, a multiple of 8.
func packKernel(dst []byte, idx []uint8, bits int) int {
	switch bits {
	case 4:
		n := len(idx) &^ 31
		if n > 0 {
			pack4NEON(&dst[0], &idx[0], n)
		}
		return n
	case 6:
		n := len(idx) &^ 63
		if n > 0 {
			pack6NEON(&dst[0], &idx[0], n)
		}
		return n
	}
	return 0
}

// haveF16: every arm64 processor converts between half and single precision (FCVTN, FCVTL).
var haveF16 = true

//go:noescape
func f32ToF16NEON(dst *byte, src *float32, n int) (nan bool)

//go:noescape
func f16ToF32NEON(dst *float32, src *byte, n int) (nan bool)

// The conversions run 16 values at a time and report NaNs, which the caller converts in Go.
func f32ToF16Kernel(dst []byte, src []float32) int {
	n := len(src) &^ 15
	if n == 0 || f32ToF16NEON(&dst[0], &src[0], n) {
		return 0
	}
	return n
}

func f16ToF32Kernel(dst []float32, src []byte) int {
	n := len(dst) &^ 15
	if n == 0 || f16ToF32NEON(&dst[0], &src[0], n) {
		return 0
	}
	return n
}
