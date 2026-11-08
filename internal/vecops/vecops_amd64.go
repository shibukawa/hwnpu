package vecops

// haveKernels is AVX2 with the operating system saving the YMM registers.
var haveKernels = hasAVX2()

func hasAVX2() bool

//go:noescape
func assignAVX2(dst *uint8, vals *float32, n int, tree *float32, levels int)

//go:noescape
func minMaxAVX2(vals *float32, n int) (lo, hi float32, nan bool)

//go:noescape
func absMaxAVX2(vals *float32, n int) (top float32, nan bool)

//go:noescape
func clipErrorsAVX2(errs *[MaxScales]float32, vals *float32, n int, scales *[MaxScales]float32, nz *[MaxScales]uint32, qmin, qmax float32)

//go:noescape
func quantizeSymAVX2(dst *int8, vals *float32, n int, s, qmin, qmax float32)

//go:noescape
func bf16AVX2(dst *float32, src *byte, n int)

// assignKernel runs the binary search on 32 values at a time: the first four levels look their
// thresholds up in a register (VPERMPS), the others gather them.
func assignKernel(t *Thresholds, dst []uint8, vals []float32) int {
	n := len(vals) &^ 31
	if !haveKernels || t.levels < 4 || n == 0 {
		return 0
	}
	assignAVX2(&dst[0], &vals[0], n, &t.tree[0], t.levels)
	return n
}

func minMaxKernel(vals []float32) (int, float32, float32, bool) {
	n := len(vals) &^ 31
	if n == 0 {
		return 0, 3.4028234663852886e+38, -3.4028234663852886e+38, false
	}
	lo, hi, nan := minMaxAVX2(&vals[0], n)
	return n, lo, hi, nan
}

func absMaxKernel(vals []float32) (int, float32, bool) {
	n := len(vals) &^ 31
	if n == 0 {
		return 0, 0, false
	}
	top, nan := absMaxAVX2(&vals[0], n)
	return n, top, nan
}

func clipErrorsKernel(e *[MaxScales]float32, vals []float32, s *[MaxScales]float32, nz *[MaxScales]uint32, qmin, qmax float32) {
	if len(vals) == 0 {
		return
	}
	clipErrorsAVX2(e, &vals[0], len(vals), s, nz, qmin, qmax)
}

func quantizeSymKernel(dst []int8, vals []float32, s, qmin, qmax float32) int {
	n := len(vals) &^ 31
	if n == 0 {
		return 0
	}
	quantizeSymAVX2(&dst[0], &vals[0], n, s, qmin, qmax)
	return n
}

func bf16Kernel(dst []float32, src []byte) int {
	n := len(dst) &^ 31
	if n == 0 {
		return 0
	}
	bf16AVX2(&dst[0], &src[0], n)
	return n
}

const kernelScales = MaxScales
