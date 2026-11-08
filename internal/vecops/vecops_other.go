//go:build !amd64 && !arm64

package vecops

var haveKernels = false

func assignKernel(t *Thresholds, dst []uint8, vals []float32) int { return 0 }

func minMaxKernel(vals []float32) (int, float32, float32, bool) { return 0, 0, 0, false }

func absMaxKernel(vals []float32) (int, float32, bool) { return 0, 0, false }

func clipErrorsKernel(e *[MaxScales]float32, vals []float32, s *[MaxScales]float32, nz *[MaxScales]uint32, qmin, qmax float32) {
}

func quantizeSymKernel(dst []int8, vals []float32, s, qmin, qmax float32) int { return 0 }

func bf16Kernel(dst []float32, src []byte) int { return 0 }

const kernelScales = 0
