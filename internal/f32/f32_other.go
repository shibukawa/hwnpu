//go:build !amd64

package f32

func expKernel(dst, src []float32, mode int) int           { return 0 }
func axpyKernel(dst []float32, a float32, x []float32) int { return 0 }

func dw3Kernel(dst, r0, r1, r2 []float32, w *[9]float32, bias float32, relu bool) int { return 0 }
func maxpool2Kernel(dst, a, b []float32) int                                          { return 0 }
func reluKernel(dst, src []float32) int                                               { return 0 }

func wrowsKernel(dst []float32, rows []*float32, w []float32, bias float32, relu bool) int {
	return 0
}

func wrows4Kernel(dst *[4]*float32, n int, rows []*float32, w []float32, bias *[4]float32, relu bool) int {
	return 0
}

func deint2Kernel(even, odd, src []float32) int { return 0 }
