//go:build !amd64

package f32

func expKernel(dst, src []float32, mode int) int           { return 0 }
func axpyKernel(dst []float32, a float32, x []float32) int { return 0 }
