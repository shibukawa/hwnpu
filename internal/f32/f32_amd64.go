package f32

import "golang.org/x/sys/cpu"

var haveAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasFMA

//go:noescape
func expAVX2(dst, src *float32, n, mode int)

//go:noescape
func axpyAVX2(dst *float32, a float32, x *float32, n int)

// expKernel runs Exp (mode 0), Sigmoid (1) or SiLU (2) over the first multiple of 8 of src and
// returns how many it did.
func expKernel(dst, src []float32, mode int) int {
	n := len(src) &^ 7
	if !haveAVX2 || n == 0 {
		return 0
	}
	expAVX2(&dst[0], &src[0], n, mode)
	return n
}

func axpyKernel(dst []float32, a float32, x []float32) int {
	n := len(dst) &^ 7
	if !haveAVX2 || n == 0 {
		return 0
	}
	axpyAVX2(&dst[0], a, &x[0], n)
	return n
}
