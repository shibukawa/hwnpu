package f32

// The kernels of conv.go and Axpy on NEON (every arm64 core has it). Exp, Sigmoid and SiLU
// stay Go's loops on arm64.

//go:noescape
func dw3NEON(dst, r0, r1, r2, w *float32, bias float32, relu, n int)

//go:noescape
func maxpool2NEON(dst, a, b *float32, n int)

//go:noescape
func reluNEON(dst, src *float32, n int)

//go:noescape
func axpyNEON(dst *float32, a float32, x *float32, n int)

//go:noescape
func wrowsNEON(dst *float32, rows **float32, w *float32, k int, bias float32, relu, n int)

//go:noescape
func wrows4NEON(dst **float32, rows **float32, w *float32, k int, bias *float32, relu, n, ws int)

//go:noescape
func wrows8NEON(dst **float32, rows **float32, w *float32, k int, bias *float32, relu, n int)

//go:noescape
func deint2NEON(even, odd, src *float32, n int)

func flag(b bool) int {
	if b {
		return 1
	}
	return 0
}

func expKernel(dst, src []float32, mode int) int { return 0 }

func axpyKernel(dst []float32, a float32, x []float32) int {
	n := len(dst) &^ 15
	if n == 0 {
		return 0
	}
	axpyNEON(&dst[0], a, &x[0], n)
	return n
}

func dw3Kernel(dst, r0, r1, r2 []float32, w *[9]float32, bias float32, relu bool) int {
	n := len(dst) &^ 7
	if n == 0 {
		return 0
	}
	dw3NEON(&dst[0], &r0[0], &r1[0], &r2[0], &w[0], bias, flag(relu), n)
	return n
}

func maxpool2Kernel(dst, a, b []float32) int {
	n := len(dst) &^ 7
	if n == 0 {
		return 0
	}
	maxpool2NEON(&dst[0], &a[0], &b[0], n)
	return n
}

func reluKernel(dst, src []float32) int {
	n := len(dst) &^ 15
	if n == 0 {
		return 0
	}
	reluNEON(&dst[0], &src[0], n)
	return n
}

func wrowsKernel(dst []float32, rows []*float32, w []float32, bias float32, relu bool) int {
	n := len(dst) &^ 15
	if n == 0 || len(rows) == 0 {
		return 0
	}
	wrowsNEON(&dst[0], &rows[0], &w[0], len(rows), bias, flag(relu), n)
	return n
}

func wrows4Kernel(dst *[4]*float32, n int, rows []*float32, w []float32, ws int, bias *[4]float32, relu bool) int {
	n &^= 15
	if n == 0 || len(rows) == 0 {
		return 0
	}
	wrows4NEON(&dst[0], &rows[0], &w[0], len(rows), &bias[0], flag(relu), n, 4*ws)
	return n
}

const wrows8Step = 8

func wrows8Kernel(dst *[8]*float32, n int, rows []*float32, w []float32, bias *[8]float32, relu bool) int {
	n &^= 7
	if n == 0 || len(rows) == 0 {
		return 0
	}
	wrows8NEON(&dst[0], &rows[0], &w[0], len(rows), &bias[0], flag(relu), n)
	return n
}

func deint2Kernel(even, odd, src []float32) int {
	n := len(even) &^ 7
	if n == 0 {
		return 0
	}
	deint2NEON(&even[0], &odd[0], &src[0], n)
	return n
}
