package f32

//go:noescape
func dw3AVX2(dst, r0, r1, r2, w *float32, bias float32, relu, n int)

//go:noescape
func maxpool2AVX2(dst, a, b *float32, n int)

//go:noescape
func reluAVX2(dst, src *float32, n int)

func dw3Kernel(dst, r0, r1, r2 []float32, w *[9]float32, bias float32, relu bool) int {
	n := len(dst) &^ 7
	if !haveAVX2 || n == 0 {
		return 0
	}
	r := 0
	if relu {
		r = 1
	}
	dw3AVX2(&dst[0], &r0[0], &r1[0], &r2[0], &w[0], bias, r, n)
	return n
}

func maxpool2Kernel(dst, a, b []float32) int {
	n := len(dst) &^ 7
	if !haveAVX2 || n == 0 {
		return 0
	}
	maxpool2AVX2(&dst[0], &a[0], &b[0], n)
	return n
}

func reluKernel(dst, src []float32) int {
	n := len(dst) &^ 31
	if !haveAVX2 || n == 0 {
		return 0
	}
	reluAVX2(&dst[0], &src[0], n)
	return n
}

//go:noescape
func wrowsAVX2(dst *float32, rows **float32, w *float32, k int, bias float32, relu, n int)

func wrowsKernel(dst []float32, rows []*float32, w []float32, bias float32, relu bool) int {
	n := len(dst) &^ 15
	if !haveAVX2 || n == 0 || len(rows) == 0 {
		return 0
	}
	r := 0
	if relu {
		r = 1
	}
	wrowsAVX2(&dst[0], &rows[0], &w[0], len(rows), bias, r, n)
	return n
}

//go:noescape
func wrows4AVX2(dst **float32, rows **float32, w *float32, k int, bias *float32, relu, n int)

func wrows4Kernel(dst *[4]*float32, n int, rows []*float32, w []float32, bias *[4]float32, relu bool) int {
	n &^= 15
	if !haveAVX2 || n == 0 || len(rows) == 0 {
		return 0
	}
	r := 0
	if relu {
		r = 1
	}
	wrows4AVX2(&dst[0], &rows[0], &w[0], len(rows), &bias[0], r, n)
	return n
}

//go:noescape
func deint2AVX2(even, odd, src *float32, n int)

func deint2Kernel(even, odd, src []float32) int {
	n := len(even) &^ 7
	if !haveAVX2 || n == 0 {
		return 0
	}
	deint2AVX2(&even[0], &odd[0], &src[0], n)
	return n
}
