package f32

// Every arm64 processor has NEON: the loops below run on all of them (conv_arm64.go has the
// convolutions' and Axpy).

//go:noescape
func expNEON(dst, src *float32, n, mode int)

// expKernel runs Exp (mode 0), Sigmoid (1) or SiLU (2) over the first multiple of 8 of src and
// returns how many it did: the same steps as the AVX2 loop (the same values).
func expKernel(dst, src []float32, mode int) int {
	n := len(src) &^ 7
	if n == 0 {
		return 0
	}
	expNEON(&dst[0], &src[0], n, mode)
	return n
}

//go:noescape
func gluNEON(dst, gu *float32, n int, alpha, limit, add float32)

// gluKernel runs GLU over the first multiple of 8 of dst (LD2 splits the gates and the ups).
func gluKernel(dst, gu []float32, alpha, limit, add float32) int {
	n := len(dst) &^ 7
	if n == 0 {
		return 0
	}
	gluNEON(&dst[0], &gu[0], n, alpha, limit, add)
	return n
}
