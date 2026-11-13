package sgemm

// The kernel: NEON (6 rows of A, four vectors of B), on every arm64.
var mr = 6

//go:noescape
func kernelNEON(kc int, a, b, c *float32, ldc int)

func kernel(kc int, a, b, c []float32, ldc int) {
	_ = a[kc*mr-1]
	_ = b[kc*NR-1]
	_ = c[(mr-1)*ldc+NR-1]
	kernelNEON(kc, &a[0], &b[0], &c[0], ldc)
}

func haveKernel() bool { return true }
