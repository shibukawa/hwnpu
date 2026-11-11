package sgemm

import "golang.org/x/sys/cpu"

// The kernels: AVX-512 (12 rows of A, a ZMM of B), else AVX2 and FMA (6 rows, two YMM), else Go.
var (
	haveAVX512 = cpu.X86.HasAVX512F
	haveAVX2   = cpu.X86.HasAVX2 && cpu.X86.HasFMA
	mr         = map[bool]int{true: 12, false: 6}[haveAVX512]
)

//go:noescape
func kernelAVX2(kc int, a, b, c *float32, ldc int)

//go:noescape
func kernelAVX512(kc int, a, b, c *float32, ldc int)

func kernel(kc int, a, b, c []float32, ldc int) {
	_ = a[kc*mr-1]
	_ = b[kc*NR-1]
	_ = c[(mr-1)*ldc+NR-1]
	switch {
	case mr == 12 && haveAVX512:
		kernelAVX512(kc, &a[0], &b[0], &c[0], ldc)
	case mr == 6 && haveAVX2:
		kernelAVX2(kc, &a[0], &b[0], &c[0], ldc)
	default:
		kernelGo(kc, a, b, c, ldc)
	}
}

func haveKernel() bool { return haveAVX2 || haveAVX512 }
