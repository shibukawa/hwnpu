//go:build !amd64

package sgemm

var mr = 6

func kernel(kc int, a, b, c []float32, ldc int) { kernelGo(kc, a, b, c, ldc) }

func haveKernel() bool { return false }
