// Package f32 holds float32 vector loops of hwnpu's CPU executor (internal/milcpu): exp, sigmoid
// and SiLU over arrays, and axpy, in AVX2 and FMA on amd64 and in Go elsewhere.
package f32

import "math"

// Exp writes e^x of src into dst (Cephes' expf: x = n ln 2 + r, a polynomial of degree 6 in r,
// times 2^n; within 2 ulp). x is clamped to [-86.6, 88.72283]: below, about 2.3e-38; above,
// +Inf. NaN stays NaN.
func Exp(dst, src []float32) {
	dst = dst[:len(src)]
	n := expKernel(dst, src, 0)
	for i := n; i < len(src); i++ {
		dst[i] = exp(src[i])
	}
}

// Sigmoid writes 1 / (1 + e^-x) of src into dst.
func Sigmoid(dst, src []float32) {
	dst = dst[:len(src)]
	n := expKernel(dst, src, 1)
	for i := n; i < len(src); i++ {
		dst[i] = 1 / (1 + exp(-src[i]))
	}
}

// SiLU writes x / (1 + e^-x) of src into dst.
func SiLU(dst, src []float32) {
	dst = dst[:len(src)]
	n := expKernel(dst, src, 2)
	for i := n; i < len(src); i++ {
		dst[i] = src[i] / (1 + exp(-src[i]))
	}
}

// Axpy adds a x to dst.
func Axpy(dst []float32, a float32, x []float32) {
	x = x[:len(dst)]
	n := axpyKernel(dst, a, x)
	for i := n; i < len(dst); i++ {
		dst[i] += a * x[i]
	}
}

// exp is one value of Exp.
func exp(x float32) float32 {
	if x != x {
		return x
	}
	x = min(max(x, -86.6), 88.72283)
	const (
		log2e = 1.44269504088896341
		c1    = 0.693359375
		c2    = -2.12194440e-4
	)
	n := float32(math.Floor(float64(x*log2e + 0.5)))
	r := x - n*c1 - n*c2
	p := float32(1.9875691500e-4)
	p = p*r + 1.3981999507e-3
	p = p*r + 8.3334519073e-3
	p = p*r + 4.1665795894e-2
	p = p*r + 1.6666665459e-1
	p = p*r + 5.0000001201e-1
	y := p*r*r + r + 1
	return (y + y) * math.Float32frombits(uint32(int32(n)+126)<<23)
}

// Erf writes erf(x) of src into dst: Abramowitz and Stegun 7.1.26 (1 - t (a1 + t (a2 + ...))
// e^(-x^2), t = 1 / (1 + p |x|)): within 1.5e-7 of erf, 5e-7 in float32 (absolute), with Exp's
// vector loop.
func Erf(dst, src []float32) {
	dst = dst[:len(src)]
	for i, x := range src {
		dst[i] = -x * x
	}
	Exp(dst, dst)
	const (
		p  = 0.3275911
		a1 = 0.254829592
		a2 = -0.284496736
		a3 = 1.421413741
		a4 = -1.453152027
		a5 = 1.061405429
	)
	for i, x := range src {
		ax := x
		if ax < 0 {
			ax = -ax
		}
		t := 1 / (1 + p*ax)
		y := 1 - t*(a1+t*(a2+t*(a3+t*(a4+t*a5))))*dst[i]
		if x < 0 {
			y = -y
		}
		if x != x {
			y = x
		}
		dst[i] = y
	}
}
