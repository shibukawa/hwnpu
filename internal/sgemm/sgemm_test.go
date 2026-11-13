package sgemm

import (
	"math"
	"math/rand/v2"
	"testing"
)

func naive(m, n, k int, a A, b func(r, c int) float32, c []float32) {
	for i := range m {
		for j := range n {
			var s float64
			for kk := range k {
				s += float64(a.at(i, kk)) * float64(b(kk, j))
			}
			c[i*n+j] += float32(s)
		}
	}
}

func randf(r *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	return x
}

// TestMulAdd checks the product against a float64 one, at sizes off the blocks (edges of MR,
// NR, KC, MC), with A and B transposed, on 1 and 4 goroutines, adding to C.
func TestMulAdd(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, s := range [][3]int{{1, 1, 1}, {6, 16, 8}, {7, 17, 3}, {13, 50, 300}, {100, 33, 513}, {5, 4000, 70}, {200, 9, 64}} {
		m, n, k := s[0], s[1], s[2]
		for _, trans := range []bool{false, true} {
			for _, threads := range []int{1, 4} {
				ad := randf(r, m*k)
				bd := randf(r, k*n)
				a := A{Data: ad, Stride: k}
				if trans {
					a = A{Data: ad, Stride: m, Trans: true}
				}
				c0 := randf(r, m*n)
				want := append([]float32(nil), c0...)
				var src Source = Matrix{Data: bd, Stride: n, N: n}
				bat := func(rr, cc int) float32 { return bd[rr*n+cc] }
				if trans {
					src = Transposed{Data: bd, Stride: k, N: n} // bd as n x k
					bat = func(rr, cc int) float32 { return bd[cc*k+rr] }
				}
				naive(m, n, k, a, bat, want)
				got := append([]float32(nil), c0...)
				MulAdd(m, n, k, a, src, got, n, threads)
				for i := range got {
					if d := math.Abs(float64(got[i] - want[i])); d > 1e-4*math.Sqrt(float64(k))+1e-5 {
						t.Fatalf("%dx%dx%d trans %v threads %d: c[%d] = %v, want %v", m, n, k, trans, threads, i, got[i], want[i])
					}
				}
			}
		}
	}
}

func TestKernelGo(t *testing.T) {
	if !haveKernel() {
		t.Skip("no assembly kernel")
	}
	r := rand.New(rand.NewPCG(3, 4))
	const kc = 37
	a, b := randf(r, kc*MR), randf(r, kc*NR)
	c1, c2 := randf(r, MR*20), make([]float32, MR*20)
	copy(c2, c1)
	kernel(kc, a, b, c1, 20)
	kernelGo(kc, a, b, c2, 20)
	for i := range c1 {
		if math.Abs(float64(c1[i]-c2[i])) > 1e-4 {
			t.Fatalf("c[%d] %v and %v", i, c1[i], c2[i])
		}
	}
}

func BenchmarkMulAdd(b *testing.B) {
	r := rand.New(rand.NewPCG(5, 6))
	const m, n, k = 256, 4096, 576
	a := A{Data: randf(r, m*k), Stride: k}
	src := Matrix{Data: randf(r, k*n), Stride: n, N: n}
	c := make([]float32, m*n)
	b.ResetTimer()
	for range b.N {
		MulAdd(m, n, k, a, src, c, n, 1)
	}
	b.ReportMetric(float64(2*m*n*k)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOPS")
}
