package f32

import (
	"math"
	"testing"
)

func ulps(a, b float32) float64 {
	return math.Abs(float64(int64(math.Float32bits(a)) - int64(math.Float32bits(b))))
}

// TestExp checks the vector loops against math.Exp (2 ulp; sigmoid and SiLU 4), the tails, NaN
// and the clamps.
func TestExp(t *testing.T) {
	var xs []float32
	for x := float32(-90); x < 90; x += 0.0173 {
		xs = append(xs, x)
	}
	xs = append(xs, float32(math.NaN()), 0, -0.0, 1e-30)
	got := make([]float32, len(xs))
	for _, c := range []struct {
		f    func(dst, src []float32)
		want func(float64) float64
		tol  float64
	}{
		{Exp, math.Exp, 2},
		{Sigmoid, func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }, 4},
		{SiLU, func(x float64) float64 { return x / (1 + math.Exp(-x)) }, 4},
	} {
		c.f(got, xs)
		for i, x := range xs {
			w := float32(c.want(float64(x)))
			switch {
			case x != x:
				if got[i] == got[i] {
					t.Errorf("NaN gave %v", got[i])
				}
			case x < -86.6 || x > 88.72283:
				// the clamps: e^x 1.2e-38 or +Inf; the sigmoids' limits
			case ulps(got[i], w) > c.tol && math.Abs(float64(got[i]-w)) > 1e-37:
				t.Fatalf("f(%v) = %v, want %v (%g ulp)", x, got[i], w, ulps(got[i], w))
			}
		}
	}
}

func TestAxpy(t *testing.T) {
	for _, n := range []int{0, 3, 8, 21} {
		d, x := make([]float32, n), make([]float32, n)
		for i := range d {
			d[i], x[i] = float32(i), float32(2*i+1)
		}
		Axpy(d, 0.5, x)
		for i, v := range d {
			if want := float32(i) + 0.5*float32(2*i+1); v != want {
				t.Fatalf("n %d: [%d] %v, want %v", n, i, v, want)
			}
		}
	}
}

func TestErf(t *testing.T) {
	var xs []float32
	for x := float32(-6); x < 6; x += 0.0071 {
		xs = append(xs, x)
	}
	got := make([]float32, len(xs))
	Erf(got, xs)
	for i, x := range xs {
		if d := math.Abs(float64(got[i]) - math.Erf(float64(x))); d > 5e-7 {
			t.Fatalf("Erf(%v) = %v, want %v", x, got[i], math.Erf(float64(x)))
		}
	}
}
