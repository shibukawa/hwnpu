package anima

import (
	"math"
	"testing"
)

// On data x0 ~ N(0, a^2) the flow's exact velocity is E[eps - x0 | x] = (sigma - (1 - sigma) a^2)
// x / var, var = (1 - sigma)^2 a^2 + sigma^2, and the ODE scales x by the std: from x at sigma 1
// it ends at a x. DPM-Solver++ 2M must land nearer than Euler in the same steps, and both must
// get there with many.
func TestDPM2M(t *testing.T) {
	const a = 0.5
	vel := func(x, s float64) float64 {
		v := (1-s)*(1-s)*a*a + s*s
		return (s - (1-s)*a*a) * x / v
	}
	solve := func(steps int, dpm bool) float64 {
		sig := Sigmas(steps, 3)
		x := []float32{1}
		var d0, d1 []float32
		var h float64
		for i := 0; i+1 < len(sig); i++ {
			v := []float32{float32(vel(float64(x[0]), sig[i]))}
			if dpm {
				h = dpm2m(x, v, &d0, &d1, sig[i], sig[i+1], h)
			} else {
				x[0] += float32(sig[i+1]-sig[i]) * v[0]
			}
		}
		return float64(x[0])
	}
	for _, n := range []int{10, 20} {
		e, d := math.Abs(solve(n, false)-a), math.Abs(solve(n, true)-a)
		t.Logf("%d steps: Euler error %.2e, DPM-Solver++ 2M %.2e", n, e, d)
		if d >= e {
			t.Errorf("%d steps: DPM-Solver++ 2M's error %.2e is not under Euler's %.2e", n, d, e)
		}
	}
	if d := math.Abs(solve(200, true) - a); d > 1e-3 {
		t.Errorf("200 steps: DPM-Solver++ 2M ends %.2e from %g", d, a)
	}
}
