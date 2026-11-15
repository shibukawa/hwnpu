package anima

import (
	"encoding/binary"
	"math"
	"testing"
)

// The host's GroupNorm coefficients against the definition, from lines of sums as the device
// writes them: y = (x - mean) gamma / sqrt(var + eps) + beta = x a + b.
func TestHostNormCoefficients(t *testing.T) {
	const C, G, pixels, lines = 64, 32, 96, 3
	n := HostNorm{Name: "t", Channels: C, Groups: G, Pixels: pixels, Eps: 1e-6, Gamma: make([]float64, C), Beta: make([]float64, C)}
	x := make([][]float64, pixels) // pixel, channel
	for p := range x {
		x[p] = make([]float64, C)
		for c := range x[p] {
			x[p][c] = math.Sin(float64(7*p+3*c)) + float64(c%5) - 2
		}
	}
	for c := 0; c < C; c++ {
		n.Gamma[c], n.Beta[c] = 0.5+float64(c)/C, float64(c%3)-1
	}
	stats := make([]byte, lines*8*C)
	for l := 0; l < lines; l++ {
		for c := 0; c < C; c++ {
			var s, q float64
			for p := l * pixels / lines; p < (l+1)*pixels/lines; p++ {
				s += x[p][c]
				q += x[p][c] * x[p][c]
			}
			binary.LittleEndian.PutUint32(stats[l*8*C+4*c:], math.Float32bits(float32(s)))
			binary.LittleEndian.PutUint32(stats[l*8*C+4*(C+c):], math.Float32bits(float32(q)))
		}
	}
	row := make([]byte, 2*4*C+32)
	if err := n.Coefficients(stats, row); err != nil {
		t.Fatal(err)
	}
	limb := func(at int) float64 {
		return float64(fromBF16(binary.LittleEndian.Uint16(row[2*at:]))) + float64(fromBF16(binary.LittleEndian.Uint16(row[2*(C+at):])))
	}
	cg := C / G
	for g := 0; g < G; g++ {
		var s, q float64
		for c := g * cg; c < (g+1)*cg; c++ {
			for p := range x {
				s += x[p][c]
				q += x[p][c] * x[p][c]
			}
		}
		mean := s / float64(pixels*cg)
		sd := math.Sqrt(q/float64(pixels*cg) - mean*mean + n.Eps)
		for c := g * cg; c < (g+1)*cg; c++ {
			a, b := n.Gamma[c]/sd, n.Beta[c]-mean*n.Gamma[c]/sd
			// two bf16 limbs hold 16 bits; the sums came through float32
			if got := limb(c); math.Abs(got-a) > 2e-4*math.Abs(a) {
				t.Errorf("channel %d: factor %g, want %g", c, got, a)
			}
			if got := limb(2*C + c); math.Abs(got-b) > 2e-4*math.Max(1, math.Abs(b)) {
				t.Errorf("channel %d: term %g, want %g", c, got, b)
			}
		}
	}
	if err := n.Coefficients(stats[:8*C-4], row); err == nil {
		t.Error("statistics of no whole line: no error")
	}
}
