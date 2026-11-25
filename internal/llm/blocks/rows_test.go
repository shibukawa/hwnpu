package blocks

import (
	"math"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/palette"
)

// A lut8r weight's rows each span their table: rows of magnitudes 1000 times apart in one table
// of 16 rows keep the small ones' error relative to their own size, unlike lut8's.
func TestRowScales(t *testing.T) {
	const rows, cols = 32, 64
	w := make([]float32, rows*cols)
	for r := range rows {
		mag := float32(math.Pow(10, -3*float64(r%4)/3))
		for c := range cols {
			w[r*cols+c] = mag * float32(math.Sin(float64(r*cols+c)))
		}
	}
	plain := palette.QuantizeBits(append([]float32(nil), w...), rows, cols, 16, 0, 8, 8, palette.Uniform).Dequantize()
	n := append([]float32(nil), w...)
	s := RowScales(n, rows, cols)
	scaled := palette.QuantizeBits(n, rows, cols, 16, 0, 8, 8, palette.Uniform).Dequantize()
	worst := func(d []float32, scale func(int) float32) float64 {
		var e float64
		for r := 3; r < rows; r += 4 { // the smallest rows
			var top, err float64
			for c := range cols {
				i := r*cols + c
				top = max(top, math.Abs(float64(w[i])))
				err = max(err, math.Abs(float64(d[i]*scale(r)-w[i])))
			}
			e = max(e, err/top)
		}
		return e
	}
	a := worst(plain, func(int) float32 { return 1 })
	b := worst(scaled, func(r int) float32 { return s[r] })
	if b > 0.01 || b > a/10 {
		t.Fatalf("the smallest rows' error over their size: lut8 %.4f, lut8r %.4f", a, b)
	}
	if s[0] != 1 {
		t.Fatalf("the largest row's scale %g, want 1", s[0])
	}
	if RowScaledBits("lut8r") != 8 || RowScaledBits("lut8") != 0 {
		t.Fatal("RowScaledBits")
	}
}
