package palette

import (
	"math"
	"math/rand"
	"testing"
)

// The tables fitted with the columns' weights leave less of the weighted error than those
// fitted without; codes written with them (column blocks too) are read back as they were.
func TestQuantizeWeightedAndCodes(t *testing.T) {
	rows, cols := 32, 64
	rng := rand.New(rand.NewSource(1))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	cw := make([]float32, cols) // a few columns whose inputs are large
	for j := range cw {
		cw[j] = 0.01
		if j%16 == 3 {
			cw[j] = 100
		}
	}
	werr := func(q LUT4) float64 {
		dq := q.Dequantize()
		var e float64
		for i, x := range w {
			d := float64(dq[i] - x)
			e += float64(cw[i%cols]) * d * d
		}
		return e
	}
	for _, cb := range []int{0, 16} {
		plain := QuantizeBits(w, rows, cols, 16, cb, 4, 20, KMeans)
		weighted := QuantizeWeighted(w, rows, cols, 16, cb, 4, 20, cw)
		if a, b := werr(weighted), werr(plain); !(a < 0.9*b) {
			t.Errorf("col block %d: weighted error %g with the weights, %g without", cb, a, b)
		}
		dir := t.TempDir()
		wr, err := NewCodesWriter(dir)
		if err != nil {
			t.Fatal(err)
		}
		if err := wr.Add("w", w, weighted); err != nil {
			t.Fatal(err)
		}
		if err := wr.Close(); err != nil {
			t.Fatal(err)
		}
		c, err := LoadCodes(dir)
		if err != nil {
			t.Fatal(err)
		}
		got, ok, err := c.Lookup(w, rows, cols, 16, 0, 4, 1)
		if err != nil || !ok {
			t.Fatalf("col block %d: lookup %v %v", cb, ok, err)
		}
		a, b := got.Dequantize(), weighted.Dequantize()
		for i := range a {
			if a[i] != b[i] || math.IsNaN(float64(a[i])) {
				t.Fatalf("col block %d: weight %d read back as %v, written %v", cb, i, a[i], b[i])
			}
		}
		if want := weighted.ColBlock; got.ColBlock != want {
			t.Errorf("col block %d: read back with blocks of %d", cb, got.ColBlock)
		}
		// asked for other blocks, the codes are not taken
		if cb > 0 {
			if _, ok, _ := c.Lookup(w, rows, cols, 16, cols, 4, 1); ok {
				t.Error("codes of column blocks taken for whole rows")
			}
		}
	}
}

// A block of a palettized matrix holds the same values, with tables of whole rows and of column
// blocks.
func TestSub(t *testing.T) {
	rows, cols := 48, 96
	rng := rand.New(rand.NewSource(3))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	for _, cb := range []int{0, 32} {
		q := QuantizeBits(w, rows, cols, 16, cb, 4, 4, KMeans)
		full := q.Dequantize()
		for _, s := range [][4]int{{16, 48, 32, 96}, {0, 16, 0, 32}, {0, 48, 64, 96}} {
			if cb == 0 && s[2] == 0 {
				s[3] = 40 // tables of whole rows: any columns
			}
			sub := q.Sub(s[0], s[1], s[2], s[3])
			got := sub.Dequantize()
			for r := 0; r < s[1]-s[0]; r++ {
				for c := 0; c < s[3]-s[2]; c++ {
					if a, b := got[r*(s[3]-s[2])+c], full[(s[0]+r)*cols+s[2]+c]; a != b {
						t.Fatalf("cols %d, block %v: (%d, %d) %v, want %v", cb, s, r, c, a, b)
					}
				}
			}
		}
	}
}
