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

// QuantizeWeightedMinMax keeps a group's few large weights (its top level starts at the maximum),
// and its weighted error is no worse than the uniform levels' it starts from (Lloyd's passes do not
// raise it).
func TestQuantizeWeightedMinMax(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	rows, cols := 16, 256
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64() * 0.02)
	}
	w[5*cols+7] = 1.5 // an outlier, as the DiT's
	x2 := make([]float32, cols)
	for i := range x2 {
		x2[i] = float32(0.1 + rng.Float64())
	}
	werr := func(q LUT4) float64 {
		dq := q.Dequantize()
		s := 0.0
		for i, v := range w {
			d := float64(dq[i] - v)
			s += float64(x2[i%cols]) * d * d
		}
		return s
	}
	mm := QuantizeWeightedMinMax(w, rows, cols, 16, 0, 4, 20, x2)
	uni := QuantizeBits(w, rows, cols, 16, 0, 4, 8, Uniform)
	if d := mm.Dequantize()[5*cols+7]; math.Abs(float64(d)-1.5) > 0.05 {
		t.Errorf("the outlier comes out %g from min-max, want about 1.5", d)
	}
	if e, u := werr(mm), werr(uni); e > u*1.0001 {
		t.Errorf("min-max start's weighted error %g is past the uniform levels' %g", e, u)
	}
}
