package palette

import (
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/safetensors"
)

func relErr(w, q []float32) float64 {
	var se, ss float64
	for i := range w {
		d := float64(w[i] - q[i])
		se += d * d
		ss += float64(w[i]) * float64(w[i])
	}
	return math.Sqrt(se / ss)
}

// A 16 level Lloyd-Max quantizer of Gaussian data reaches about 20 dB SNR, a relative RMS
// error near 0.1; heavy tails make it worse, but not by much with a good start.
func TestQuantizeGaussian(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const rows, cols = 64, 1024
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	q := Quantize(w, rows, cols, 16, 0, 16, KMeans)
	e := relErr(w, q.Dequantize())
	t.Logf("gaussian, 16 row tables: rel RMS err %.4f", e)
	if e > 0.12 {
		t.Errorf("rel RMS err %.4f, expected about 0.10", e)
	}
	// heavy tails: a few values 20 sigma out
	for i := 0; i < len(w); i += 997 {
		w[i] *= 20
	}
	q = Quantize(w, rows, cols, 16, 0, 16, KMeans)
	t.Logf("with outliers, 16 row tables: rel RMS err %.4f", relErr(w, q.Dequantize()))
}

// Error on real weights when HWNPU_LFM3_DIR is set, for a few block shapes.
func TestQuantizeLFM3(t *testing.T) {
	dir := os.Getenv("HWNPU_LFM3_DIR")
	if dir == "" {
		t.Skip("HWNPU_LFM3_DIR not set")
	}
	f, err := safetensors.Open(filepath.Join(dir, "LFM3-1.7B.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	for _, name := range []string{"model.layers.0.feed_forward.w1.weight", "model.layers.2.self_attn.q_proj.weight", "model.layers.0.conv.in_proj.weight"} {
		w, err := f.Float32(name)
		if err != nil {
			t.Fatal(err)
		}
		tt, _ := f.Lookup(name)
		rows, cols := tt.Shape[0], tt.Shape[1]
		for _, blk := range [][2]int{{16, 0}, {16, 64}, {1, 0}} {
			for _, mode := range []Mode{KMeans, Uniform, KMeansMinMax} {
				q := Quantize(w, rows, cols, blk[0], blk[1], 16, mode)
				t.Logf("%s %dx%d block %dx%d mode %d: rel RMS err %.4f", name, rows, cols, blk[0], q.ColBlock, mode, relErr(w, q.Dequantize()))
			}
		}
	}
}

// Wider tables halve the error with every two bits or so, and the indices read back as packed.
func TestQuantizeBits(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const rows, cols = 32, 1000
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	prev := 1.0
	for _, bits := range []int{4, 6, 8} {
		q := QuantizeBits(w, rows, cols, 16, 0, bits, 8, KMeansMinMax)
		if len(q.Indices) != (rows*cols*bits+7)/8 || len(q.LUT) != rows/16<<bits {
			t.Fatalf("%d bits: %d index bytes, %d table entries", bits, len(q.Indices), len(q.LUT))
		}
		d := q.Dequantize()
		e := relErr(w, d)
		t.Logf("%d bits: rel RMS err %.4f", bits, e)
		if e > prev/2 {
			t.Errorf("%d bits: rel RMS err %.4f, not under half of %.4f", bits, e, prev)
		}
		prev = e
		// every value is an entry of its row's table
		for i, v := range d {
			k := q.index(i)
			if want := f16val(q.LUT[(i/cols/16)<<bits+k]); v != want {
				t.Fatalf("%d bits: value %d is %g, table entry %g", bits, i, v, want)
			}
		}
	}
	if Quantize(w, rows, cols, 16, 0, 8, KMeansMinMax).Bits != 4 {
		t.Error("Quantize is not 4 bits")
	}
}

// Symmetric linear quantization of Gaussian rows: int8 keeps a relative RMS error under 1%, int4
// per row near 0.14 (7 levels each side of a maximum near 3.5 sigma), better with groups of 128
// columns and with the clipping search. The values packed are those Dequantize reads back.
func TestQuantizeLinear(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	const rows, cols = 33, 1024 // an odd number of rows: the last 4 bit row pair is a single row
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	for _, c := range []struct {
		bits, block int
		search      bool
		max         float64
	}{
		{8, 0, false, 0.01},
		{4, 0, false, 0.15},
		{4, 0, true, 0.12},
		{4, 128, false, 0.125},
		{4, 128, true, 0.11},
	} {
		l := QuantizeLinear(w, rows, cols, c.block, c.bits, c.search)
		e := relErr(w, l.Dequantize())
		t.Logf("int%d block %d search %v: rel RMS err %.4f", c.bits, c.block, c.search, e)
		if e > c.max {
			t.Errorf("int%d block %d search %v: rel RMS err %.4f, expected under %.2f", c.bits, c.block, c.search, e, c.max)
		}
	}
	// exact levels come back exactly
	l := QuantizeLinear([]float32{-7, -1, 0, 3, 7, 2}, 1, 6, 0, 4, false)
	for i, want := range []float32{-7, -1, 0, 3, 7, 2} {
		if got := l.Dequantize()[i]; got != want {
			t.Errorf("element %d: %v, want %v", i, got, want)
		}
	}
}
