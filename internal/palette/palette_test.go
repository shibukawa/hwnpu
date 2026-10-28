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
