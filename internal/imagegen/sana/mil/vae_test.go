package mil

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

// TestVaeCoreAI runs the decoder of a small picture on the CPU in Core AI's stages (a block a
// stage, the norm's max with the channels last, the multiscale's grouped 1 x 1 as a dense one)
// and in the others', from the same latents: the same picture. HWNPU_TEST_SANA is a Sana
// repository (transformer/, vae/; the weights are read).
func TestVaeCoreAI(t *testing.T) {
	root := os.Getenv("HWNPU_TEST_SANA")
	if root == "" {
		t.Skip("HWNPU_TEST_SANA is not set")
	}
	w, err := OpenWeights(root, "")
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	const size = 128 // 4 x 4 latents
	rng := rand.New(rand.NewSource(1))
	z := make([]float32, 32*4*4)
	for i := range z {
		z[i] = float32(rng.NormFloat64())
	}
	decode := func(coreAI bool) []float32 {
		x := z
		for i, st := range VaeStages(size, size, coreAI) {
			x = runPart(t, Vae(size, size, i, 0.41407, coreAI), w, x, 1, st.C, st.H, st.W)
		}
		return x
	}
	a, b := decode(false), decode(true)
	if len(a) != len(b) || len(a) != 3*size*size {
		t.Fatalf("%d and %d values", len(a), len(b))
	}
	var worst, sum float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		worst = max(worst, d)
		sum += d
	}
	t.Logf("largest difference %.2g, mean %.2g", worst, sum/float64(len(a)))
	if worst > 0.02 {
		t.Errorf("the pictures differ by up to %.3g", worst)
	}
}
