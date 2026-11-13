package anima

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/imagegen/anima/mil"
)

// TestEmptyCond checks EmptyCond against diffusers' conditioning of an empty prompt: the row
// tools/ref/ref_anima.py's encode("") gives, as raw float32 in HWNPU_ANIMA_EMPTY_COND, of the
// checkpoint HWNPU_ANIMA_DIT in HWNPU_ANIMA_DIR (skipped without them).
func TestEmptyCond(t *testing.T) {
	ref, root, dit := os.Getenv("HWNPU_ANIMA_EMPTY_COND"), os.Getenv("HWNPU_ANIMA_DIR"), os.Getenv("HWNPU_ANIMA_DIT")
	if ref == "" || root == "" || dit == "" {
		t.Skip("HWNPU_ANIMA_EMPTY_COND, HWNPU_ANIMA_DIR and HWNPU_ANIMA_DIT not set")
	}
	b, err := os.ReadFile(ref)
	if err != nil {
		t.Fatal(err)
	}
	w, err := mil.OpenWeights(root, filepath.Base(dit))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	got, err := EmptyCond(w)
	if err != nil {
		t.Fatal(err)
	}
	var dot, gg, rr float64
	for i, g := range got {
		r := float64(math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:])))
		dot, gg, rr = dot+float64(g)*r, gg+float64(g)*float64(g), rr+r*r
	}
	cos := dot / math.Sqrt(gg*rr)
	t.Logf("cosine %.6f, |got| %.3f, |ref| %.3f", cos, math.Sqrt(gg), math.Sqrt(rr))
	if cos < 0.9999 {
		t.Errorf("cosine %.6f with diffusers' row", cos)
	}
}
