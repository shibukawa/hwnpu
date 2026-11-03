package lfm3vl

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
)

// The vision encoder lowered to OpenVINO IR computes the reference's image embeddings on
// OpenVINO's CPU (TestAgainstFP64 for Intel's NPU). Needs HWNPU_OV_PYTHON. The IR's output is
// fp16.
func TestIntelIR(t *testing.T) {
	py := ovcpu.Python()
	if py == "" {
		t.Skip("HWNPU_OV_PYTHON is not set")
	}
	cfg, err := ParseConfig([]byte(tiny))
	if err != nil {
		t.Fatal(err)
	}
	w := scaled{llm.NewSynthetic(cfg)}
	const iw, ih = 24, 16
	gw, gh := iw/cfg.Vision.PatchSize, ih/cfg.Vision.PatchSize
	x := make([]float32, gw*gh*cfg.PatchDim())
	px := make([]float64, len(x))
	for i := range x {
		x[i] = float32(round16(math.Sin(float64(i)*0.37) * 1.5))
		px[i] = float64(x[i])
	}
	want := reference(t, cfg, w, px, gw, gh)
	path := filepath.Join(t.TempDir(), "m.xml")
	opts := llm.Options{Format: "fp16", Backend: llm.BackendIntel, Params: map[string]string{"image": fmt.Sprintf("%dx%d", iw, ih)}}
	if _, err := llm.WritePart(path, cfg, w, opts, nil); err != nil {
		t.Fatal(err)
	}
	res, err := ovcpu.Run(py, path, map[string][]float32{"x": x}, ovcpu.Options{})
	if err != nil {
		t.Fatal(err)
	}
	got := res["y"]
	if len(got) != len(want)*cfg.Text.HiddenSize {
		t.Fatalf("%d outputs", len(got))
	}
	e := relErr(got, want)
	if !(e <= 2e-3) {
		t.Errorf("relative error %g against the reference", e)
	}
	t.Logf("relative error %.2g", e)
}
