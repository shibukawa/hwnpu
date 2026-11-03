package lfm3audio

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
)

// The audio encoder lowered to OpenVINO IR computes the reference's embeddings of a clip shorter
// than the program on OpenVINO's CPU (TestAgainstFP64 for Intel's NPU), from both checkpoint
// layouts. Needs HWNPU_OV_PYTHON. The IR's output is fp16.
func TestIntelIR(t *testing.T) {
	py := ovcpu.Python()
	if py == "" {
		t.Skip("HWNPU_OV_PYTHON is not set")
	}
	for _, js := range []string{tiny, tinyOmni} {
		cfg, err := ParseConfig([]byte(js))
		if err != nil {
			t.Fatal(err)
		}
		w := scaled{llm.NewSynthetic(cfg)}
		const clip, frames = 37, 50
		mel := make([][]float64, clip)
		x := make([]float32, frames*cfg.MelBins)
		mask := make([]float32, frames)
		for i := range mel {
			mask[i] = 1
			mel[i] = make([]float64, cfg.MelBins)
			for j := range mel[i] {
				v := float32(round16(math.Sin(float64(i*cfg.MelBins+j)*0.61) * 1.2))
				x[i*cfg.MelBins+j] = v
				mel[i][j] = float64(v)
			}
		}
		want := reference(t, cfg, w, mel)
		path := filepath.Join(t.TempDir(), "m.xml")
		opts := llm.Options{Format: "fp16", Backend: llm.BackendIntel, Params: map[string]string{"frames": fmt.Sprint(frames)}}
		if _, err := llm.WritePart(path, cfg, w, opts, nil); err != nil {
			t.Fatal(err)
		}
		res, err := ovcpu.Run(py, path, map[string][]float32{"x": x, "mask": mask}, ovcpu.Options{})
		if err != nil {
			t.Fatal(err)
		}
		got := res["y"]
		if len(got) != Tokens(frames)*cfg.TextHidden {
			t.Fatalf("%d outputs", len(got))
		}
		e := relErr(got, want)
		if !(e <= 2e-3) {
			t.Errorf("omni %v: relative error %g against the reference", cfg.Omni, e)
		}
		t.Logf("omni %v: relative error %.2g", cfg.Omni, e)
	}
}
