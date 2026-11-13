package qwen35

import (
	"context"
	"fmt"
	"io"
	"math"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
)

// On OpenVINO's CPU: the model split at its attention (Gated DeltaNet layers in the parts, their
// convolution windows and recurrent states carried by the host), converted to OpenVINO IR as
// hwnpu-chat converts it for Intel's NPU, with a program of 4 tokens per call (its first step
// padded) and one of 1 sharing the state, gives the logits of the MIL program without a cache.
// Also in int8, Intel's default. Needs HWNPU_OV_PYTHON.
func TestIntelIRPaged(t *testing.T) {
	if ovcpu.Python() == "" {
		t.Skip("HWNPU_OV_PYTHON is not set")
	}
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "openvino", "")
	if err != nil {
		t.Skip(err)
	}
	defer dev.Close()
	cfg := tinyConfig(t)
	w := llm.NewSynthetic(cfg)
	tokens := []int32{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	n := len(tokens)
	emb := embed16(t, cfg, w, tokens)
	row := func(i int) []float32 { return emb[i*cfg.HiddenSize : (i+1)*cfg.HiddenSize] }
	for _, k := range []struct {
		format string
		tol    float64
	}{{"fp16", 5e-3}, {"int8", 0.05}} {
		t.Run(k.format, func(t *testing.T) {
			full := runCPU(t, cfg, w, llm.Options{Seq: n, Format: k.format}, emb)
			dir := t.TempDir()
			var paths []string
			for _, q := range []int{4, 1} {
				opts := llm.Options{Seq: q, Cache: 12, Paged: 4, Format: k.format, Backend: llm.BackendIntel}
				path, err := llm.Convert(Arch, cfg, w, filepath.Join(dir, fmt.Sprintf("m-q%d.xml", q)), "none", 0, opts, io.Discard)
				if err != nil {
					t.Fatal(err)
				}
				paths = append(paths, path)
			}
			s, cl, err := runner.OpenSession(ctx, dev, paths, cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer cl.Close()
			worst := 0.0
			for pos := 0; pos < n; {
				v := 1
				switch {
				case pos == 0:
					v = 3 // a padded step of the 4-token program
				case pos < 7:
					v = 4
				}
				v = min(v, n-pos)
				var step [][]float32
				for i := 0; i < v; i++ {
					step = append(step, row(pos+i))
				}
				logits, err := s.Step(step)
				if err != nil {
					t.Fatal(err)
				}
				pos += v
				worst = math.Max(worst, relErr(logits, full[(pos-1)*cfg.VocabSize:pos*cfg.VocabSize]))
			}
			if !(worst <= k.tol) {
				t.Errorf("relative error %g against the program without a cache", worst)
			}
			t.Logf("worst relative error %.2g", worst)
		})
	}
}
