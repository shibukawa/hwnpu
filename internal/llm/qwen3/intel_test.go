package qwen3

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

// On OpenVINO's CPU: the model split at its attention, converted to OpenVINO IR as hwnpu-chat
// converts it for Intel's NPU, with a program of 4 tokens per call and one of 1 sharing the
// host's cache, gives the logits of the MIL program without a cache. Needs HWNPU_OV_PYTHON.
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
	c := tinyConfig(t)
	w := llm.NewSynthetic(c)
	n := len(tokens)
	full := build(t, c, w, llm.Options{Seq: n, Format: "fp16"}).run(t, map[string][]float32{"x": rows(t, w, c.Embedding(), tokens)})["y"]
	dir := t.TempDir()
	var paths []string
	for _, q := range []int{4, 1} {
		opts := llm.Options{Seq: q, Cache: 13, Paged: 4, Format: "fp16", Backend: llm.BackendIntel}
		path, err := llm.Convert(Arch, c, w, filepath.Join(dir, fmt.Sprintf("m-q%d.xml", q)), "none", 0, opts, io.Discard)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	s, cl, err := runner.OpenSession(ctx, dev, paths, c)
	if err != nil {
		t.Fatal(err)
	}
	defer cl.Close()
	worst := 0.0
	for pos := 0; pos < n; {
		v := min(4, n-pos)
		if pos >= 4 {
			v = 1
		}
		logits := step(t, s, c, w, pos, pos+v)
		pos += v
		worst = math.Max(worst, relErr32(logits, full[(pos-1)*c.VocabSize:pos*c.VocabSize]))
	}
	if !(worst <= 5e-3) {
		t.Errorf("relative error %g against the program without a cache", worst)
	}
	t.Logf("worst relative error %.2g", worst)
}
