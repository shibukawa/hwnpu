package detect

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/onnx"
)

// TestBisect compares values of the graph (HWNPU_DETECT_AT, names separated by commas) on the
// NPU and on the CPU interpreter (fp32) from the reference's input: where the NPU drifts.
func TestBisect(t *testing.T) {
	path, ref, at := os.Getenv("HWNPU_DETECT"), os.Getenv("HWNPU_DETECT_REF"), os.Getenv("HWNPU_DETECT_AT")
	if path == "" || ref == "" || at == "" {
		t.Skip("HWNPU_DETECT, HWNPU_DETECT_REF and HWNPU_DETECT_AT are not set")
	}
	g, err := onnx.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	x := readF32(t, ref+".x.bin")
	d, _, err := llm.OpenDevice(context.Background(), "", t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	for _, name := range strings.Split(at, ",") {
		p := &onnx.Program{G: g, Inputs: []onnx.ValueInfo{{Name: g.Inputs[0].Name, Shape: []int{1, 3, Size, Size}}}, Outputs: []string{name}, Axis: -1, ExpSigmoid: os.Getenv("HWNPU_DETECT_EXP") != "0"}
		dir := filepath.Join(t.TempDir(), "p.mlpackage")
		if _, err := llm.WritePart(dir, p, nil, llm.Options{Format: "fp16"}, nil); err != nil {
			t.Fatal(err)
		}
		var ys [2][]float32
		for i, dev := range []any{nil, d} {
			var prog runner.Program
			if dev == nil {
				prog, err = runner.Open(context.Background(), nil, dir)
			} else {
				prog, err = runner.Open(context.Background(), d, dir)
			}
			if err != nil {
				t.Fatal(err)
			}
			if ys[i], err = prog.Run(context.Background(), map[string][]float32{"x": x}); err != nil {
				t.Fatal(err)
			}
			if i == 1 && !prog.OnNPU() {
				t.Logf("%s: not all on the NPU", name)
			}
			prog.Close()
		}
		var se, sw, worst float64
		for i := range ys[0] {
			e := float64(ys[1][i] - ys[0][i])
			se += e * e
			sw += float64(ys[0][i]) * float64(ys[0][i])
			worst = math.Max(worst, math.Abs(e))
		}
		t.Logf("%-8s %v: relative RMS error %.5f, largest %.4f", name, p.OutShapes(), math.Sqrt(se/sw), worst)
	}
}
