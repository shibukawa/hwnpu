package notes

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/onnx"
)

// On the Neural Engine (BONSAIAI_TEST_NPU), the program gives a window's activations.
func TestOnDevice(t *testing.T) {
	path, ref := os.Getenv("BONSAIAI_TEST_BASICPITCH_FILE"), os.Getenv("BONSAIAI_TEST_BASICPITCH_REF")
	if path == "" || ref == "" || os.Getenv("BONSAIAI_TEST_NPU") == "" {
		t.Skip("BONSAIAI_TEST_BASICPITCH_FILE, BONSAIAI_TEST_BASICPITCH_REF and BONSAIAI_TEST_NPU are not set")
	}
	ctx := context.Background()
	dev, _, err := llm.OpenDevice(ctx, "", "")
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	g, err := onnx.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	p, _, err := newProgram(g)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "bp.mlpackage")
	if _, err := llm.WritePart(dir, p, nil, llm.Options{Format: "fp16"}, nil); err != nil {
		t.Fatal(err)
	}
	tr, err := Open(ctx, dev, path, dir)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	y, err := tr.window(ctx, readF32(t, ref+".cut.bin"))
	if err != nil {
		t.Fatal(err)
	}
	want := readF32(t, ref+".window.bin")
	for k, part := range []struct {
		name     string
		from, to int
	}{{"note", 0, Pitches}, {"onset", Pitches, 2 * Pitches}, {"contour", 2 * Pitches, outputs}} {
		var g, w []float32
		for f := range WindowFrames {
			g = append(g, y[f*outputs+part.from:f*outputs+part.to]...)
			w = append(w, want[f*outputs+part.from:f*outputs+part.to]...)
		}
		t.Logf("%d %s: relative error %.3g", k, part.name, relErr(g, w))
	}
}
