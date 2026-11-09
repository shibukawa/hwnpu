package notes

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/onnx"
)

// On the Neural Engine (HWNPU_BASIC_PITCH_DEVICE=1), the program gives a window's activations.
func TestOnDevice(t *testing.T) {
	path, ref := os.Getenv("HWNPU_BASIC_PITCH"), os.Getenv("HWNPU_BASIC_PITCH_REF")
	if path == "" || ref == "" || os.Getenv("HWNPU_BASIC_PITCH_DEVICE") == "" {
		t.Skip("HWNPU_BASIC_PITCH, HWNPU_BASIC_PITCH_REF and HWNPU_BASIC_PITCH_DEVICE are not set")
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
