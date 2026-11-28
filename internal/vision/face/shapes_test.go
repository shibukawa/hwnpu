package face

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// The outputs' shapes a Core ML package keeps in its metadata (a run without the ONNX file) read
// back as written; a package without them asks for the ONNX file.
func TestOutShapesMetadata(t *testing.T) {
	shapes := [][]int{{1, 1600, 1}, {1, 1600, 4}, {1, 400, 10}}
	if got := formatShapes(shapes); got != "1x1600x1;1x1600x4;1x400x10" {
		t.Fatalf("formatShapes = %q", got)
	}
	write := func(name string, meta map[string]string) string {
		dir := filepath.Join(t.TempDir(), name+".mlpackage")
		pkg, err := mlpackage.Create(dir)
		if err != nil {
			t.Fatal(err)
		}
		for k, v := range meta {
			pkg.Metadata[k] = v
		}
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		x := b.Input("x", mlpackage.Float16, 1, 4)
		y := b.NamedOp("relu", "y", map[string]any{"x": x}, mlpackage.Float16, 1, 4)
		ins := []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: []int{1, 4}}}
		outs := []mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: []int{1, 4}}}
		if err := pkg.Finish(b.Finish(y), 9, ins, outs); err != nil {
			t.Fatal(err)
		}
		return dir
	}
	got, err := readOutShapes(write("with", map[string]string{metaOutShapes: formatShapes(shapes)}))
	if err != nil || len(got) != len(shapes) {
		t.Fatalf("readOutShapes = %v, %v", got, err)
	}
	for i := range shapes {
		if !slices.Equal(got[i], shapes[i]) {
			t.Errorf("shape %d = %v, want %v", i, got[i], shapes[i])
		}
	}
	if _, err := readOutShapes(write("without", nil)); err == nil {
		t.Error("a package without the shapes read")
	}
}
