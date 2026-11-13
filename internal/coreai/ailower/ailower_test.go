package ailower

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/coreai/mlirbc"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// opsOf lowers the main function of the program the builder holds, its outputs outs, and
// returns the operation names of its bytecode.
func opsOf(t *testing.T, b *mlpackage.Builder, w *mlpackage.BlobWriter, path string, outs ...*mlpackage.Var) []string {
	t.Helper()
	prog := b.Finish(outs...)
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	weights, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	p, err := Lower(prog.GetFunctions()["main"], bytes.NewReader(weights))
	if err != nil {
		t.Fatal(err)
	}
	f, err := mlirbc.Read(p.Bytecode)
	if err != nil {
		t.Fatal(err)
	}
	return f.OpNames
}

func newBuilder(t *testing.T) (*mlpackage.Builder, *mlpackage.BlobWriter, string) {
	path := filepath.Join(t.TempDir(), "weight.bin")
	w, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	return mlpackage.NewBuilder("CoreML8", w), w, path
}

// The image models' operations, as coreai-torch writes them: max_pool as pad_v1 and
// max_pool_2d_v1, conv_transpose as conv_transpose2d_v1 then a slice for its crop, and a
// division by a constant as a product (Core AI's divide_v1 by a constant went wrong in PP-OCR's
// detector).
func TestImageOps(t *testing.T) {
	f16 := mlpackage.Float16
	b, bw, path := newBuilder(t)
	x := b.Input("x", f16, 1, 4, 12, 12)
	mp := b.Op("max_pool", map[string]any{"x": x, "kernel_sizes": b.Ints(3, 3), "strides": b.Ints(2, 2),
		"pad_type": b.Str("c", "custom"), "pad": b.Ints(1, 1, 1, 1), "ceil_mode": b.ScalarBool(false)}, f16, 1, 4, 6, 6)
	w, err := b.F16s("w", []int{4, 3, 3, 4}, make([]float32, 4*3*3*4))
	if err != nil {
		t.Fatal(err)
	}
	ct := b.Op("conv_transpose", map[string]any{"x": mp, "weight": w, "strides": b.Ints(2, 3),
		"pad_type": b.Str("c", "custom"), "pad": b.Ints(1, -2, 1, 0), "dilations": b.Ints(1, 1), "groups": b.ScalarI32(1)}, f16, 1, 3, 14, 18)
	y := b.NamedOp("real_div", "y", map[string]any{"x": ct, "y": b.ScalarF16(1.4140625)}, f16, 1, 3, 14, 18)
	ops := opsOf(t, b, bw, path, y)
	for _, want := range []string{"aicode.primitive.pad_v1", "aicode.primitive.max_pool_2d_v1", "aicode.primitive.conv_transpose2d_v1",
		"aicode.primitive.slice_v1", "aicode.primitive.mul_v1"} {
		if !slices.Contains(ops, want) {
			t.Errorf("no %s in %v", want, ops)
		}
	}
	if slices.Contains(ops, "aicode.primitive.divide_v1") {
		t.Errorf("a divide_v1 by a constant: %v", ops)
	}
}
