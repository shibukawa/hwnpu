package ailower

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/milspec"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// streamFn builds a function of two weights over the inline limit (resources), one of them used
// twice, with seed's values.
func streamFn(t *testing.T, seed float32) (*milspec.Function, io.ReaderAt) {
	f16 := mlpackage.Float16
	b, bw, path := newBuilder(t)
	x := b.Input("x", f16, 1, 32, 8)
	vals := func(n int, k float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(i%97)*k + seed
		}
		return v
	}
	w1, err := b.F16s("w1", []int{1, 32, 8}, vals(32*8, 0.01))
	if err != nil {
		t.Fatal(err)
	}
	w2, err := b.F16s("w2", []int{1, 32, 8}, vals(32*8, 0.02))
	if err != nil {
		t.Fatal(err)
	}
	mul := func(x, w *mlpackage.Var) *mlpackage.Var {
		return b.Op("mul", map[string]any{"x": x, "y": w}, f16, 1, 32, 8)
	}
	y := mul(mul(mul(x, w1), w2), w1)
	fn := b.Function(y)
	if err := bw.Close(); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return fn, bytes.NewReader(raw)
}

// TestWriteStreamed writes the same bytes as Lower and LowerMany, the hash of them, with the
// weights through the spill file (removed after).
func TestWriteStreamed(t *testing.T) {
	for _, keep := range []int64{0, 600, 1 << 30} { // all spilled, the first blob kept, all kept
		t.Run(fmt.Sprint(keep), func(t *testing.T) {
			defer func(k int64) { SpillKeep = k }(SpillKeep)
			SpillKeep = keep
			testWriteStreamed(t)
		})
	}
}

func testWriteStreamed(t *testing.T) {
	dir := t.TempDir()
	check := func(name string, want *Program, got *Program) {
		t.Helper()
		b, err := os.ReadFile(filepath.Join(dir, name, "main.mlirb"))
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(b, want.Bytecode) {
			t.Errorf("%s: %d bytes streamed, not the %d of the bytecode in memory", name, len(b), len(want.Bytecode))
		}
		if got.Size != int64(len(want.Bytecode)) {
			t.Errorf("%s: Size %d, want %d", name, got.Size, len(want.Bytecode))
		}
		h, _ := os.ReadFile(filepath.Join(dir, name, "main.hash"))
		if sum := sha256.Sum256(want.Bytecode); !bytes.Equal(h, sum[:]) {
			t.Errorf("%s: main.hash is not the bytecode's", name)
		}
		if _, err := os.Stat(filepath.Join(dir, name+".blobs.tmp")); err == nil {
			t.Errorf("%s: the spill file is left", name)
		}
	}
	fn, w := streamFn(t, 0)
	want, err := Lower(fn, w)
	if err != nil {
		t.Fatal(err)
	}
	called := false
	got, err := WriteOne(filepath.Join(dir, "one.aimodel"), fn, w, func() error { called = true; return nil })
	if err != nil {
		t.Fatal(err)
	}
	if len(want.Bytecode) < 1024 {
		t.Fatalf("%d bytes: the weights are not resources", len(want.Bytecode))
	}
	check("one.aimodel", want, got)
	if !called {
		t.Error("lowered was not called")
	}
	if _, err := os.Stat(filepath.Join(dir, "one.aimodel", IOFile)); err != nil {
		t.Errorf("one function: no %s: %v", IOFile, err)
	}

	fn2, w2 := streamFn(t, 1)
	fns, ws, names := []*milspec.Function{fn, fn2}, []io.ReaderAt{w, w2}, []string{"a", "b"}
	want, err = LowerMany(fns, ws, names)
	if err != nil {
		t.Fatal(err)
	}
	got, err = WriteMany(filepath.Join(dir, "many.aimodel"), fns, ws, names, nil)
	if err != nil {
		t.Fatal(err)
	}
	check("many.aimodel", want, got)
}
