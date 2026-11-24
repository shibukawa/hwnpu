package sfx

import (
	"bytes"
	"encoding/binary"
	"math"
	"path/filepath"
	"slices"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/safetensors"
)

func f32s(v ...float32) safetensors.Out {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(x))
	}
	return safetensors.Out{DType: "F32", Shape: []int{1, len(v)}, Size: int64(len(b)), Data: bytes.NewReader(b)}
}

func named(name string, o safetensors.Out) safetensors.Out { o.Name = name; return o }

// The tables written from two checkpoints (one under a prefix) open with the prefix "" and give
// the same tensors by the same names: what Open reads in place of the checkpoints.
func TestWriteTables(t *testing.T) {
	dir := t.TempDir()
	text, dit := filepath.Join(dir, "text.safetensors"), filepath.Join(dir, "dit.safetensors")
	if err := safetensors.Write(text, []safetensors.Out{named("model.norm.weight", f32s(1, 2)), named("other", f32s(9))}, nil); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Write(dit, []safetensors.Out{named("scale_shift_table", f32s(3, 4, 5))}, nil); err != nil {
		t.Fatal(err)
	}
	w, err := OpenWeights(nil, map[string][]string{"": {text}, "dit.": {dit}})
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	out := filepath.Join(dir, TablesFile)
	names := []string{"model.norm.weight", "dit.scale_shift_table"}
	if err := w.WriteTables(names, out); err != nil {
		t.Fatal(err)
	}
	if err := w.WriteTables([]string{"dit.missing"}, filepath.Join(dir, "x")); err == nil {
		t.Error("a missing tensor was written")
	}
	tw, err := OpenWeights(nil, map[string][]string{"": {out}})
	if err != nil {
		t.Fatal(err)
	}
	defer tw.Close()
	for name, want := range map[string][]float32{"model.norm.weight": {1, 2}, "dit.scale_shift_table": {3, 4, 5}} {
		got, _, err := tw.Get(name)
		if err != nil || !slices.Equal(got, want) {
			t.Errorf("%s: %v %v, want %v", name, got, err, want)
		}
	}
	if _, _, err := tw.Get("other"); err == nil {
		t.Error("a tensor not asked for was written")
	}
	if row, err := tw.Rows("dit.scale_shift_table", 0, 1); err != nil || len(row) != 3 {
		t.Errorf("rows: %v %v", row, err)
	}
}
