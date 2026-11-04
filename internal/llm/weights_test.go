package llm

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// writeSafetensors writes F32 tensors (name -> shape and values) as one safetensors file.
func writeSafetensors(t *testing.T, path string, tensors map[string][]float32, shapes map[string][]int) {
	t.Helper()
	header := map[string]any{}
	var data []byte
	for _, name := range slices.Sorted(func(yield func(string) bool) {
		for n := range tensors {
			if !yield(n) {
				return
			}
		}
	}) {
		begin := len(data)
		for _, v := range tensors[name] {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(v))
		}
		header[name] = map[string]any{"dtype": "F32", "shape": shapes[name], "data_offsets": []int{begin, len(data)}}
	}
	h, _ := json.Marshal(header)
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	out = append(append(out, h...), data...)
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatal(err)
	}
}

// A checkpoint in two safetensors files opens by its index, or by the name of the single file it
// would be (model.safetensors), reads each tensor from the file the index names, and lists them
// all for the inventory; an index that names the wrong file for a tensor is an error.
func TestShardedSafetensors(t *testing.T) {
	dir := t.TempDir()
	writeSafetensors(t, filepath.Join(dir, "a.safetensors"), map[string][]float32{"x": {1, 2, 3, 4}}, map[string][]int{"x": {2, 2}})
	writeSafetensors(t, filepath.Join(dir, "b.safetensors"), map[string][]float32{"y": {5, 6}}, map[string][]int{"y": {2}})
	index := func(m map[string]string) string {
		raw, _ := json.Marshal(map[string]any{"metadata": map[string]any{}, "weight_map": m})
		p := filepath.Join(dir, "model.safetensors.index.json")
		if err := os.WriteFile(p, raw, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	a := &Arch{Name: "test"}
	p := index(map[string]string{"x": "a.safetensors", "y": "b.safetensors"})
	for _, path := range []string{p, filepath.Join(dir, "model.safetensors")} {
		w, err := OpenWeights(path, a)
		if err != nil {
			t.Fatal(err)
		}
		x, shape, err := w.Get("x")
		if err != nil || !slices.Equal(x, []float32{1, 2, 3, 4}) || !slices.Equal(shape, []int{2, 2}) {
			t.Errorf("x: %v %v %v", x, shape, err)
		}
		if r, err := w.Rows("x", 1, 2); err != nil || !slices.Equal(r, []float32{3, 4}) {
			t.Errorf("row 1 of x: %v %v", r, err)
		}
		if y, _, err := w.Get("y"); err != nil || !slices.Equal(y, []float32{5, 6}) {
			t.Errorf("y: %v %v", y, err)
		}
		if l, ok := w.(lister); !ok || !slices.Equal(l.Names(), []string{"x", "y"}) {
			t.Errorf("names: %v", w)
		}
		if _, _, err := w.Get("z"); err == nil {
			t.Error("read a tensor the checkpoint lacks")
		}
		w.Close()
	}
	index(map[string]string{"x": "b.safetensors", "y": "b.safetensors"})
	if _, err := OpenWeights(p, a); err == nil || !strings.Contains(err.Error(), "does not hold") {
		t.Errorf("an index naming the wrong file: %v", err)
	}
}
