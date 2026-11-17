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

// A table WriteTables stores in int8 (a scale every group values of a row) reads back by Rows
// and Get within half a step of each group's scale, ranges across rows included; the other
// tables are copied as they are.
func TestWriteTablesInt8(t *testing.T) {
	dir := t.TempDir()
	const rows, cols, group = 5, 8, 4
	table := make([]float32, rows*cols)
	for i := range table {
		table[i] = float32(math.Sin(float64(i)*0.7)) * float32(1+i%cols/group*10)
	}
	table[2*cols] = 0 // a group of zeros keeps a zero scale
	table[2*cols+1], table[2*cols+2], table[2*cols+3] = 0, 0, 0
	src := filepath.Join(dir, "model.safetensors")
	writeSafetensors(t, src, map[string][]float32{"ple": table, "emb": {1, 2, 3, 4}}, map[string][]int{"ple": {rows, cols}, "emb": {2, 2}})
	a := &Arch{Name: "test"}
	w, err := OpenWeights(src, a)
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "tables.safetensors")
	if err := WriteTables(w, []string{"emb", "ple"}, map[string]int{"ple": group}, out); err != nil {
		t.Fatal(err)
	}
	w.Close()
	tw, err := OpenWeights(out, a)
	if err != nil {
		t.Fatal(err)
	}
	defer tw.Close()
	if e, err := tw.Rows("emb", 1, 2); err != nil || !slices.Equal(e, []float32{3, 4}) {
		t.Fatalf("emb row 1: %v %v", e, err)
	}
	all, shape, err := tw.Get("ple")
	if err != nil || !slices.Equal(shape, []int{rows, cols}) {
		t.Fatalf("Get: %v %v", shape, err)
	}
	for g := 0; g < rows*cols/group; g++ {
		m := float32(0)
		for _, v := range table[g*group : (g+1)*group] {
			m = max(m, float32(math.Abs(float64(v))))
		}
		for i := g * group; i < (g+1)*group; i++ {
			if d := math.Abs(float64(all[i] - table[i])); d > float64(m)/127*0.51+1e-6 {
				t.Errorf("value %d: %v for %v (largest %v)", i, all[i], table[i], m)
			}
		}
	}
	r, err := tw.Rows("ple", 1, 4)
	if err != nil || !slices.Equal(r, all[cols:4*cols]) {
		t.Fatalf("Rows 1..4 differ from Get: %v", err)
	}
}
