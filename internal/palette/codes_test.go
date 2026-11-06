package palette

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// WeightKey is iron_q4.weight_key: this key is Python's for the same values.
func TestWeightKeyPython(t *testing.T) {
	w := make([]float32, 6)
	for i := range w {
		w[i] = float32(i)*0.37 - 1
	}
	if got, want := WeightKey(w, 2, 3), "a7b219085e7ded33b4b67dbd3a382fd380de49ff"; got != want {
		t.Fatalf("key %s, Python's %s", got, want)
	}
}

func TestCodesLookup(t *testing.T) {
	const rows, cols, group, bits = 32, 8, 16, 4
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(math.Sin(float64(i)))
	}
	dir := t.TempDir()
	key := WeightKey(w, rows, cols)
	idx := make([]byte, rows*cols)
	for i := range idx {
		idx[i] = byte(i % 16)
	}
	tab := make([]byte, rows/group*16*4)
	for i := 0; i < rows/group*16; i++ {
		binary.LittleEndian.PutUint32(tab[4*i:], math.Float32bits(float32(i)/8))
	}
	os.WriteFile(filepath.Join(dir, key+".idx"), idx, 0o644)
	os.WriteFile(filepath.Join(dir, key+".tab"), tab, 0o644)
	os.WriteFile(filepath.Join(dir, "index.json"), []byte(`{"`+key+`": {"name": "w", "fmt": "lut4", "shape": [32, 8], "group": 16},
		"other": {"name": "q", "fmt": "q4", "shape": [64, 512]}}`), 0o644)
	c, err := LoadCodes(dir)
	if err != nil {
		t.Fatal(err)
	}
	if c.Len() != 1 {
		t.Fatalf("%d codes, want the lut4 one", c.Len())
	}
	q, ok, err := c.Lookup(w, rows, cols, group, 0, bits, 2)
	if err != nil || !ok {
		t.Fatalf("lookup: %v %v", ok, err)
	}
	d := q.Dequantize()
	for i := range d {
		r := i / cols
		want := float32(r/group*16+i%16) / 8 * 2
		if d[i] != want {
			t.Fatalf("weight %d: %g, want %g", i, d[i], want)
		}
	}
	if _, ok, _ := c.Lookup(w, rows, cols, group, 0, 6, 1); ok {
		t.Error("lut4 codes taken for lut6")
	}
	w[0] += 0.5
	if _, ok, _ := c.Lookup(w, rows, cols, group, 0, bits, 1); ok {
		t.Error("codes taken for another weight")
	}
}
