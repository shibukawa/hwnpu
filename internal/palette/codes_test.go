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

func TestCodesLookupLinear(t *testing.T) {
	const rows, cols = 4, 8
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(math.Cos(float64(i)))
	}
	dir := t.TempDir()
	key := WeightKey(w, rows, cols)
	data := make([]byte, rows*cols)
	for i := range data {
		data[i] = byte(int8(i - 16))
	}
	tab := make([]byte, rows*4)
	for r := range rows {
		binary.LittleEndian.PutUint32(tab[4*r:], math.Float32bits(float32(r+1)/128))
	}
	os.WriteFile(filepath.Join(dir, key+".idx"), data, 0o644)
	os.WriteFile(filepath.Join(dir, key+".tab"), tab, 0o644)
	os.WriteFile(filepath.Join(dir, "index.json"), []byte(`{"`+key+`": {"name": "w", "fmt": "int8", "shape": [4, 8], "group": 1}}`), 0o644)
	c, err := LoadCodes(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := c.Lookup(w, rows, cols, 16, 0, 8, 1); ok {
		t.Fatal("an int8 weight's codes taken as lut8")
	}
	q, ok, err := c.LookupLinear(w, rows, cols, 2)
	if err != nil || !ok {
		t.Fatalf("lookup: %v %v", ok, err)
	}
	for i := range data {
		r := i / cols
		got := float32(int8(q.Data[i])) * f16val(q.Scales[r])
		if want := float32(i-16) * float32(r+1) / 128 * 2; got != want {
			t.Fatalf("weight %d: %g, want %g", i, got, want)
		}
	}
	if _, ok, _ := c.LookupLinear(w[:cols*2], 2, cols, 1); ok {
		t.Fatal("codes of another weight")
	}
}

// A lutNr weight's codes: lutN's indices and tables, and the rows' scales; Lookup of lutN does
// not take them.
func TestCodesLookupRows(t *testing.T) {
	const rows, cols, group = 16, 8, 16
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(math.Sin(float64(i)))
	}
	dir := t.TempDir()
	key := WeightKey(w, rows, cols)
	idx := make([]byte, rows*cols)
	for i := range idx {
		idx[i] = byte(i % 256)
	}
	tab := make([]byte, 256*4)
	for i := range 256 {
		binary.LittleEndian.PutUint32(tab[4*i:], math.Float32bits(float32(i-128)/64))
	}
	sc := make([]byte, rows*4)
	for r := range rows {
		binary.LittleEndian.PutUint32(sc[4*r:], math.Float32bits(1/float32(r+1)))
	}
	os.WriteFile(filepath.Join(dir, key+".idx"), idx, 0o644)
	os.WriteFile(filepath.Join(dir, key+".tab"), tab, 0o644)
	os.WriteFile(filepath.Join(dir, key+".rows"), sc, 0o644)
	os.WriteFile(filepath.Join(dir, "index.json"), []byte(`{"`+key+`": {"name": "w", "fmt": "lut8r", "shape": [16, 8], "group": 16}}`), 0o644)
	c, err := LoadCodes(dir)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok, _ := c.Lookup(w, rows, cols, group, 0, 8, 1); ok {
		t.Fatal("a lut8r weight's codes taken as lut8")
	}
	q, s, ok, err := c.LookupRows(w, rows, cols, group, 0, 8)
	if err != nil || !ok {
		t.Fatalf("lookup: %v %v", ok, err)
	}
	if len(s) != rows || s[3] != 0.25 || q.Bits != 8 || f16val(q.LUT[130]) != 2.0/64 {
		t.Fatalf("scales %v, table entry %g", s, f16val(q.LUT[130]))
	}
	if _, _, ok, _ := c.LookupRows(w, rows, cols, group, 0, 6); ok {
		t.Fatal("lut8r codes taken as lut6r")
	}
}
