package main

import (
	"bytes"
	"encoding/json"
	"math/rand/v2"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/palette"
)

func q4(rows, cols int, seed uint64) palette.Q4NX {
	rng := rand.New(rand.NewPCG(seed, 1))
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64())
	}
	return palette.QuantizeQ4NX(w, rows, cols, false)
}

// The q4nx file holds each matrix's kernel blocks at a 64 byte aligned offset, once, and the
// index says where; a matrix that does not cut into 32 x 256 blocks is listed but not written.
func TestQ4NXWriter(t *testing.T) {
	base := filepath.Join(t.TempDir(), "m")
	w, err := newQ4NXWriter(base)
	if err != nil {
		t.Fatal(err)
	}
	a, b, odd := q4(32, 256, 1), q4(64, 512, 2), q4(16, 256, 3)
	w.add("a", a)
	w.add("odd", odd)
	w.add("b", b)
	w.add("a", q4(32, 256, 4)) // a later tier's copy of the same constant
	if err := w.close(); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(base + ".q4nx.json")
	if err != nil {
		t.Fatal(err)
	}
	var idx q4nxIndex
	if err := json.Unmarshal(raw, &idx); err != nil {
		t.Fatal(err)
	}
	if len(idx.Matrices) != 2 || len(idx.Unblocked) != 1 || idx.Unblocked[0].Name != "odd" || idx.Group != 32 || idx.Block != [2]int{32, 256} {
		t.Fatalf("index %+v", idx)
	}
	data, err := os.ReadFile(base + ".q4nx")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		m q4nxMatrix
		q palette.Q4NX
	}{{idx.Matrices[0], a}, {idx.Matrices[1], b}} {
		if c.m.Offset%64 != 0 {
			t.Errorf("%s at offset %d", c.m.Name, c.m.Offset)
		}
		if got := data[c.m.Offset : c.m.Offset+int64(c.m.Bytes)]; !bytes.Equal(got, c.q.Blocks()) {
			t.Errorf("%s: the blocks differ", c.m.Name)
		}
	}
}

func TestUsesQ4NX(t *testing.T) {
	for _, c := range []struct {
		o    llm.Options
		want bool
	}{{llm.Options{Format: "q4nx"}, true}, {llm.Options{Format: "lut4"}, false}, {llm.Options{Format: "fp16", FormatFor: []string{"mlp=q4nx"}}, true}} {
		if got := usesQ4NX(c.o); got != c.want {
			t.Errorf("%+v: %v", c.o, got)
		}
	}
}
