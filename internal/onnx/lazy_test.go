package onnx

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"google.golang.org/protobuf/encoding/protowire"
)

// addModel is a model of y = Add(x, w) for x and w of n floats, w an initializer (raw_data), and
// s = Add(y, k) for a small initializer k of one.
func addModel(n int) []byte {
	tensor := func(name string, f []float32) []byte {
		var t []byte
		t = protowire.AppendTag(t, 1, protowire.VarintType)
		t = protowire.AppendVarint(t, uint64(len(f)))
		t = protowire.AppendTag(t, 2, protowire.VarintType)
		t = protowire.AppendVarint(t, Float)
		t = protowire.AppendTag(t, 8, protowire.BytesType)
		t = protowire.AppendString(t, name)
		raw := make([]byte, 4*len(f))
		for i, v := range f {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
		}
		t = protowire.AppendTag(t, 9, protowire.BytesType)
		return protowire.AppendBytes(t, raw)
	}
	node := func(op string, in []string, out string) []byte {
		var b []byte
		for _, s := range in {
			b = protowire.AppendTag(b, 1, protowire.BytesType)
			b = protowire.AppendString(b, s)
		}
		b = protowire.AppendTag(b, 2, protowire.BytesType)
		b = protowire.AppendString(b, out)
		b = protowire.AppendTag(b, 4, protowire.BytesType)
		return protowire.AppendString(b, op)
	}
	w := make([]float32, n)
	for i := range w {
		w[i] = float32(i) / 8
	}
	var g []byte
	g = protowire.AppendTag(g, 1, protowire.BytesType)
	g = protowire.AppendBytes(g, node("Add", []string{"x", "w"}, "y"))
	g = protowire.AppendTag(g, 1, protowire.BytesType)
	g = protowire.AppendBytes(g, node("Add", []string{"y", "k"}, "s"))
	g = protowire.AppendTag(g, 5, protowire.BytesType)
	g = protowire.AppendBytes(g, tensor("w", w))
	g = protowire.AppendTag(g, 5, protowire.BytesType)
	g = protowire.AppendBytes(g, tensor("k", []float32{0.5}))
	var m []byte
	m = protowire.AppendTag(m, 7, protowire.BytesType)
	return protowire.AppendBytes(m, g)
}

func TestLoadLazy(t *testing.T) {
	const n = lazyMin * 2
	p := filepath.Join(t.TempDir(), "add.onnx")
	if err := os.WriteFile(p, addModel(n), 0o644); err != nil {
		t.Fatal(err)
	}
	eager, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	lazy, err := LoadLazy(p)
	if err != nil {
		t.Fatal(err)
	}
	w, k := lazy.Init["w"], lazy.Init["k"]
	if !w.Lazy() || w.F != nil || k.Lazy() || eager.Init["w"].Lazy() {
		t.Fatalf("lazy: w %v, k %v, eager w %v", w.Lazy(), k.Lazy(), eager.Init["w"].Lazy())
	}
	if !slices.Equal(w.Values(), eager.Init["w"].F) {
		t.Fatal("the lazy values differ")
	}
	x := &Tensor{Shape: []int{n}, F: make([]float64, n), Type: Float}
	for i := range x.F {
		x.F[i] = float64(i)
	}
	want, err := eager.Eval(map[string]*Tensor{"x": x}, []string{"s"})
	if err != nil {
		t.Fatal(err)
	}
	got, err := lazy.Eval(map[string]*Tensor{"x": x}, []string{"s"})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got["s"].F, want["s"].F) || want["s"].F[3] != 3+3.0/8+0.5 {
		t.Fatalf("eval: %v, want %v", got["s"].F[:4], want["s"].F[:4])
	}
	if !w.Lazy() {
		t.Fatal("Eval kept the values of a lazy initializer")
	}
	lazy.DecodeFor([]string{"y"}, []string{"s"}) // from y: w is not read
	if !w.Lazy() {
		t.Fatal("DecodeFor decoded an initializer the nodes do not read")
	}
	lazy.DecodeFor([]string{"x"}, []string{"s"})
	if w.Lazy() || !slices.Equal(w.F, eager.Init["w"].F) {
		t.Fatal("DecodeFor left the initializer lazy")
	}
}
