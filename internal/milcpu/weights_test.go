package milcpu

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/vecops"
)

// linear of a weight kept as its table (lutWeight) or in fp16 (f16Weight) is linear of the same
// weight in float32, to the bit: the packing differs, not the products.
func TestLinearKeptWeights(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	for _, c := range []struct{ m, n, k, gn, gk int }{
		{1, 300, 520, 300, 1}, // a table a row, k past one block of sgemm's
		{3, 64, 96, 8, 4},     // a grid of blocks of rows and columns
		{7, 33, 17, 1, 1},     // one table, n not a multiple of the panel
	} {
		const size = 256
		idx := make([]byte, c.n*c.k)
		for i := range idx {
			idx[i] = byte(r.IntN(size))
		}
		lut := make([]float32, c.gn*c.gk*size)
		for i := range lut {
			lut[i] = r.Float32()*2 - 1
		}
		x := &value{shape: []int{c.m, c.k}, f: make([]float32, c.m*c.k)}
		for i := range x.f {
			x.f[i] = r.Float32()*2 - 1
		}
		dense, err := lutToDense(&value{shape: []int{c.n, c.k}, q: idx}, &value{shape: []int{c.gn, c.gk, size, 1}, f: lut})
		if err != nil {
			t.Fatal(err)
		}
		want, err := linear(x, dense, nil)
		if err != nil {
			t.Fatal(err)
		}
		w, err := newLUTWeight(&value{shape: []int{c.n, c.k}, q: idx}, &value{shape: []int{c.gn, c.gk, size, 1}, f: lut})
		if err != nil {
			t.Fatal(err)
		}
		got, err := linear(x, w, nil)
		if err != nil || !slices.Equal(got.f, want.f) {
			t.Fatalf("%+v: the table's linear differs (%v)", c, err)
		}
		// fp16: the dense weight rounded to fp16, then the same products
		h := make([]byte, 2*len(dense.f))
		vecops.F32ToF16(h, dense.f)
		vecops.F16ToF32(dense.f, h)
		want, _ = linear(x, dense, nil)
		got, err = linear(x, &value{shape: []int{c.n, c.k}, f16: &f16Weight{n: c.n, k: c.k, h: h}}, nil)
		if err != nil || !slices.Equal(got.f, want.f) {
			t.Fatalf("%+v: the fp16 linear differs (%v)", c, err)
		}
	}
}
