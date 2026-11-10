package milcpu

import (
	"math/rand/v2"
	"testing"
)

// TestElementBinary checks the collapsed runs against broadcast's walk on shapes of every
// broadcast pattern.
func TestElementBinary(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	val := func(shape ...int) *value {
		v := &value{shape: shape, f: make([]float32, numel(shape))}
		for i := range v.f {
			v.f[i] = float32(r.NormFloat64()) + 3
		}
		return v
	}
	cases := [][2][]int{
		{{2, 3, 4, 5}, {2, 3, 4, 5}}, {{2, 3, 4, 5}, {1, 3, 1, 1}}, {{1, 3, 1, 1}, {2, 3, 4, 5}},
		{{2, 3, 4, 5}, {5}}, {{4, 1}, {1, 6}}, {{2, 3, 4, 5}, {1}}, {{1}, {2, 3}}, {{2, 1, 4}, {3, 1}},
		{{6, 7}, {6, 1}}, {{1, 8, 1, 9}, {2, 1, 3, 9}}, {{}, {}},
	}
	for _, c := range cases {
		for _, typ := range []string{"add", "sub", "mul", "real_div", "maximum", "minimum"} {
			x, y := val(c[0]...), val(c[1]...)
			want, err := broadcast(x, y, binaryOp(typ))
			if err != nil {
				t.Fatal(err)
			}
			got, err := elementBinary(x, y, typ)
			if err != nil {
				t.Fatal(err)
			}
			if !sameShape(got.shape, want.shape) {
				t.Fatalf("%v %s %v: shape %v, want %v", c[0], typ, c[1], got.shape, want.shape)
			}
			for i := range want.f {
				if got.f[i] != want.f[i] {
					t.Fatalf("%v %s %v: [%d] %v, want %v", c[0], typ, c[1], i, got.f[i], want.f[i])
				}
			}
		}
	}
}
