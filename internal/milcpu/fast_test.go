package milcpu

import (
	"math"
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
			got, err := elementBinary(nil, x, y, typ)
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

// TestConv1d checks conv1d (conv2d's over one row) against the direct sum: dense, pointwise,
// grouped, depthwise, dilated, strided and padded.
func TestConv1d(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	val := func(shape ...int) *value {
		v := &value{shape: shape, f: make([]float32, numel(shape))}
		for i := range v.f {
			v.f[i] = float32(r.NormFloat64())
		}
		return v
	}
	for _, c := range []struct{ n, cin, w, cout, k, s, d, pl, pr, groups int }{
		{2, 24, 40, 64, 7, 1, 1, 0, 0, 1},
		{1, 64, 33, 128, 1, 1, 1, 0, 0, 1},
		{1, 64, 30, 64, 5, 1, 4, 8, 8, 64},
		{1, 32, 31, 64, 3, 2, 1, 1, 2, 4},
		{2, 16, 20, 16, 5, 1, 2, 3, 1, 1},
	} {
		x, w, b := val(c.n, c.cin, c.w), val(c.cout, c.cin/c.groups, c.k), val(c.cout)
		got, err := conv1d(nil, false, x, w, b, []int32{int32(c.s)}, []int32{int32(c.d)}, "custom", []int32{int32(c.pl), int32(c.pr)}, c.groups)
		if err != nil {
			t.Fatal(err)
		}
		ow := (c.w+c.pl+c.pr-c.d*(c.k-1)-1)/c.s + 1
		if want := []int{c.n, c.cout, ow}; !equalInts(got.shape, want) {
			t.Fatalf("%+v: shape %v, want %v", c, got.shape, want)
		}
		cg, per := c.cin/c.groups, c.cout/c.groups
		for bi := range c.n {
			for oc := range c.cout {
				g := oc / per
				for o := range ow {
					acc := float64(b.f[oc])
					for ic := range cg {
						for j := range c.k {
							if p := o*c.s + j*c.d - c.pl; p >= 0 && p < c.w {
								acc += float64(w.f[(oc*cg+ic)*c.k+j]) * float64(x.f[(bi*c.cin+g*cg+ic)*c.w+p])
							}
						}
					}
					if v := got.f[(bi*c.cout+oc)*ow+o]; math.Abs(float64(v)-acc) > 1e-4*(1+math.Abs(acc)) {
						t.Fatalf("%+v: y[%d][%d][%d] = %g, want %g", c, bi, oc, o, v, acc)
					}
				}
			}
		}
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
