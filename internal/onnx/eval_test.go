package onnx

import "testing"

// binop without its tables of indices (an operand of the result's shape, the other repeated
// whole) gives what the tables give.
func TestBinopRepeats(t *testing.T) {
	seq := func(shape ...int) *Tensor {
		x := &Tensor{Shape: shape, F: make([]float64, Len(shape)), Type: Float}
		for i := range x.F {
			x.F[i] = float64(i*7%11) + 0.5
		}
		return x
	}
	sub := func(x, y float64) float64 { return x - 2*y }
	for _, c := range [][2][]int{
		{{2, 3, 4}, {2, 3, 4}}, {{2, 3, 4}, {}}, {{}, {2, 3, 4}}, {{2, 3, 4}, {4}}, {{3, 4}, {2, 3, 4}},
		{{2, 3, 4}, {1, 3, 4}}, {{1, 1, 4}, {2, 3, 4}}, {{2, 3, 4}, {1, 1, 1}}, {{}, {}},
		{{2, 3, 4}, {3, 1}}, {{2, 1, 4}, {3, 1}}, {{2, 3, 1}, {2, 3, 4}}, {{1, 4}, {3, 1}}, // by the tables
	} {
		a, b := seq(c[0]...), seq(c[1]...)
		got, err := binop(a, b, Float, sub)
		if err != nil {
			t.Fatal(err)
		}
		ia, ib := index(a.Shape, got.Shape), index(b.Shape, got.Shape)
		for k := range got.F {
			if want := sub(a.F[ia[k]], b.F[ib[k]]); got.F[k] != want {
				t.Fatalf("%v and %v: [%d] %v, want %v", c[0], c[1], k, got.F[k], want)
			}
		}
	}
	for _, c := range []struct {
		x, out []int
		want   int
	}{{[]int{4}, []int{2, 3, 4}, 4}, {[]int{1, 3, 4}, []int{2, 3, 4}, 12}, {[]int{3, 1}, []int{2, 3, 4}, 0},
		{nil, []int{2, 3}, 1}, {[]int{2, 3}, []int{2, 3}, 6}, {[]int{2, 1, 1}, []int{2, 3, 4}, 0}} {
		if got := repeats(c.x, c.out); got != c.want {
			t.Errorf("repeats(%v, %v) = %d, want %d", c.x, c.out, got, c.want)
		}
	}
}
