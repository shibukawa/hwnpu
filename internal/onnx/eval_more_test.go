package onnx

import (
	"math"
	"slices"
	"testing"
)

func seq(shape []int, f func(i int) float64) *Tensor {
	t := &Tensor{Shape: shape, F: make([]float64, Len(shape)), Type: Float}
	for i := range t.F {
		t.F[i] = f(i)
	}
	return t
}

func near(a, b []float64, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(a[i]-b[i]) > tol {
			return false
		}
	}
	return true
}

// TestConvTranspose checks the tiled path (stride = kernel) against the general one, and the
// general one on a small case by hand.
func TestConvTranspose(t *testing.T) {
	x := seq([]int{1, 3, 4, 5}, func(i int) float64 { return math.Sin(float64(i)) })
	w := seq([]int{3, 2, 2, 2}, func(i int) float64 { return math.Cos(float64(i) * 0.7) })
	b := &Tensor{Shape: []int{2}, F: []float64{0.5, -1}, Type: Float}
	n := &Node{Op: "ConvTranspose", Attrs: map[string]Attr{"strides": {Ints: []int64{2, 2}}, "kernel_shape": {Ints: []int64{2, 2}}}}
	got, err := convTranspose(n, x, w, b)
	if err != nil {
		t.Fatal(err)
	}
	ref := &Tensor{Shape: []int{1, 2, 8, 10}, F: make([]float64, 160), Type: Float}
	for c := range 3 {
		for m := range 2 {
			for y := range 4 {
				for i := range 5 {
					for ky := range 2 {
						for kx := range 2 {
							ref.F[(m*8+2*y+ky)*10+2*i+kx] += x.F[(c*4+y)*5+i] * w.F[((c*2+m)*2+ky)*2+kx]
						}
					}
				}
			}
		}
	}
	for m := range 2 {
		for j := range 80 {
			ref.F[m*80+j] += b.F[m]
		}
	}
	if !slices.Equal(got.Shape, ref.Shape) || !near(got.F, ref.F, 1e-5) {
		t.Errorf("tiled ConvTranspose %v differs from the sum", got.Shape)
	}
	// a 3 x 3 kernel, stride 2, pad 1: the general loop
	w3 := seq([]int{3, 2, 3, 3}, func(i int) float64 { return float64(i%5) - 2 })
	n3 := &Node{Op: "ConvTranspose", Attrs: map[string]Attr{"strides": {Ints: []int64{2, 2}}, "pads": {Ints: []int64{1, 1, 1, 1}}}}
	g3, err := convTranspose(n3, x, w3, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(g3.Shape, []int{1, 2, 7, 9}) {
		t.Errorf("shape %v", g3.Shape)
	}
	// output (m, 3, 4) gathers input (1, 2) at ky = kx = 1 ((3 + 1) / 2 = 2: y 2 ky 0 is 4 - 1 =
	// 3 too), so sum the taps that land there
	var want float64
	for c := range 3 {
		for y := range 4 {
			for i := range 5 {
				for ky := range 3 {
					for kx := range 3 {
						if 2*y+ky-1 == 3 && 2*i+kx-1 == 4 {
							want += x.F[(c*4+y)*5+i] * w3.F[((c*2+1)*3+ky)*3+kx]
						}
					}
				}
			}
		}
	}
	if got := g3.F[(1*7+3)*9+4]; math.Abs(got-want) > 1e-9 {
		t.Errorf("output (1, 3, 4) = %v, want %v", got, want)
	}
}

func TestResizeLinear(t *testing.T) {
	x := &Tensor{Shape: []int{1, 1, 2, 2}, F: []float64{0, 1, 2, 3}, Type: Float}
	n := &Node{Op: "Resize", Attrs: map[string]Attr{"mode": {S: "linear"}, "coordinate_transformation_mode": {S: "half_pixel"}}}
	got, err := resize(n, x, nil, &Tensor{Shape: []int{4}, F: []float64{1, 1, 4, 4}, Type: Int64})
	if err != nil {
		t.Fatal(err)
	}
	// half_pixel: output 0 at -0.25 (clamped to 0), 1 at 0.25, 2 at 0.75, 3 at 1.25 (1)
	row := []float64{0, 0.25, 0.75, 1}
	want := make([]float64, 16)
	for y, fy := range row {
		for i, fx := range row {
			want[y*4+i] = 2*fy + fx
		}
	}
	if !near(got.F, want, 1e-12) {
		t.Errorf("Resize = %v, want %v", got.F, want)
	}
}

func TestScatterNDAndNorms(t *testing.T) {
	data := seq([]int{2, 3}, func(i int) float64 { return float64(i) })
	idx := &Tensor{Shape: []int{1, 2}, F: []float64{1, 2}, Type: Int64}
	got, err := scatterND(data, idx, &Tensor{Shape: []int{1}, F: []float64{9}, Type: Float}, "none")
	if err != nil || !slices.Equal(got.F, []float64{0, 1, 2, 3, 4, 9}) {
		t.Errorf("ScatterND = %v, %v", got, err)
	}
	x := &Tensor{Shape: []int{1, 2, 2}, F: []float64{1, 3, 2, 6}, Type: Float}
	ln, err := layerNorm(x, &Tensor{Shape: []int{2}, F: []float64{1, 2}}, &Tensor{Shape: []int{2}, F: []float64{0, 1}}, -1, 0)
	if err != nil || !near(ln.F, []float64{-1, 3, -1, 3}, 1e-9) {
		t.Errorf("LayerNormalization = %v, %v", ln, err)
	}
	in := instanceNorm(&Tensor{Shape: []int{1, 1, 4}, F: []float64{1, 2, 3, 4}}, &Tensor{F: []float64{1}}, &Tensor{F: []float64{0}}, 0)
	s := math.Sqrt(1.25)
	if !near(in.F, []float64{-1.5 / s, -0.5 / s, 0.5 / s, 1.5 / s}, 1e-9) {
		t.Errorf("InstanceNormalization = %v", in.F)
	}
}

// TestIf checks that an If runs the branch its condition picks, reading an outer value.
func TestIf(t *testing.T) {
	th := &Graph{Init: map[string]*Tensor{"two": {Shape: []int{}, F: []float64{2}, Type: Float}},
		Nodes:   []*Node{{Op: "Mul", In: []string{"x", "two"}, Out: []string{"y"}}},
		Outputs: []ValueInfo{{Name: "y"}}}
	el := &Graph{Init: map[string]*Tensor{}, Nodes: []*Node{{Op: "Identity", In: []string{"x"}, Out: []string{"z"}}}, Outputs: []ValueInfo{{Name: "z"}}}
	n := &Node{Op: "If", In: []string{"c"}, Out: []string{"out"}, Graphs: []*Graph{th, el}}
	for _, b := range n.Graphs {
		n.In = append(n.In, b.outer()...)
	}
	if !slices.Equal(n.In, []string{"c", "x", "x"}) {
		t.Fatalf("If's inputs %v", n.In)
	}
	g := &Graph{Init: map[string]*Tensor{}, Nodes: []*Node{n}}
	for c, want := range map[float64]float64{1: 6, 0: 3} {
		out, err := g.Eval(map[string]*Tensor{"c": {Shape: []int{}, F: []float64{c}, Type: Bool}, "x": {Shape: []int{}, F: []float64{3}, Type: Float}}, []string{"out"})
		if err != nil || out["out"].F[0] != want {
			t.Errorf("If(%v) = %v, %v; want %v", c, out, err, want)
		}
	}
}

// TestResizeCubic checks a cubic Resize (half_pixel, a = -0.75) against the weights by hand at
// one output, and that a constant stays constant.
func TestResizeCubic(t *testing.T) {
	x := seq([]int{1, 1, 4, 4}, func(i int) float64 { return float64(i*i%7) - 3 })
	n := &Node{Op: "Resize", Attrs: map[string]Attr{"mode": {S: "cubic"}}}
	got, err := resize(n, x, nil, &Tensor{Shape: []int{4}, F: []float64{1, 1, 8, 8}, Type: Int64})
	if err != nil {
		t.Fatal(err)
	}
	// output (3, 5): source (3.5 / 2 - 0.5, 5.5 / 2 - 0.5) = (1.25, 2.25)
	w := func(f float64) ([4]int, [4]float64) {
		a, i0 := -0.75, math.Floor(f)
		u := f - i0
		ws := [4]float64{((a*(u+1)-5*a)*(u+1)+8*a)*(u+1) - 4*a, ((a+2)*u-(a+3))*u*u + 1, ((a+2)*(1-u)-(a+3))*(1-u)*(1-u) + 1}
		ws[3] = 1 - ws[0] - ws[1] - ws[2]
		var is [4]int
		for k := range 4 {
			is[k] = min(max(int(i0)-1+k, 0), 3)
		}
		return is, ws
	}
	iy, wy := w(1.25)
	ix, wx := w(2.25)
	var want float64
	for a := range 4 {
		for b := range 4 {
			want += x.F[iy[a]*4+ix[b]] * wy[a] * wx[b]
		}
	}
	if g := got.F[3*8+5]; math.Abs(g-want) > 1e-12 {
		t.Errorf("output (3, 5) = %v, want %v", g, want)
	}
	c, _ := resize(n, seq([]int{1, 1, 3, 3}, func(int) float64 { return 2 }), nil, &Tensor{F: []float64{1, 1, 7, 5}})
	for _, v := range c.F {
		if math.Abs(v-2) > 1e-12 {
			t.Fatalf("a constant resized to %v", v)
		}
	}
}
