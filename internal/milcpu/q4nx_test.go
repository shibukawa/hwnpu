package milcpu

import (
	"math"
	"math/rand/v2"
	"os"
	"testing"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/palette"
)

// The kernel's arithmetic stays close to the float32 linear, and computes what its formula says.
func TestLinearQ4NXKernel(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	rows, cols, m := 64, 256, 3
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(rng.NormFloat64()) * 0.05
	}
	x := make([]float32, m*cols)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	q := palette.QuantizeQ4NX(w, rows, cols, true)
	scales, offsets := q.ShiftScale()
	runWith := func(opts Options) []float32 {
		t.Helper()
		path := t.TempDir() + "/w.bin"
		blobs, err := mlpackage.NewBlobWriter(path)
		if err != nil {
			t.Fatal(err)
		}
		b := mlpackage.NewBuilder("CoreML8", blobs)
		in := b.Input("x", mlpackage.Float16, m, cols)
		v, err := b.UInt4ShiftScaled("w", []int{rows, cols}, []int{1, palette.Q4NXGroup}, q.Codes, scales, offsets)
		if err != nil {
			t.Fatal(err)
		}
		y := b.NamedOp("linear", "y", map[string]any{"x": in, "weight": v}, mlpackage.Float16, m, rows)
		prog := b.Finish(y)
		if err := blobs.Close(); err != nil {
			t.Fatal(err)
		}
		f, err := os.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		p, err := Load(prog, f)
		if err != nil {
			t.Fatal(err)
		}
		res, err := p.Run(map[string]Tensor{"x": {Shape: []int{m, cols}, Data: x}}, opts)
		if err != nil {
			t.Fatal(err)
		}
		return res["y"].Data
	}
	plain, kernel, floor := runWith(Options{}), runWith(Options{Q4NXKernel: true}), runWith(Options{Q4NXKernel: true, Q4NXFloor: true})
	rel := func(a, b []float32) float64 {
		var d, n float64
		for i := range a {
			d += float64(a[i]-b[i]) * float64(a[i]-b[i])
			n += float64(b[i]) * float64(b[i])
		}
		return math.Sqrt(d / n)
	}
	if e := rel(kernel, plain); e == 0 || e > 0.02 {
		t.Errorf("kernel arithmetic: relative error %g against float32", e)
	}
	if e := rel(floor, kernel); e == 0 {
		t.Error("floor rounding changed nothing")
	}
	// one output by the formula
	j, i := 7, 2
	var acc float32
	for g := 0; g < cols/32; g++ {
		var dot, sx float32
		for c := g * 32; c < (g+1)*32; c++ {
			xb := palette.BF16Value(palette.BF16(x[i*cols+c]))
			dot += float32(q.Code(j*cols+c)) * xb
			sx += xb
		}
		gi := j*(cols/32) + g
		acc += palette.BF16Value(palette.BF16(dot)) * palette.BF16Value(q.Scales[gi])
		acc += palette.BF16Value(q.Mins[gi]) * palette.BF16Value(palette.BF16(sx))
	}
	if got := kernel[i*rows+j]; math.Abs(float64(got-acc)) > 1e-6*math.Abs(float64(acc))+1e-7 {
		t.Errorf("y[%d,%d] = %g, the formula gives %g", i, j, got, acc)
	}
}
