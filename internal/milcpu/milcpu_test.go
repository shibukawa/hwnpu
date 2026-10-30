package milcpu

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/palette"
)

// run builds a one input program with build and runs it on x.
func run(t *testing.T, xShape []int, x []float32, build func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var) Tensor {
	t.Helper()
	path := filepath.Join(t.TempDir(), "weight.bin")
	blobs, err := mlpackage.NewBlobWriter(path)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", blobs)
	xv := b.Input("x", mlpackage.Float16, xShape...)
	y := build(b, xv)
	y = b.NamedOp("identity", "y", map[string]any{"x": y}, mlpackage.Float16, y.Shape...)
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
	out, err := p.Run(map[string]Tensor{"x": {Shape: xShape, Data: x}}, Options{})
	if err != nil {
		t.Fatal(err)
	}
	return out["y"]
}

func seq(n int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32(i)
	}
	return v
}

func near(t *testing.T, what string, got Tensor, shape []int, want []float32, tol float64) {
	t.Helper()
	if !sameShape(got.Shape, shape) {
		t.Fatalf("%s: shape %v, want %v", what, got.Shape, shape)
	}
	for i := range want {
		if math.Abs(float64(got.Data[i]-want[i])) > tol {
			t.Fatalf("%s: [%d] = %g, want %g (all %v)", what, i, got.Data[i], want[i], got.Data)
		}
	}
}

func f16s(t *testing.T, b *mlpackage.Builder, shape []int, v []float32) *mlpackage.Var {
	x, err := b.F16s("w", shape, v)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func TestElementwise(t *testing.T) {
	// broadcasting a row and a column, then the unary operations
	got := run(t, []int{2, 3}, seq(6), func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		col := f16s(t, b, []int{2, 1}, []float32{10, 20})
		y := b.Op("add", map[string]any{"x": x, "y": col}, mlpackage.Float16, 2, 3)
		row := f16s(t, b, []int{3}, []float32{1, 2, 3})
		return b.Op("mul", map[string]any{"x": y, "y": row}, mlpackage.Float16, 2, 3)
	})
	near(t, "add mul", got, []int{2, 3}, []float32{10, 22, 36, 23, 48, 75}, 0)

	x := []float32{-2, 0, 1.5}
	for _, c := range []struct {
		op   string
		want func(float64) float64
	}{
		{"silu", func(a float64) float64 { return a / (1 + math.Exp(-a)) }},
		{"sigmoid", func(a float64) float64 { return 1 / (1 + math.Exp(-a)) }},
		{"relu", func(a float64) float64 { return math.Max(a, 0) }},
	} {
		got := run(t, []int{3}, x, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
			return b.Op(c.op, map[string]any{"x": x}, mlpackage.Float16, 3)
		})
		var want []float32
		for _, v := range x {
			want = append(want, float32(c.want(float64(v))))
		}
		near(t, c.op, got, []int{3}, want, 1e-6)
	}
	for _, c := range []struct {
		mode string
		want []float32
	}{
		// torch.nn.functional.gelu(torch.tensor([-2., 0., 1.5]), approximate="none" / "tanh")
		{"EXACT", []float32{-0.04550028, 0, 1.39978919}},
		{"TANH_APPROXIMATION", []float32{-0.04540231, 0, 1.39957154}},
	} {
		got := run(t, []int{3}, x, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
			return b.Op("gelu", map[string]any{"x": x, "mode": b.Str("c", c.mode)}, mlpackage.Float16, 3)
		})
		near(t, "gelu "+c.mode, got, []int{3}, c.want, 1e-6)
	}
	got = run(t, []int{2, 4}, []float32{1, 2, 3, 4, 0, 0, 0, 8}, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		gamma := f16s(t, b, []int{4}, []float32{1, 1, 2, 2})
		beta := f16s(t, b, []int{4}, []float32{0, 0, 0, 1})
		return b.Op("layer_norm", map[string]any{"x": x, "axes": b.Ints(-1), "gamma": gamma, "beta": beta, "epsilon": b.ScalarF16(0)}, mlpackage.Float16, 2, 4)
	})
	// row 0: mean 2.5, std sqrt(1.25); row 1: mean 2, std sqrt(12)
	s0, s1 := float32(math.Sqrt(1.25)), float32(math.Sqrt(12))
	near(t, "layer_norm", got, []int{2, 4}, []float32{-1.5 / s0, -0.5 / s0, 2 * 0.5 / s0, 2*1.5/s0 + 1, -2 / s1, -2 / s1, 2 * -2 / s1, 2*6/s1 + 1}, 1e-6)
	got = run(t, []int{2}, []float32{3, 0}, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		return b.Op("rsqrt", map[string]any{"x": x, "epsilon": b.ScalarF16(1)}, mlpackage.Float16, 2)
	})
	near(t, "rsqrt", got, []int{2}, []float32{0.5, 1}, 1e-6)
}

func TestReductions(t *testing.T) {
	got := run(t, []int{2, 3}, seq(6), func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		return b.Op("reduce_mean", map[string]any{"x": x, "axes": b.Ints(-1), "keep_dims": b.ScalarBool(true)}, mlpackage.Float16, 2, 1)
	})
	near(t, "reduce_mean last", got, []int{2, 1}, []float32{1, 4}, 1e-6)
	got = run(t, []int{2, 3}, seq(6), func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		return b.Op("reduce_mean", map[string]any{"x": x, "axes": b.Ints(0), "keep_dims": b.ScalarBool(false)}, mlpackage.Float16, 3)
	})
	near(t, "reduce_mean first", got, []int{3}, []float32{1.5, 2.5, 3.5}, 1e-6)

	got = run(t, []int{2, 2}, []float32{0, 0, 1, 3}, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		return b.Op("softmax", map[string]any{"x": x, "axis": b.ScalarI32(-1)}, mlpackage.Float16, 2, 2)
	})
	e := float32(1 / (1 + math.Exp(2)))
	near(t, "softmax", got, []int{2, 2}, []float32{0.5, 0.5, e, 1 - e}, 1e-6)
	got = run(t, []int{2, 2}, []float32{0, 1, 0, 3}, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		return b.Op("softmax", map[string]any{"x": x, "axis": b.ScalarI32(0)}, mlpackage.Float16, 2, 2)
	})
	near(t, "softmax axis 0", got, []int{2, 2}, []float32{0.5, e, 0.5, 1 - e}, 1e-6)
}

func TestProducts(t *testing.T) {
	// linear: (2, 3) x (2 out, 3 in) transposed, with an fp16 weight from the weight file
	w := []float32{1, 0, -1, 2, 1, 0}
	got := run(t, []int{2, 3}, seq(6), func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		b.ImmediateLimit = 0
		return b.Op("linear", map[string]any{"x": x, "weight": f16s(t, b, []int{2, 3}, w)}, mlpackage.Float16, 2, 2)
	})
	near(t, "linear", got, []int{2, 2}, []float32{-2, 1, -2, 10}, 0)

	// matmul with a broadcast batch axis and transpose_y: (2, 2, 3) x (1, 2, 3)^T
	got = run(t, []int{2, 2, 3}, seq(12), func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		y := f16s(t, b, []int{1, 2, 3}, []float32{1, 0, 0, 0, 1, 1})
		return b.Op("matmul", map[string]any{"x": x, "y": y, "transpose_x": b.ScalarBool(false), "transpose_y": b.ScalarBool(true)}, mlpackage.Float16, 2, 2, 2)
	})
	near(t, "matmul", got, []int{2, 2, 2}, []float32{0, 3, 3, 9, 6, 15, 9, 21}, 0)
	got = run(t, []int{3, 2}, seq(6), func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		y := f16s(t, b, []int{3, 1}, []float32{1, 1, 1})
		return b.Op("matmul", map[string]any{"x": x, "y": y, "transpose_x": b.ScalarBool(true), "transpose_y": b.ScalarBool(false)}, mlpackage.Float16, 2, 1)
	})
	near(t, "matmul transpose_x", got, []int{2, 1}, []float32{6, 9}, 0)
}

func TestLayout(t *testing.T) {
	x := seq(6) // (2, 3)
	got := run(t, []int{2, 3}, x, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		return b.Op("transpose", map[string]any{"x": x, "perm": b.Ints(1, 0)}, mlpackage.Float16, 3, 2)
	})
	near(t, "transpose", got, []int{3, 2}, []float32{0, 3, 1, 4, 2, 5}, 0)
	got = run(t, []int{2, 3}, x, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		return b.Op("reshape", map[string]any{"x": x, "shape": b.Ints(3, -1)}, mlpackage.Float16, 3, 2)
	})
	near(t, "reshape", got, []int{3, 2}, x, 0)
	got = run(t, []int{2, 1}, []float32{1, 2}, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		return b.Op("tile", map[string]any{"x": x, "reps": b.Ints(2, 3)}, mlpackage.Float16, 4, 3)
	})
	near(t, "tile", got, []int{4, 3}, []float32{1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2}, 0)
	got = run(t, []int{2, 3}, x, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		bools, _ := b.Bools("c", []int{2}, []bool{true, false})
		ends, _ := b.Bools("c", []int{2}, []bool{true, true})
		return b.Op("slice_by_index", map[string]any{"x": x, "begin": b.Ints(0, 1), "end": b.Ints(0, 0), "begin_mask": bools, "end_mask": ends}, mlpackage.Float16, 2, 2)
	})
	near(t, "slice", got, []int{2, 2}, []float32{1, 2, 4, 5}, 0)
	got = run(t, []int{2, 3}, x, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		return b.Op("slice_by_index", map[string]any{"x": x, "begin": b.Ints(1, -1), "end": b.Ints(2, -4), "stride": b.Ints(1, -1)}, mlpackage.Float16, 1, 3)
	})
	near(t, "slice reversed", got, []int{1, 3}, []float32{5, 4, 3}, 0)
	got = run(t, []int{2, 3}, x, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		return b.Op("pad", map[string]any{"x": x, "pad": b.Ints(1, 0, 0, 2), "mode": b.Str("c", "constant"), "constant_val": b.ScalarF16(-1)}, mlpackage.Float16, 3, 5)
	})
	near(t, "pad", got, []int{3, 5}, []float32{-1, -1, -1, -1, -1, 0, 1, 2, -1, -1, 3, 4, 5, -1, -1}, 0)
	got = run(t, []int{2, 3}, x, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		y := f16s(t, b, []int{2, 1}, []float32{9, 8})
		return b.Op("concat", map[string]any{"values": []*mlpackage.Var{y, x}, "axis": b.ScalarI32(-1), "interleave": b.ScalarBool(false)}, mlpackage.Float16, 2, 4)
	})
	near(t, "concat", got, []int{2, 4}, []float32{9, 0, 1, 2, 8, 3, 4, 5}, 0)
}

func TestConv(t *testing.T) {
	// depthwise causal convolution of 2 channels with kernel 3, left padded by 2
	x := []float32{1, 2, 3, 4, 10, 20, 30, 40} // (1, 2, 4)
	w := []float32{1, 0, 0, 0, 1, 1}           // channel 0 picks t-2, channel 1 sums t-1 and t
	got := run(t, []int{1, 2, 4}, x, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		bias := f16s(t, b, []int{2}, []float32{0, 0.5})
		return b.Op("conv", map[string]any{"x": x, "weight": f16s(t, b, []int{2, 1, 3}, w), "bias": bias, "strides": b.Ints(1), "pad_type": b.Str("c", "custom"),
			"pad": b.Ints(2, 0), "dilations": b.Ints(1), "groups": b.ScalarI32(2)}, mlpackage.Float16, 1, 2, 4)
	})
	near(t, "conv", got, []int{1, 2, 4}, []float32{0, 0, 1, 2, 10.5, 30.5, 50.5, 70.5}, 0)
}

func TestConstexpr(t *testing.T) {
	// a 4 bit palettized (32, 8) matrix with tables per 16 rows equals its dequantized values
	rows, cols := 32, 8
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(math.Sin(float64(i)))
	}
	q := palette.Quantize(w, rows, cols, 16, 0, 8, palette.Uniform)
	eye := make([]float32, cols*cols)
	for i := 0; i < cols; i++ {
		eye[i*cols+i] = 1
	}
	got := run(t, []int{cols, cols}, eye, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		v, err := b.LUT4("w", []int{rows, cols}, []int{q.Group, q.ColBlock}, q.Indices, q.LUT)
		if err != nil {
			t.Fatal(err)
		}
		return b.Op("linear", map[string]any{"x": x, "weight": v}, mlpackage.Float16, cols, rows)
	})
	var maxErr float64
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			// linear(eye, W) is W transposed
			maxErr = math.Max(maxErr, math.Abs(float64(got.Data[c*rows+r]-w[r*cols+c])))
		}
	}
	if maxErr > 0.1 {
		t.Errorf("lut4: max error %g against the original weights", maxErr)
	}

	// int4 with one scale per row of 4: values -8..7 times the scale
	data := []byte{0x21, 0xF0, 0x87, 0x00} // row 0: 1 2 0 -1, row 1: 7 -8 0 0
	got = run(t, []int{4, 4}, []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
		v, err := b.Int4Scaled("w", []int{2, 4}, []int{1, 4}, data, []uint16{mlpackage.Float16Bits(0.5), mlpackage.Float16Bits(2)})
		if err != nil {
			t.Fatal(err)
		}
		return b.Op("linear", map[string]any{"x": x, "weight": v}, mlpackage.Float16, 4, 2)
	})
	near(t, "int4", got, []int{4, 2}, []float32{0.5, 14, 1, -16, 0, 0, -0.5, 0}, 0)
}

// A package written to disk runs from the directory, and RoundF16 rounds results to fp16.
func TestPackage(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "m.mlpackage")
	pkg, err := mlpackage.Create(dir)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	x := b.Input("x", mlpackage.Float16, 1, 2)
	third := f16s(t, b, nil, []float32{1})
	y := b.NamedOp("real_div", "y", map[string]any{"x": x, "y": b.Op("add", map[string]any{"x": third, "y": b.ScalarF16(2)}, mlpackage.Float16)}, mlpackage.Float16, 1, 2)
	feat := func(v *mlpackage.Var) []mlpackage.Feature {
		return []mlpackage.Feature{{Name: v.Name, DType: mlpackage.Float16, Shape: v.Shape}}
	}
	if err := pkg.Finish(b.Finish(y), 9, feat(x), feat(y)); err != nil {
		t.Fatal(err)
	}
	p, c, err := OpenPackage(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	in := map[string]Tensor{"x": {Shape: []int{1, 2}, Data: []float32{1, 2}}}
	out, err := p.Run(in, Options{})
	if err != nil {
		t.Fatal(err)
	}
	near(t, "fp32", out["y"], []int{1, 2}, []float32{1.0 / 3, 2.0 / 3}, 1e-7)
	out, err = p.Run(in, Options{RoundF16: true})
	if err != nil {
		t.Fatal(err)
	}
	if h := mlpackage.Float16Value(mlpackage.Float16Bits(1.0 / 3)); out["y"].Data[0] != h {
		t.Errorf("RoundF16: %g, want %g", out["y"].Data[0], h)
	}
	if _, err := p.Run(map[string]Tensor{}, Options{}); err == nil {
		t.Error("missing input accepted")
	}
}

// A linear layer of LFM3-1.7B's MLP size (32 tokens, 2048 -> 8192).
func BenchmarkLinear(b *testing.B) {
	x := &value{shape: []int{1, 32, 2048}, f: make([]float32, 32*2048)}
	w := &value{shape: []int{8192, 2048}, f: make([]float32, 8192*2048)}
	for i := range w.f {
		w.f[i] = float32(i%7) - 3
	}
	b.ResetTimer()
	for b.Loop() {
		linear(x, w, nil)
	}
	b.ReportMetric(float64(2*32*2048*8192)*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
}

// 6 and 8 bit tables (uint6 indices packed low bits first, uint8 one per byte) give the values
// palette.Dequantize gives, exactly.
func TestConstexprWideTables(t *testing.T) {
	rows, cols := 32, 12
	w := make([]float32, rows*cols)
	for i := range w {
		w[i] = float32(math.Sin(float64(i) * 0.7))
	}
	eye := make([]float32, cols*cols)
	for i := 0; i < cols; i++ {
		eye[i*cols+i] = 1
	}
	for _, bits := range []int{4, 6, 8} {
		q := palette.QuantizeBits(w, rows, cols, 16, 0, bits, 8, palette.KMeansMinMax)
		want := q.Dequantize()
		got := run(t, []int{cols, cols}, eye, func(b *mlpackage.Builder, x *mlpackage.Var) *mlpackage.Var {
			v, err := b.LUT("w", bits, []int{rows, cols}, []int{q.Group, q.ColBlock}, q.Indices, q.LUT)
			if err != nil {
				t.Fatal(err)
			}
			return b.Op("linear", map[string]any{"x": x, "weight": v}, mlpackage.Float16, cols, rows)
		})
		for r := 0; r < rows; r++ {
			for c := 0; c < cols; c++ {
				if g := got.Data[c*rows+r]; g != want[r*cols+c] {
					t.Fatalf("%d bits: (%d, %d) = %g, want %g", bits, r, c, g, want[r*cols+c])
				}
			}
		}
	}
}
