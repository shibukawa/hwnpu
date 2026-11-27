// Package experts is the mixture of experts layers of a model cut at them (llm.Routed): the
// program that precedes such a layer outputs its router's logits, the host picks the layer's top
// k experts from them, and a program without weights runs the k chosen ones on one row, their
// weights given at run time from arrays the host holds for every expert of every layer
// (docs/design/apple-neural-engine.md). On an M3's Neural Engine with Core AI, LFM2-8B-A1B's four
// experts of 2048 x 1792 take 2.7 to 3.0 ms a row this way, every layer's 7.75 GB held.
//
// An expert is a SwiGLU: down(silu(gate x) * up x). Its weights are int8, a scale a row, as two
// arrays of the host: GateUp (hidden, 2 inter), the gate's and the up projection's rows as
// columns side by side, and Down (hidden, inter), the down projection as the checkpoint has it.
// The scales of a layer's experts are one table (experts, 2 inter + hidden), from which the
// program picks the chosen experts' rows with gather_nd on the Neural Engine (its gather takes
// int16 indices and runs on an M3; on an M1 it does not, docs/design/apple-neural-engine.md).
//
// The program's inputs are x (1, 1, hidden), the row the layer's feed forward takes; i (k, 1),
// the chosen experts' numbers, fp16 (cast to int16 inside: Core ML's model description has no
// int16 input); S (experts, 2 inter + hidden), the layer's scales; gu0.. and dn0.., each chosen
// expert's arrays (int8); r (1, 1, k), their routing weights. Its output y (1, 1, hidden) is the
// weighted sum of their outputs.
package experts

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/milspec"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Shape is the experts' sizes.
type Shape struct {
	Hidden, Inter, Experts, TopK int
}

// ScaleRow is the length of an expert's row of the scales table: the gate's and the up
// projection's (2 inter), then the down projection's (hidden).
func (s Shape) ScaleRow() int { return 2*s.Inter + s.Hidden }

// Build writes the program of k chosen experts on one row into b.
func Build(b *mlpackage.Builder, sh Shape) (ins, outs []mlpackage.Feature, err error) {
	H, I, E, K := sh.Hidden, sh.Inter, sh.Experts, sh.TopK
	if H < 1 || I < 1 || K < 1 || E < K || E > 32767 {
		return nil, nil, fmt.Errorf("experts: %d of %d experts of %d x %d", K, E, H, I)
	}
	S := sh.ScaleRow()
	f16 := func(typ string, in map[string]any, shape ...int) *mlpackage.Var {
		return b.Op(typ, in, mlpackage.Float16, shape...)
	}
	input := func(name string, dt milspec.DataType, shape ...int) *mlpackage.Var {
		ins = append(ins, mlpackage.Feature{Name: name, DType: dt, Shape: shape})
		return b.Input(name, dt, shape...)
	}
	slice := func(x *mlpackage.Var, axis, lo, hi int) *mlpackage.Var {
		begin, end := make([]int32, len(x.Shape)), make([]int32, len(x.Shape))
		out := append([]int(nil), x.Shape...)
		for i, d := range x.Shape {
			end[i] = int32(d)
		}
		begin[axis], end[axis], out[axis] = int32(lo), int32(hi), hi-lo
		return f16("slice_by_index", map[string]any{"x": x, "begin": b.Ints(begin...), "end": b.Ints(end...)}, out...)
	}
	mul := func(x, y *mlpackage.Var) *mlpackage.Var {
		return f16("mul", map[string]any{"x": x, "y": y}, x.Shape...)
	}
	matmul := func(x, y *mlpackage.Var, ty bool, shape ...int) *mlpackage.Var {
		return f16("matmul", map[string]any{"x": x, "y": y, "transpose_x": b.ScalarBool(false), "transpose_y": b.ScalarBool(ty)}, shape...)
	}
	cast := func(x *mlpackage.Var) *mlpackage.Var {
		return f16("cast", map[string]any{"x": x, "dtype": b.Str("c", "fp16")}, x.Shape...)
	}
	// silu as x / (1 + exp(-x)): the Neural Engine's silu (and x times its sigmoid) is about 1.4%
	// off (blocks.SiLU)
	silu := func(x *mlpackage.Var) *mlpackage.Var {
		neg := mul(x, b.ScalarF16(-1))
		den := f16("add", map[string]any{"x": f16("exp", map[string]any{"x": neg}, x.Shape...), "y": b.ScalarF16(1)}, x.Shape...)
		return f16("real_div", map[string]any{"x": x, "y": den}, x.Shape...)
	}

	x := input("x", mlpackage.Float16, 1, 1, H)
	idx := input("i", mlpackage.Float16, K, 1)
	tbl := input("S", mlpackage.Float16, E, S)
	r := input("r", mlpackage.Float16, 1, 1, K)
	i16 := b.Op("cast", map[string]any{"x": idx, "dtype": b.Str("c", "int16")}, mlpackage.Int16, K, 1)
	sc := f16("gather_nd", map[string]any{"x": tbl, "indices": i16, "batch_dims": b.ScalarI32(0), "validate_indices": b.ScalarBool(false)}, K, S)
	var acc *mlpackage.Var
	for k := 0; k < K; k++ {
		gu := input(fmt.Sprintf("gu%d", k), mlpackage.Int8, 1, H, 2*I)
		dn := input(fmt.Sprintf("dn%d", k), mlpackage.Int8, 1, H, I)
		s := f16("reshape", map[string]any{"x": slice(sc, 0, k, k+1), "shape": b.Ints(1, 1, int32(S))}, 1, 1, S)
		h := mul(matmul(x, cast(gu), false, 1, 1, 2*I), slice(s, 2, 0, 2*I))
		a := mul(silu(slice(h, 2, 0, I)), slice(h, 2, I, 2*I))
		d := mul(matmul(a, cast(dn), true, 1, 1, H), slice(s, 2, 2*I, S))
		y := mul(d, slice(r, 2, k, k+1))
		if acc == nil {
			acc = y
		} else {
			acc = f16("add", map[string]any{"x": acc, "y": y}, 1, 1, H)
		}
	}
	y := b.NamedOp("identity", "y", map[string]any{"x": acc}, mlpackage.Float16, 1, 1, H)
	return ins, []mlpackage.Feature{{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}}, nil
}

// BuildRows writes the program of one expert on rows rows (a call that reads a prompt): the host
// runs it for each expert any of the rows chose, their routing weights r (1, rows, 1) 0 for the
// rows that did not, and the output of each call is the next call's acc, so that the last call's
// y (1, rows, hidden) is the layer's experts' output for every row. Its inputs are x (1, rows,
// hidden), i (1, 1) the expert's number (fp16), S the layer's scales (as Build's), gu and dn the
// expert's arrays, r and acc (1, rows, hidden). An expert reads its weights once for all the
// rows, where Build reads them for each.
func BuildRows(b *mlpackage.Builder, sh Shape, rows int) (ins, outs []mlpackage.Feature, err error) {
	H, I, E := sh.Hidden, sh.Inter, sh.Experts
	if H < 1 || I < 1 || E < 1 || E > 32767 || rows < 1 {
		return nil, nil, fmt.Errorf("experts: %d rows of an expert of %d x %d", rows, H, I)
	}
	S := sh.ScaleRow()
	f16 := func(typ string, in map[string]any, shape ...int) *mlpackage.Var {
		return b.Op(typ, in, mlpackage.Float16, shape...)
	}
	input := func(name string, dt milspec.DataType, shape ...int) *mlpackage.Var {
		ins = append(ins, mlpackage.Feature{Name: name, DType: dt, Shape: shape})
		return b.Input(name, dt, shape...)
	}
	slice := func(x *mlpackage.Var, axis, lo, hi int) *mlpackage.Var {
		begin, end := make([]int32, len(x.Shape)), make([]int32, len(x.Shape))
		out := append([]int(nil), x.Shape...)
		for i, d := range x.Shape {
			end[i] = int32(d)
		}
		begin[axis], end[axis], out[axis] = int32(lo), int32(hi), hi-lo
		return f16("slice_by_index", map[string]any{"x": x, "begin": b.Ints(begin...), "end": b.Ints(end...)}, out...)
	}
	mul := func(x, y *mlpackage.Var, shape ...int) *mlpackage.Var {
		if shape == nil {
			shape = x.Shape
		}
		return f16("mul", map[string]any{"x": x, "y": y}, shape...)
	}
	matmul := func(x, y *mlpackage.Var, ty bool, shape ...int) *mlpackage.Var {
		return f16("matmul", map[string]any{"x": x, "y": y, "transpose_x": b.ScalarBool(false), "transpose_y": b.ScalarBool(ty)}, shape...)
	}
	cast := func(x *mlpackage.Var) *mlpackage.Var {
		return f16("cast", map[string]any{"x": x, "dtype": b.Str("c", "fp16")}, x.Shape...)
	}
	silu := func(x *mlpackage.Var) *mlpackage.Var {
		neg := mul(x, b.ScalarF16(-1))
		den := f16("add", map[string]any{"x": f16("exp", map[string]any{"x": neg}, x.Shape...), "y": b.ScalarF16(1)}, x.Shape...)
		return f16("real_div", map[string]any{"x": x, "y": den}, x.Shape...)
	}
	x := input("x", mlpackage.Float16, 1, rows, H)
	idx := input("i", mlpackage.Float16, 1, 1)
	tbl := input("S", mlpackage.Float16, E, S)
	gu := input("gu", mlpackage.Int8, 1, H, 2*I)
	dn := input("dn", mlpackage.Int8, 1, H, I)
	r := input("r", mlpackage.Float16, 1, rows, 1)
	acc := input("acc", mlpackage.Float16, 1, rows, H)
	i16 := b.Op("cast", map[string]any{"x": idx, "dtype": b.Str("c", "int16")}, mlpackage.Int16, 1, 1)
	sc := f16("gather_nd", map[string]any{"x": tbl, "indices": i16, "batch_dims": b.ScalarI32(0), "validate_indices": b.ScalarBool(false)}, 1, S)
	s := f16("reshape", map[string]any{"x": sc, "shape": b.Ints(1, 1, int32(S))}, 1, 1, S)
	h := mul(matmul(x, cast(gu), false, 1, rows, 2*I), slice(s, 2, 0, 2*I), 1, rows, 2*I)
	a := mul(silu(slice(h, 2, 0, I)), slice(h, 2, I, 2*I), 1, rows, I)
	d := mul(matmul(a, cast(dn), true, 1, rows, H), slice(s, 2, 2*I, S), 1, rows, H)
	sum := f16("add", map[string]any{"x": acc, "y": mul(d, r, 1, rows, H)}, 1, rows, H)
	y := b.NamedOp("identity", "y", map[string]any{"x": sum}, mlpackage.Float16, 1, rows, H)
	return ins, []mlpackage.Feature{{Name: y.Name, DType: mlpackage.Float16, Shape: y.Shape}}, nil
}
