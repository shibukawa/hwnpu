package ailower

import (
	"encoding/base64"
	"fmt"
	"math"
	"os"
	"slices"

	"github.com/shibukawa/hwnpu/internal/coreai/mlirbc"
	"github.com/shibukawa/hwnpu/internal/milspec"
)

var binaryOps = map[string]string{"add": "add_v1", "sub": "sub_v1", "mul": "mul_v1", "real_div": "divide_v1",
	"maximum": "maximum_v1", "minimum": "minimum_v1", "pow": "pow_v1"}

var unaryOps = map[string]string{"exp": "exp_v1", "tanh": "tanh_v1", "sqrt": "sqrt_v1", "abs": "abs_v1",
	"cos": "cos_v1", "sin": "sin_v1", "erf": "erf_v1", "log": "log_v1"}

// op lowers one MIL operation.
func (l *lowerer) op(op *milspec.Operation) {
	typ := op.GetType()
	if len(op.GetOutputs()) != 1 {
		l.fail("%s: %d outputs", typ, len(op.GetOutputs()))
	}
	out := op.GetOutputs()[0].GetName()
	if typ == "const" {
		l.consts[out] = op.GetAttributes()["val"]
		return
	}
	if typ == "identity" {
		if a := op.GetInputs()["x"].GetArguments(); len(a) == 1 && a[0].GetName() != "" {
			l.alias[out] = a[0].GetName()
			return
		}
	}
	oshape := shapeOf(op.GetOutputs()[0].GetType())
	set := func(v val) {
		if !slices.Equal(v.shape, oshape) {
			l.fail("%s %s: lowered to %v, MIL says %v", typ, out, v.shape, oshape)
		}
		l.vals[out] = v
	}
	switch {
	case typ == "real_div" && l.scalarConst(op, "y") && os.Getenv("HWNPU_AILOWER_DIVIDE") == "":
		// x / c as x times 1 / c: on the Neural Engine, Core AI's divide_v1 by a constant came
		// out as about c x plus an offset a channel in PP-OCR's detector (the exact gelu's x /
		// sqrt 2 after a depthwise 3 x 3 and a 1 x 1: twice the value, 0.05 the detector's
		// cosine), though not on its CPU or GPU, nor in a program of those three operations
		// alone (design.md 16.182, 16.183). HWNPU_AILOWER_DIVIDE=1 keeps divide_v1, for
		// looking into it
		x := l.in(op, "x")
		set(l.binary("mul_v1", x, l.scalar(x.elem, 1/l.float(op, "y", 1)), oshape))
		return
	case binaryOps[typ] != "":
		set(l.binary(binaryOps[typ], l.in(op, "x"), l.in(op, "y"), oshape))
		return
	case unaryOps[typ] != "":
		x := l.in(op, "x")
		set(l.emit(unaryOps[typ], x.elem, x.shape, nil, x))
		return
	}
	switch typ {
	case "identity":
		set(l.in(op, "x"))
	case "rsqrt":
		x := l.in(op, "x")
		if eps := l.float(op, "epsilon", 1e-12); eps != 0 {
			x = l.binary("add_v1", x, l.scalar(x.elem, eps), x.shape)
		}
		set(l.emit("rsqrt_v1", x.elem, x.shape, nil, x))
	case "relu":
		x := l.in(op, "x")
		set(l.binary("maximum_v1", x, l.scalar(x.elem, 0), x.shape))
	case "sigmoid":
		set(l.sigmoid(l.in(op, "x")))
	case "silu":
		x := l.in(op, "x")
		set(l.binary("mul_v1", x, l.sigmoid(x), x.shape))
	case "softplus": // log(1 + exp(x))
		x := l.in(op, "x")
		e := l.emit("exp_v1", x.elem, x.shape, nil, x)
		set(l.emit("log_v1", x.elem, x.shape, nil, l.binary("add_v1", e, l.scalar(x.elem, 1), x.shape)))
	case "elu": // x above 0, alpha (exp(x) - 1) below: max(x, 0) + alpha (exp(min(x, 0)) - 1)
		x := l.in(op, "x")
		zero := l.scalar(x.elem, 0)
		e := l.emit("exp_v1", x.elem, x.shape, nil, l.binary("minimum_v1", x, zero, x.shape))
		neg := l.binary("mul_v1", l.binary("sub_v1", e, l.scalar(x.elem, 1), x.shape), l.scalar(x.elem, l.float(op, "alpha", 1)), x.shape)
		set(l.binary("add_v1", l.binary("maximum_v1", x, zero, x.shape), neg, x.shape))
	case "gelu":
		set(l.gelu(l.in(op, "x"), l.str(op, "mode", "EXACT")))
	case "clip":
		x := l.in(op, "x")
		x = l.binary("maximum_v1", x, l.scalar(x.elem, l.float(op, "alpha", 0)), x.shape)
		set(l.binary("minimum_v1", x, l.scalar(x.elem, l.float(op, "beta", 0)), x.shape))
	case "softmax":
		x := l.in(op, "x")
		set(l.softmax(x, l.axis(int(l.ints(op, "axis", -1)[0]), len(x.shape))))
	case "reduce_mean", "reduce_max", "reduce_sum":
		x := l.in(op, "x")
		axes := l.ints(op, "axes")
		for i := range axes {
			axes[i] = l.axis(axes[i], len(x.shape))
		}
		r := l.reduce(typ, x, axes)
		if !l.boolean(op, "keep_dims") {
			r = l.reshape(r, oshape)
		}
		set(r)
	case "matmul":
		set(l.matmul(l.in(op, "x"), l.in(op, "y"), l.boolean(op, "transpose_x"), l.boolean(op, "transpose_y"), oshape))
	case "linear":
		x := l.in(op, "x")
		w := l.in(op, "weight") // (out, in)
		y := l.matmul(x, w, false, true, oshape)
		if l.has(op, "bias") {
			y = l.binary("add_v1", y, l.in(op, "bias"), oshape)
		}
		set(y)
	case "reshape":
		set(l.reshape(l.in(op, "x"), oshape))
	case "transpose":
		x := l.in(op, "x")
		set(l.transpose(x, l.ints(op, "perm")))
	case "slice_by_index":
		set(l.sliceByIndex(op, oshape))
	case "concat":
		set(l.concat(op, oshape))
	case "tile":
		x := l.in(op, "x")
		reps := l.ints(op, "reps")
		for a, r := range reps {
			if r > 1 {
				parts := make([]val, r)
				for i := range parts {
					parts[i] = x
				}
				x = l.concatVals(a, parts)
			}
		}
		set(x)
	case "upsample_nearest_neighbor":
		x := l.in(op, "x")
		sh, sw := l.ints(op, "scale_factor_height", 1)[0], l.ints(op, "scale_factor_width", 1)[0]
		set(l.upsample(x, sh, sw))
	case "pixel_shuffle":
		x := l.in(op, "x")
		r := l.ints(op, "upscale_factor")[0]
		n, c, h, w := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
		co := c / (r * r)
		y := l.reshape(x, []int{n * co, r, r, h, w})
		y = l.transpose(y, []int{0, 3, 1, 4, 2})
		set(l.reshape(y, []int{n, co, h * r, w * r}))
	case "pad":
		x := l.in(op, "x")
		m := l.str(op, "mode", "constant")
		if m != "constant" && m != "replicate" && m != "reflect" {
			l.fail("pad %s: mode %s", out, m)
		}
		pads := l.ints(op, "pad")
		full := make([]int, 2*len(x.shape))
		copy(full[len(full)-len(pads):], pads)
		set(l.padMode(x, full, l.float(op, "constant_val", 0), m))
	case "layer_norm":
		set(l.layerNorm(op))
	case "cast":
		x := l.in(op, "x")
		dt := l.str(op, "dtype", "")
		elem := map[string]string{"fp16": "float16", "fp32": "float32", "int32": "sint32"}[dt]
		if elem == "" {
			l.fail("cast %s: dtype %s", out, dt)
		}
		if elem == x.elem {
			set(x)
			return
		}
		set(l.emit("cast_v1", elem, x.shape, nil, x))
	case "conv":
		set(l.conv(op, oshape))
	case "max_pool":
		set(l.maxPool(op, oshape))
	case "conv_transpose":
		set(l.convTranspose(op, oshape))
	case "constexpr_lut_to_dense":
		set(l.lutToDense(op, oshape))
	case "constexpr_blockwise_shift_scale":
		set(l.shiftScale(op, oshape))
	default:
		l.fail("operation %s (%s) is not supported", typ, out)
	}
}

// scalarConst reports whether op's input key is a constant float scalar.
func (l *lowerer) scalarConst(op *milspec.Operation, key string) bool {
	a := op.GetInputs()[key]
	if a == nil || len(a.GetArguments()) != 1 {
		return false
	}
	b := a.GetArguments()[0]
	c := b.GetValue()
	if n := b.GetName(); n != "" {
		c = l.consts[l.resolve(n)]
	}
	if c == nil {
		return false
	}
	elem, _, raw := l.decode(c)
	return elem == "float16" && len(raw) == 2 || elem == "float32" && len(raw) == 4
}

func (l *lowerer) axis(a, rank int) int {
	if a < 0 {
		a += rank
	}
	if a < 0 || a >= rank {
		l.fail("axis %d of rank %d", a, rank)
	}
	return a
}

// broadcastTo is x broadcast to shape (numpy rules: aligned at the right).
func (l *lowerer) broadcastTo(x val, shape []int) val {
	if slices.Equal(x.shape, shape) {
		return x
	}
	if len(x.shape) > len(shape) {
		l.fail("broadcast %v to %v", x.shape, shape)
	}
	if len(x.shape) < len(shape) {
		padded := make([]int, len(shape))
		for i := range padded {
			padded[i] = 1
		}
		copy(padded[len(shape)-len(x.shape):], x.shape)
		x = l.reshape(x, padded)
		if slices.Equal(x.shape, shape) {
			return x
		}
	}
	var sizes, axes []int
	for i, d := range shape {
		switch x.shape[i] {
		case d:
		case 1:
			sizes, axes = append(sizes, d), append(axes, i)
		default:
			l.fail("broadcast %v to %v", x.shape, shape)
		}
	}
	return l.emit("broadcast_in_dims_v1", x.elem, shape, nil, x, l.u32s(sizes...), l.i32s(axes...))
}

// binary is a primitive binary operation of x and y broadcast to shape.
func (l *lowerer) binary(name string, x, y val, shape []int) val {
	if x.elem != y.elem {
		l.fail("%s of %s and %s", name, x.elem, y.elem)
	}
	return l.emit(name, x.elem, shape, nil, l.broadcastTo(x, shape), l.broadcastTo(y, shape))
}

func (l *lowerer) reshape(x val, shape []int) val {
	if slices.Equal(x.shape, shape) {
		return x
	}
	return l.emit("reshape_v2", x.elem, shape, nil, x, l.u32s(shape...))
}

func (l *lowerer) transpose(x val, perm []int) val {
	shape := make([]int, len(perm))
	for i, p := range perm {
		shape[i] = x.shape[l.axis(p, len(perm))]
	}
	return l.emit("transpose_v1", x.elem, shape, nil, x, l.u32s(perm...))
}

// sigmoid is 1 / (1 + exp(-x)), as coreai-core writes it.
func (l *lowerer) sigmoid(x val) val {
	n := l.binary("sub_v1", l.scalar(x.elem, 0), x, x.shape)
	e := l.emit("exp_v1", x.elem, x.shape, nil, n)
	d := l.binary("add_v1", l.scalar(x.elem, 1), e, x.shape)
	return l.binary("divide_v1", l.scalar(x.elem, 1), d, x.shape)
}

func (l *lowerer) gelu(x val, mode string) val {
	half := l.binary("mul_v1", x, l.scalar(x.elem, 0.5), x.shape)
	switch mode {
	case "EXACT": // 0.5 x (1 + erf(x / sqrt 2))
		e := l.emit("erf_v1", x.elem, x.shape, nil, l.binary("mul_v1", x, l.scalar(x.elem, float32(1/math.Sqrt2)), x.shape))
		return l.binary("mul_v1", half, l.binary("add_v1", e, l.scalar(x.elem, 1), x.shape), x.shape)
	case "TANH_APPROXIMATION": // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
		x3 := l.binary("mul_v1", l.binary("mul_v1", x, x, x.shape), x, x.shape)
		in := l.binary("add_v1", x, l.binary("mul_v1", x3, l.scalar(x.elem, 0.044715), x.shape), x.shape)
		t := l.emit("tanh_v1", x.elem, x.shape, nil, l.binary("mul_v1", in, l.scalar(x.elem, float32(math.Sqrt(2/math.Pi))), x.shape))
		return l.binary("mul_v1", half, l.binary("add_v1", t, l.scalar(x.elem, 1), x.shape), x.shape)
	case "SIGMOID_APPROXIMATION": // x sigmoid(1.702 x)
		return l.binary("mul_v1", x, l.sigmoid(l.binary("mul_v1", x, l.scalar(x.elem, 1.702), x.shape)), x.shape)
	}
	l.fail("gelu mode %s", mode)
	return val{}
}

// softmax over axis: exp(x - max) times the reciprocal of its sum (not divided by the sum: Core AI
// would take matmul, that, matmul for an attention it runs off the Neural Engine).
func (l *lowerer) softmax(x val, axis int) val {
	m := l.reduce("reduce_max", x, []int{axis})
	e := l.emit("exp_v1", x.elem, x.shape, nil, l.binary("sub_v1", x, m, x.shape))
	s := l.reduce("reduce_sum", e, []int{axis})
	r := l.binary("divide_v1", l.scalar(x.elem, 1), s, s.shape)
	return l.binary("mul_v1", e, r, x.shape)
}

// reduce is the reduction over axes, the dimensions kept (as size 1), written as coreai-core
// writes it: the composite reduce_max_v1 or reduce_sum_v1 holding the primitive reduce_v1 (whose
// result has dynamic dimensions; Core AI does not legalize a primitive reduce_v1 of its own).
// Sums are taken in fp32 (as coreai-core's reduce_sum), the mean is the sum times 1/n.
func (l *lowerer) reduce(typ string, x val, axes []int) val {
	shape := slices.Clone(x.shape)
	n := 1
	for _, a := range axes {
		n *= shape[a]
		shape[a] = 1
	}
	if x.elem != "float16" {
		l.fail("%s of %s", typ, x.elem)
	}
	st := func(e string) int { return l.ttype(e, nil) }
	dyn := make([]string, len(x.shape))
	for i := range dyn {
		dyn[i] = "-9223372036854775808"
	}
	dynType := l.w.Type("aicode", fmt.Sprintf("!aicode.tensor_v1<[%s] : !aicode.%s_v1>", joinStr(dyn), x.elem))
	ax := l.i32s(axes...)

	outer, oa := l.w.NewBlock(l.ttype(x.elem, x.shape), ax.v.Type())
	var body *mlirbc.BlockW
	var init mlirbc.Value
	name := "aicode.reduce_max_v1"
	switch typ {
	case "reduce_max":
		b := make([]byte, 2)
		b[0], b[1] = 0xff, 0xfb // fp16 -65504
		init = outer.Op("aicode.primitive.constant_v1", []mlirbc.Prop{l.prop(denseText(b))}, nil, []int{st("float16")})[0]
		var args []mlirbc.Value
		body, args = l.w.NewBlock(st("float16"), st("float16"))
		m := body.Op("aicode.primitive.maximum_v1", nil, []mlirbc.Value{args[0], args[1]}, []int{st("float16")})[0]
		body.Op("aicode.dataflow.yield_v1", nil, []mlirbc.Value{m}, nil)
	case "reduce_sum", "reduce_mean":
		name = "aicode.reduce_sum_v1"
		init = outer.Op("aicode.primitive.constant_v1", []mlirbc.Prop{l.prop(denseText([]byte{0, 0, 0, 0}))}, nil, []int{st("float32")})[0]
		var args []mlirbc.Value
		body, args = l.w.NewBlock(st("float32"), st("float16"))
		e := body.Op("aicode.primitive.cast_v1", nil, []mlirbc.Value{args[1]}, []int{st("float32")})[0]
		a := body.Op("aicode.primitive.add_v1", nil, []mlirbc.Value{args[0], e}, []int{st("float32")})[0]
		body.Op("aicode.dataflow.yield_v1", nil, []mlirbc.Value{a}, nil)
	default:
		l.fail("reduction %s", typ)
	}
	red := outer.Op("aicode.primitive.reduce_v1", nil, []mlirbc.Value{oa[0], oa[1], init}, []int{dynType}, body)[0]
	outer.Op("aicode.dataflow.yield_v1", nil, []mlirbc.Value{red}, nil)
	res := l.g.Op(name, nil, []mlirbc.Value{x.v, ax.v}, []int{l.ttype(x.elem, shape)}, outer)[0]
	r := val{v: res, elem: x.elem, shape: shape}
	if typ == "reduce_mean" {
		r = l.binary("mul_v1", r, l.scalar(x.elem, float32(1/float64(n))), shape)
	}
	return r
}

func joinStr(s []string) string {
	out := ""
	for i, x := range s {
		if i > 0 {
			out += ", "
		}
		out += x
	}
	return out
}

func denseText(raw []byte) string {
	return fmt.Sprintf("#aicode.serialization.dense_elements<[splat = true] %q>", base64.StdEncoding.EncodeToString(raw))
}

// matmul is x times y (either transposed first), the batch dimensions broadcast.
func (l *lowerer) matmul(x, y val, tx, ty bool, oshape []int) val {
	swap := func(v val) val {
		r := len(v.shape)
		perm := make([]int, r)
		for i := range perm {
			perm[i] = i
		}
		perm[r-2], perm[r-1] = r-1, r-2
		return l.transpose(v, perm)
	}
	if tx {
		x = swap(x)
	}
	if ty {
		y = swap(y)
	}
	r := len(oshape)
	batch := oshape[:r-2]
	// as coreai-core writes it: the composite broadcasting_batch_matmul_v1, whose body broadcasts
	// the operands to the batch dimensions (broadcast_to) and multiplies. Written as primitives at
	// the top, a quantized weight (blockwise_shift_scale) kept the whole program off the Neural
	// Engine, and a reshape of the weight made the MPSGraph lowering infer a wrong shape.
	body, args := l.w.NewBlock(l.ttype(x.elem, x.shape), l.ttype(y.elem, y.shape))
	bx, by := val{v: args[0], elem: x.elem, shape: x.shape}, val{v: args[1], elem: y.elem, shape: y.shape}
	saved := l.g
	l.g = body
	expand := func(v val) val {
		want := append(slices.Clone(batch), v.shape[len(v.shape)-2:]...)
		if slices.Equal(v.shape, want) {
			return v
		}
		return l.emit("broadcast_to_v1", v.elem, want, nil, v, l.u32s(want...))
	}
	m := l.emit("batch_matmul_v1", x.elem, oshape, nil, expand(bx), expand(by))
	body.Op("aicode.dataflow.yield_v1", nil, []mlirbc.Value{m.v}, nil)
	l.g = saved
	res := l.g.Op("aicode.broadcasting_batch_matmul_v1", nil, []mlirbc.Value{x.v, y.v}, []int{l.ttype(x.elem, oshape)}, body)[0]
	return val{v: res, elem: x.elem, shape: slices.Clone(oshape)}
}

func (l *lowerer) sliceByIndex(op *milspec.Operation, oshape []int) val {
	x := l.in(op, "x")
	r := len(x.shape)
	begin := l.ints(op, "begin")
	end := l.ints(op, "end")
	stride := l.ints(op, "stride", ones(r)...)
	bm := l.bools(op, "begin_mask", r)
	em := l.bools(op, "end_mask", r)
	sq := l.bools(op, "squeeze_mask", r)
	shape := make([]int, r)
	for i := 0; i < r; i++ {
		d := x.shape[i]
		b, e := begin[i], end[i]
		if bm[i] {
			b = 0
		}
		if em[i] {
			e = d
		}
		if b < 0 {
			b += d
		}
		if e < 0 {
			e += d
		}
		if sq[i] {
			e = b + 1
		}
		b, e = min(max(b, 0), d), min(max(e, 0), d)
		if stride[i] <= 0 {
			l.fail("slice %s: stride %d", opName(op), stride[i])
		}
		begin[i], end[i] = b, e
		shape[i] = (e - b + stride[i] - 1) / stride[i]
	}
	y := l.emit("slice_v1", x.elem, shape, nil, x, l.i32s(begin...), l.i32s(end...), l.i32s(stride...))
	return l.reshape(y, oshape)
}

func ones(n int) []int {
	o := make([]int, n)
	for i := range o {
		o[i] = 1
	}
	return o
}

func (l *lowerer) concat(op *milspec.Operation, oshape []int) val {
	if l.boolean(op, "interleave") {
		l.fail("concat %s: interleave", opName(op))
	}
	var parts []val
	for _, a := range op.GetInputs()["values"].GetArguments() {
		parts = append(parts, l.binding(a))
	}
	return l.concatVals(l.axis(l.ints(op, "axis")[0], len(oshape)), parts)
}

func (l *lowerer) concatVals(axis int, parts []val) val {
	shape := slices.Clone(parts[0].shape)
	shape[axis] = 0
	ops := []val{l.i32(axis)}
	for _, p := range parts {
		shape[axis] += p.shape[axis]
		ops = append(ops, p)
	}
	return l.emit("concat_v1", parts[0].elem, shape, nil, ops...)
}

func (l *lowerer) upsample(x val, sh, sw int) val {
	n, c, h, w := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	shape := []int{n, c, h * sh, w * sw}
	scales := make([]byte, 16)
	for i, s := range []float32{1, 1, float32(sh), float32(sw)} {
		putF32(scales[4*i:], s)
	}
	return l.emit("interpolate_v1", x.elem, shape, []mlirbc.Prop{l.prop("#aicode.interpolation_mode_v1<nearest_neighbor>")},
		x, l.i32s(shape...), l.constRaw("float32", []int{4}, scales), l.constRaw("float32", []int{4}, make([]byte, 16)))
}

func putF32(b []byte, f float32) {
	u := math.Float32bits(f)
	b[0], b[1], b[2], b[3] = byte(u), byte(u>>8), byte(u>>16), byte(u>>24)
}

// pad pads x by pads (before and after of each axis), with v.
func (l *lowerer) pad(x val, pads []int, v float32) val { return l.padMode(x, pads, v, "constant") }

// padMode pads x by pads in mode: constant (v), replicate (the border's element) or reflect
// (mirrored about the border, MIL's and Core AI's alike).
func (l *lowerer) padMode(x val, pads []int, v float32, mode string) val {
	shape := slices.Clone(x.shape)
	any := false
	for i := range shape {
		shape[i] += pads[2*i] + pads[2*i+1]
		any = any || pads[2*i] != 0 || pads[2*i+1] != 0
	}
	if !any {
		return x
	}
	return l.emit("pad_v1", x.elem, shape, []mlirbc.Prop{l.prop("#aicode.padding_mode_v1<" + mode + ">")}, x, l.u32s(pads...), l.scalar(x.elem, v))
}

// layerNorm is (x - mean) / sqrt(var + eps) gamma + beta over axes.
func (l *lowerer) layerNorm(op *milspec.Operation) val {
	x := l.in(op, "x")
	axes := l.ints(op, "axes")
	for i := range axes {
		axes[i] = l.axis(axes[i], len(x.shape))
	}
	eps := l.float(op, "epsilon", 1e-5)
	m := l.reduce("reduce_mean", x, axes)
	d := l.binary("sub_v1", x, m, x.shape)
	v := l.reduce("reduce_mean", l.binary("mul_v1", d, d, x.shape), axes)
	v = l.binary("add_v1", v, l.scalar(x.elem, eps), v.shape)
	y := l.binary("mul_v1", d, l.emit("rsqrt_v1", x.elem, v.shape, nil, v), x.shape)
	if l.has(op, "gamma") {
		y = l.binary("mul_v1", y, l.in(op, "gamma"), x.shape)
	}
	if l.has(op, "beta") {
		y = l.binary("add_v1", y, l.in(op, "beta"), x.shape)
	}
	return y
}

func (l *lowerer) conv(op *milspec.Operation, oshape []int) val {
	x := l.in(op, "x")
	w := l.in(op, "weight")
	if len(x.shape) != 4 {
		l.fail("conv %s: rank %d (2 D only)", opName(op), len(x.shape))
	}
	strides := l.ints(op, "strides", 1, 1)
	dil := l.ints(op, "dilations", 1, 1)
	groups := l.ints(op, "groups", 1)[0]
	pads := l.spatialPads(op, x, []int{w.shape[2], w.shape[3]}, strides, dil)
	x = l.pad(x, []int{0, 0, 0, 0, pads[0], pads[1], pads[2], pads[3]}, 0)
	y := l.emit("conv2d_v1", x.elem, oshape, nil, x, w, l.u32s(strides...), l.u32s(dil...), l.u32(groups))
	if l.has(op, "bias") {
		b := l.in(op, "bias")
		y = l.binary("add_v1", y, l.reshape(b, []int{1, b.shape[0], 1, 1}), oshape)
	}
	return y
}

// convTranspose is a 2 D transposed convolution: conv_transpose2d_v1 (x, the weight (Cin,
// Cout / groups, kh, kw) as MIL's, the strides, its padding, the dilations, output_pad, groups)
// without padding, then MIL's pads (top, bottom, left, right; negative past the end) as a crop
// to the output's shape (zeros where it reaches past the full output), then the bias.
func (l *lowerer) convTranspose(op *milspec.Operation, oshape []int) val {
	x := l.in(op, "x")
	w := l.in(op, "weight")
	if len(x.shape) != 4 || len(oshape) != 4 {
		l.fail("conv_transpose %s: rank %d (2 D only)", opName(op), len(x.shape))
	}
	strides := l.ints(op, "strides", 1, 1)
	dil := l.ints(op, "dilations", 1, 1)
	groups := l.ints(op, "groups", 1)[0]
	pads := []int{0, 0, 0, 0}
	switch t := l.str(op, "pad_type", "valid"); t {
	case "valid":
	case "custom":
		pads = l.ints(op, "pad", 0, 0, 0, 0)
	default:
		l.fail("conv_transpose %s: pad_type %s", opName(op), t)
	}
	full := []int{x.shape[0], w.shape[1] * groups, 0, 0}
	for i := range 2 {
		full[2+i] = (x.shape[2+i]-1)*strides[i] + dil[i]*(w.shape[2+i]-1) + 1
	}
	y := l.emit("conv_transpose2d_v1", x.elem, full, nil, x, w, l.u32s(strides...), l.u32s(0, 0), l.u32s(dil...), l.u32s(0, 0), l.u32(groups))
	// the output's rows [top, top + H) of the full output's: a pad before a negative top, a
	// slice of what lies inside, a pad past the full output's end
	before, after := make([]int, 8), make([]int, 8)
	begin, end := []int{0, 0, 0, 0}, slices.Clone(full)
	for i := range 2 {
		lo, hi := pads[2*i], pads[2*i]+oshape[2+i]
		if lo < 0 {
			before[2*(2+i)] = -lo
			lo = 0
		}
		if hi > full[2+i] {
			after[2*(2+i)+1] = hi - full[2+i]
			hi = full[2+i]
		}
		begin[2+i], end[2+i] = lo, hi
	}
	if !slices.Equal(begin, []int{0, 0, 0, 0}) || !slices.Equal(end, full) {
		shape := slices.Clone(full)
		for i := range 2 {
			shape[2+i] = end[2+i] - begin[2+i]
		}
		y = l.emit("slice_v1", y.elem, shape, nil, y, l.i32s(begin...), l.i32s(end...), l.i32s(1, 1, 1, 1))
	}
	y = l.pad(l.pad(y, before, 0), after, 0)
	if l.has(op, "bias") {
		b := l.in(op, "bias")
		y = l.binary("add_v1", y, l.reshape(b, []int{1, b.shape[0], 1, 1}), oshape)
	}
	if !slices.Equal(y.shape, oshape) {
		l.fail("conv_transpose %s: lowered to %v, MIL says %v", opName(op), y.shape, oshape)
	}
	return y
}

// spatialPads are a 2 D window's pads (top, bottom, left, right) by MIL's pad_type, for the
// kernel k, strides and dilations of a conv or a pool of x (N, C, H, W).
func (l *lowerer) spatialPads(op *milspec.Operation, x val, k, strides, dil []int) []int {
	switch t := l.str(op, "pad_type", "valid"); t {
	case "valid":
		return []int{0, 0, 0, 0}
	case "custom":
		return l.ints(op, "pad", 0, 0, 0, 0)
	case "same", "same_lower":
		pads := make([]int, 4)
		for i := range 2 {
			in := x.shape[2+i]
			outd := (in + strides[i] - 1) / strides[i]
			total := max((outd-1)*strides[i]+(k[i]-1)*dil[i]+1-in, 0)
			lo := total / 2
			if t == "same_lower" {
				lo = total - total/2
			}
			pads[2*i], pads[2*i+1] = lo, total-lo
		}
		return pads
	default:
		l.fail("%s %s: pad_type %s", op.GetType(), opName(op), t)
	}
	return nil
}

// maxPool is a 2 D max pool, as coreai-torch writes it: the padding by pad_v1 with -inf (the
// pool itself takes none), then max_pool_2d_v1 (x, the kernel, the strides, the dilations,
// ceil_mode).
func (l *lowerer) maxPool(op *milspec.Operation, oshape []int) val {
	x := l.in(op, "x")
	if len(x.shape) != 4 {
		l.fail("max_pool %s: rank %d (2 D only)", opName(op), len(x.shape))
	}
	k := l.ints(op, "kernel_sizes")
	strides := l.ints(op, "strides", 1, 1)
	if len(k) != 2 || len(strides) != 2 {
		l.fail("max_pool %s: kernel %v, strides %v", opName(op), k, strides)
	}
	pads := l.spatialPads(op, x, k, strides, []int{1, 1})
	x = l.pad(x, []int{0, 0, 0, 0, pads[0], pads[1], pads[2], pads[3]}, float32(math.Inf(-1)))
	ceil := byte(0)
	if l.boolean(op, "ceil_mode") {
		ceil = 1
	}
	return l.emit("max_pool_2d_v1", x.elem, oshape, nil, x, l.u32s(k...), l.u32s(strides...), l.u32s(1, 1), l.constRaw("bool", nil, []byte{ceil}))
}

// lutToDense is the lookup table weight: indices (rows x cols, packed) and the tables (shape
// divided by the blocks, then the entries and the vector size 1), as aicode's lut_to_dense.
func (l *lowerer) lutToDense(op *milspec.Operation, oshape []int) val {
	idx := l.in(op, "indices")
	lut := l.in(op, "lut")
	if lut.shape[len(lut.shape)-1] != 1 {
		l.fail("lut %s: vector palettization", opName(op))
	}
	if !slices.Equal(idx.shape, oshape) {
		l.fail("lut %s: indices %v for %v", opName(op), idx.shape, oshape)
	}
	return l.emit("lut_to_dense_v1", lut.elem, oshape, nil, idx, lut, l.constRaw("sint16", nil, []byte{0, 0}))
}

// shiftScale is the blockwise linearly quantized weight: data (int8 or int4) times its block's
// scale, as aicode's blockwise_shift_scale_v2 (data, scale, zero point, minimum).
func (l *lowerer) shiftScale(op *milspec.Operation, oshape []int) val {
	data := l.in(op, "data")
	scale := l.in(op, "scale")
	if l.has(op, "offset") {
		l.fail("blockwise %s: an offset (only symmetric quantization)", opName(op))
	}
	if len(scale.shape) != len(data.shape) {
		l.fail("blockwise %s: scale %v for data %v", opName(op), scale.shape, data.shape)
	}
	n := 1
	for _, d := range scale.shape {
		n *= d
	}
	zb := 1
	if data.elem == "sint4" {
		zb = 1 // one byte a zero point either way (splat)
	}
	zero := l.constRaw(data.elem, scale.shape, make([]byte, n*zb))
	minv := l.constRaw(scale.elem, scale.shape, make([]byte, 2*n))
	return l.emit("blockwise_shift_scale_v2", scale.elem, oshape, nil, data, scale, zero, minv)
}

var _ = fmt.Sprint
