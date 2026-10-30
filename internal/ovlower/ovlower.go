// Package ovlower writes a MIL program built with internal/mlpackage as OpenVINO IR, the input of
// Intel's NPU compiler. The architectures of internal/llm describe their models once, as MIL;
// this package translates each MIL operation they use into one or a few OpenVINO operations, so
// the same description serves the Neural Engine and Intel's NPU (docs/intel-npu-models.md).
//
// The translation is strict: an operation, an input or a setting it does not know stops it with
// an error rather than being dropped. Weights must be fp16 or linearly quantized
// (constexpr_blockwise_shift_scale, the intN formats): Intel's NPU has no lookup table weights.
package ovlower

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"maps"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/milspec"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/ovir"
)

// Lower writes the MIL function fn as OpenVINO IR named name and returns the xml. weights is
// the MIL weight file (weight.bin) that fn's blob values point into; the IR's constants are
// written to bin as they are reached. meta goes into the model's rt_info.
//
// The IR's inputs and outputs keep the MIL names: an output's producing layer is named after it,
// since that name is what Intel's NPU driver reports for the output.
func Lower(name string, fn *milspec.Function, weights io.ReaderAt, bin io.Writer, meta map[string]string) ([]byte, error) {
	block := fn.GetBlockSpecializations()[fn.GetOpset()]
	if block == nil {
		return nil, fmt.Errorf("ovlower: the function has no block for opset %q", fn.GetOpset())
	}
	l := &lowerer{
		b:       ovir.NewStreamBuilder(name, bin),
		weights: weights,
		vals:    map[string]ovir.Value{},
		consts:  map[string]*milspec.Value{},
		alias:   map[string]string{},
		rename:  map[string]string{},
	}
	if err := l.lower(fn, block); err != nil {
		return nil, err
	}
	for _, k := range slices.Sorted(maps.Keys(meta)) {
		l.b.RTInfo(rtKey(k), meta[k])
	}
	xml, _ := l.b.Bytes()
	if err := l.b.Err(); err != nil {
		return nil, err
	}
	return xml, nil
}

type lowerer struct {
	b       *ovir.Builder
	weights io.ReaderAt
	vals    map[string]ovir.Value     // MIL values lowered so far
	consts  map[string]*milspec.Value // MIL const operations, lowered when used as a tensor
	alias   map[string]string         // identity outputs: the value they pass on
	rename  map[string]string         // MIL values whose layer takes an output's name
}

// lowerError carries an error out of the operation code, as blocks.Fail does.
type lowerError struct{ err error }

func (l *lowerer) fail(format string, args ...any) {
	panic(lowerError{fmt.Errorf("ovlower: "+format, args...)})
}

func (l *lowerer) lower(fn *milspec.Function, block *milspec.Block) (err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(lowerError); ok {
				err = e.err
				return
			}
			panic(r)
		}
	}()
	producer := map[string]*milspec.Operation{}
	for _, op := range block.GetOperations() {
		for _, o := range op.GetOutputs() {
			producer[o.GetName()] = op
		}
		if op.GetType() == "identity" {
			if n := op.GetInputs()["x"].GetArguments()[0].GetName(); n != "" {
				l.alias[op.GetOutputs()[0].GetName()] = n
			}
		}
	}
	// An output takes over the name of the layer that computes it, unless that is an input, a
	// constant or another output's layer: then a copy is made under its name.
	copies := map[string]bool{}
	claimed := map[string]bool{}
	for _, o := range block.GetOutputs() {
		src := l.resolve(o)
		op := producer[src]
		if op == nil || op.GetType() == "const" || claimed[src] {
			copies[o] = true
			continue
		}
		claimed[src] = true
		l.rename[src] = o
	}
	for _, in := range fn.GetInputs() {
		t := in.GetType().GetTensorType()
		if t == nil {
			l.fail("input %s is not a tensor (states are not supported)", in.GetName())
		}
		l.vals[in.GetName()] = l.b.Parameter(in.GetName(), elemOf(t.GetDataType()), shapeOf(in.GetType())...)
	}
	for _, op := range block.GetOperations() {
		l.op(op)
	}
	for _, o := range block.GetOutputs() {
		v := l.tensorNamed(o)
		if copies[o] {
			v = l.b.NamedOp("Convert", "opset1", o, []string{"destination_type", v.Elem}, v.Elem, v.Shape, v)
		}
		l.b.Result(o, v)
	}
	return nil
}

// resolve follows identity operations to the value they pass on.
func (l *lowerer) resolve(name string) string {
	for {
		a, ok := l.alias[name]
		if !ok {
			return name
		}
		name = a
	}
}

// layerName is the name for the (last) layer computing the MIL value out.
func (l *lowerer) layerName(out string) string {
	if n, ok := l.rename[out]; ok {
		return n
	}
	return out
}

// tensorNamed returns the lowered value of a MIL name, lowering a constant on first use.
func (l *lowerer) tensorNamed(name string) ovir.Value {
	name = l.resolve(name)
	if v, ok := l.vals[name]; ok {
		return v
	}
	c, ok := l.consts[name]
	if !ok {
		l.fail("value %s is not defined", name)
	}
	v := l.constant(c, nil)
	l.vals[name] = v
	return v
}

// binding returns an argument as a tensor.
func (l *lowerer) binding(b *milspec.Argument_Binding) ovir.Value {
	if n := b.GetName(); n != "" {
		return l.tensorNamed(n)
	}
	return l.constant(b.GetValue(), nil)
}

// in returns the input key of op as a tensor.
func (l *lowerer) in(op *milspec.Operation, key string) ovir.Value {
	a := op.GetInputs()[key]
	if a == nil || len(a.GetArguments()) != 1 {
		l.fail("%s %s: input %s missing", op.GetType(), opName(op), key)
	}
	return l.binding(a.GetArguments()[0])
}

func (l *lowerer) has(op *milspec.Operation, key string) bool {
	a := op.GetInputs()[key]
	return a != nil && len(a.GetArguments()) > 0
}

// constOf returns the input key of op as a constant value, or nil when it is absent. An input
// that is not a constant stops the lowering: OpenVINO takes these as attributes or constants.
func (l *lowerer) constOf(op *milspec.Operation, key string) *milspec.Value {
	a := op.GetInputs()[key]
	if a == nil || len(a.GetArguments()) == 0 {
		return nil
	}
	b := a.GetArguments()[0]
	if n := b.GetName(); n != "" {
		c, ok := l.consts[l.resolve(n)]
		if !ok {
			l.fail("%s %s: input %s must be a constant", op.GetType(), opName(op), key)
		}
		return c
	}
	return b.GetValue()
}

func (l *lowerer) ints(op *milspec.Operation, key string, def ...int64) []int64 {
	c := l.constOf(op, key)
	if c == nil {
		if def == nil {
			l.fail("%s %s: input %s missing", op.GetType(), opName(op), key)
		}
		return def
	}
	tv := c.GetImmediateValue().GetTensor()
	if tv == nil || tv.GetInts() == nil {
		l.fail("%s %s: input %s is not an immediate int32 constant", op.GetType(), opName(op), key)
	}
	out := make([]int64, len(tv.GetInts().GetValues()))
	for i, v := range tv.GetInts().GetValues() {
		out[i] = int64(v)
	}
	return out
}

func (l *lowerer) bools(op *milspec.Operation, key string, n int) []bool {
	c := l.constOf(op, key)
	if c == nil {
		return make([]bool, n)
	}
	v := c.GetImmediateValue().GetTensor().GetBools().GetValues()
	if len(v) != n {
		l.fail("%s %s: %s has %d values for rank %d", op.GetType(), opName(op), key, len(v), n)
	}
	return v
}

func (l *lowerer) boolean(op *milspec.Operation, key string) bool {
	c := l.constOf(op, key)
	if c == nil {
		return false
	}
	v := c.GetImmediateValue().GetTensor().GetBools().GetValues()
	return len(v) == 1 && v[0]
}

func (l *lowerer) str(op *milspec.Operation, key, def string) string {
	c := l.constOf(op, key)
	if c == nil {
		return def
	}
	v := c.GetImmediateValue().GetTensor().GetStrings().GetValues()
	if len(v) != 1 {
		l.fail("%s %s: input %s is not a string", op.GetType(), opName(op), key)
	}
	return v[0]
}

func (l *lowerer) float(op *milspec.Operation, key string, def float32) float32 {
	c := l.constOf(op, key)
	if c == nil {
		return def
	}
	elem, _, raw := l.decode(c)
	switch {
	case elem == "f16" && len(raw) == 2:
		return mlpackage.Float16Value(binary.LittleEndian.Uint16(raw))
	case elem == "f32" && len(raw) == 4:
		return math.Float32frombits(binary.LittleEndian.Uint32(raw))
	}
	l.fail("%s %s: input %s is not a float scalar", op.GetType(), opName(op), key)
	return 0
}

// decode returns a constant's IR element type, shape and little endian bytes. fp32 becomes fp16:
// the programs compute in fp16, and OpenVINO's elementwise operations take one type.
func (l *lowerer) decode(c *milspec.Value) (elem string, shape []int, raw []byte) {
	shape = shapeOf(c.GetType())
	dt := c.GetType().GetTensorType().GetDataType()
	if bf := c.GetBlobFileValue(); bf != nil {
		if l.weights == nil {
			l.fail("a weight file value without a weight file")
		}
		bt, data, err := mlpackage.ReadBlob(l.weights, bf.GetOffset())
		if err != nil {
			l.fail("%v", err)
		}
		switch bt {
		case mlpackage.BlobFloat16:
			return "f16", shape, data
		case mlpackage.BlobFloat32:
			return "f16", shape, f32ToF16(data)
		case mlpackage.BlobInt8:
			return "i8", shape, data
		case mlpackage.BlobInt4:
			return "i4", shape, data
		case mlpackage.BlobInt32:
			return "i32", shape, data
		case mlpackage.BlobUInt4, mlpackage.BlobUInt6, mlpackage.BlobUInt8:
			l.fail("lookup table weights (lutN) do not run on Intel's NPU; convert with -format int4 or int8")
		}
		l.fail("weight blobs of type %d are not supported", bt)
	}
	tv := c.GetImmediateValue().GetTensor()
	if tv == nil {
		l.fail("unsupported constant %v", c)
	}
	switch dt {
	case milspec.DataType_FLOAT16:
		return "f16", shape, tv.GetBytes().GetValues()
	case milspec.DataType_FLOAT32:
		f := tv.GetFloats().GetValues()
		raw := make([]byte, 4*len(f))
		for i, v := range f {
			binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
		}
		return "f16", shape, f32ToF16(raw)
	case milspec.DataType_INT32:
		v := tv.GetInts().GetValues()
		raw := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(raw[4*i:], uint32(x))
		}
		return "i32", shape, raw
	case milspec.DataType_BOOL:
		v := tv.GetBools().GetValues()
		raw := make([]byte, len(v))
		for i, x := range v {
			if x {
				raw[i] = 1
			}
		}
		return "boolean", shape, raw
	}
	l.fail("constants of type %v are not supported", dt)
	return
}

// constant lowers a MIL constant, with the given shape instead of its own when not nil (the
// same elements in the same order).
func (l *lowerer) constant(c *milspec.Value, shape []int) ovir.Value {
	elem, own, raw := l.decode(c)
	if shape == nil {
		shape = own
	}
	return l.b.Const(elem, shape, raw)
}

func (l *lowerer) i64s(v ...int64) ovir.Value {
	raw := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(raw[8*i:], uint64(x))
	}
	return l.b.Const("i64", []int{len(v)}, raw)
}

func (l *lowerer) f16(v float32) ovir.Value {
	raw := make([]byte, 2)
	binary.LittleEndian.PutUint16(raw, mlpackage.Float16Bits(v))
	return l.b.Const("f16", []int{}, raw)
}

var numpy = []string{"auto_broadcast", "numpy"}

var binaryOps = map[string]string{"add": "Add", "sub": "Subtract", "mul": "Multiply", "real_div": "Divide", "maximum": "Maximum", "minimum": "Minimum"}

var unaryOps = map[string][2]string{
	"silu": {"Swish", "opset4"}, "sigmoid": {"Sigmoid", "opset1"}, "tanh": {"Tanh", "opset1"},
	"exp": {"Exp", "opset1"}, "relu": {"Relu", "opset1"}, "sqrt": {"Sqrt", "opset1"},
}

var reduceOps = map[string]string{"reduce_mean": "ReduceMean", "reduce_max": "ReduceMax", "reduce_sum": "ReduceSum"}

// op lowers one MIL operation.
func (l *lowerer) op(op *milspec.Operation) {
	typ := op.GetType()
	if len(op.GetOutputs()) != 1 {
		l.fail("%s: %d outputs", typ, len(op.GetOutputs()))
	}
	outName := op.GetOutputs()[0].GetName()
	if typ == "const" {
		l.consts[outName] = op.GetAttributes()["val"]
		return
	}
	name := l.layerName(outName)
	elem := elemOf(op.GetOutputs()[0].GetType().GetTensorType().GetDataType())
	shape := shapeOf(op.GetOutputs()[0].GetType())
	set := func(v ovir.Value) { l.vals[outName] = v }
	b := l.b
	switch typ {
	case "identity":
		if _, ok := l.alias[outName]; !ok {
			set(l.in(op, "x")) // an identity of an immediate value
		}
	case "add", "sub", "mul", "real_div", "maximum", "minimum":
		set(b.NamedOp(binaryOps[typ], "opset1", name, numpy, elem, shape, l.in(op, "x"), l.in(op, "y")))
	case "silu", "sigmoid", "tanh", "exp", "relu", "sqrt":
		u := unaryOps[typ]
		set(b.NamedOp(u[0], u[1], name, nil, elem, shape, l.in(op, "x")))
	case "rsqrt":
		// 1 / sqrt(x + epsilon)
		x := l.in(op, "x")
		eps := l.float(op, "epsilon", 1e-12)
		s := b.Op("Sqrt", "opset1", nil, elem, shape, b.Op("Add", "opset1", numpy, elem, shape, x, l.f16(eps)))
		set(b.NamedOp("Divide", "opset1", name, numpy, elem, shape, l.f16(1), s))
	case "gelu":
		x := l.in(op, "x")
		switch mode := l.str(op, "mode", "EXACT"); mode {
		case "EXACT":
			set(b.NamedOp("Gelu", "opset7", name, []string{"approximation_mode", "ERF"}, elem, shape, x))
		case "TANH_APPROXIMATION":
			set(b.NamedOp("Gelu", "opset7", name, []string{"approximation_mode", "TANH"}, elem, shape, x))
		case "SIGMOID_APPROXIMATION":
			s := b.Op("Sigmoid", "opset1", nil, elem, shape, b.Op("Multiply", "opset1", numpy, elem, shape, x, l.f16(1.702)))
			set(b.NamedOp("Multiply", "opset1", name, numpy, elem, shape, x, s))
		default:
			l.fail("gelu %s: mode %q", outName, mode)
		}
	case "softmax":
		set(b.NamedOp("SoftMax", "opset8", name, []string{"axis", itoa(l.ints(op, "axis", -1)[0])}, elem, shape, l.in(op, "x")))
	case "reduce_mean", "reduce_max", "reduce_sum":
		keep := strconv.FormatBool(l.boolean(op, "keep_dims"))
		set(b.NamedOp(reduceOps[typ], "opset1", name, []string{"keep_dims", keep}, elem, shape, l.in(op, "x"), l.i64s(l.ints(op, "axes")...)))
	case "matmul":
		attrs := []string{"transpose_a", strconv.FormatBool(l.boolean(op, "transpose_x")), "transpose_b", strconv.FormatBool(l.boolean(op, "transpose_y"))}
		set(b.NamedOp("MatMul", "opset1", name, attrs, elem, shape, l.in(op, "x"), l.in(op, "y")))
	case "linear":
		attrs := []string{"transpose_a", "false", "transpose_b", "true"}
		if !l.has(op, "bias") {
			set(b.NamedOp("MatMul", "opset1", name, attrs, elem, shape, l.in(op, "x"), l.in(op, "weight")))
			break
		}
		y := b.Op("MatMul", "opset1", attrs, elem, shape, l.in(op, "x"), l.in(op, "weight"))
		set(b.NamedOp("Add", "opset1", name, numpy, elem, shape, y, l.in(op, "bias")))
	case "reshape":
		set(b.NamedOp("Reshape", "opset1", name, []string{"special_zero", "false"}, elem, shape, l.in(op, "x"), l.i64s(l.ints(op, "shape")...)))
	case "transpose":
		set(b.NamedOp("Transpose", "opset1", name, nil, elem, shape, l.in(op, "x"), l.i64s(l.ints(op, "perm")...)))
	case "tile":
		set(b.NamedOp("Tile", "opset1", name, nil, elem, shape, l.in(op, "x"), l.i64s(l.ints(op, "reps")...)))
	case "slice_by_index":
		x := l.in(op, "x")
		n := len(x.Shape)
		ones := make([]int64, n)
		for i := range ones {
			ones[i] = 1
		}
		begin, end, stride := l.ints(op, "begin"), l.ints(op, "end"), l.ints(op, "stride", ones...)
		attrs := []string{
			"begin_mask", mask(l.bools(op, "begin_mask", n)),
			"end_mask", mask(l.bools(op, "end_mask", n)),
			"new_axis_mask", mask(make([]bool, n)),
			"shrink_axis_mask", mask(l.bools(op, "squeeze_mask", n)),
			"ellipsis_mask", mask(make([]bool, n)),
		}
		set(b.NamedOp("StridedSlice", "opset1", name, attrs, elem, shape, x, l.i64s(begin...), l.i64s(end...), l.i64s(stride...)))
	case "concat":
		if l.boolean(op, "interleave") {
			l.fail("concat %s: interleave is not supported", outName)
		}
		var xs []ovir.Value
		for _, a := range op.GetInputs()["values"].GetArguments() {
			xs = append(xs, l.binding(a))
		}
		set(b.NamedOp("Concat", "opset1", name, []string{"axis", itoa(l.ints(op, "axis")[0])}, elem, shape, xs...))
	case "pad":
		x := l.in(op, "x")
		pad := l.ints(op, "pad")
		n := len(x.Shape)
		if len(pad)%2 != 0 || len(pad)/2 > n {
			l.fail("pad %s: %d pads for rank %d", outName, len(pad), n)
		}
		// MIL pads the last len(pad)/2 axes, (begin, end) for each
		pb, pe := make([]int64, n), make([]int64, n)
		for i := 0; i < len(pad)/2; i++ {
			ax := n - len(pad)/2 + i
			pb[ax], pe[ax] = pad[2*i], pad[2*i+1]
		}
		mode := l.str(op, "mode", "constant")
		ovMode := map[string]string{"constant": "constant", "reflect": "reflect", "replicate": "edge"}[mode]
		if ovMode == "" {
			l.fail("pad %s: mode %q", outName, mode)
		}
		ins := []ovir.Value{x, l.i64s(pb...), l.i64s(pe...)}
		if mode == "constant" {
			ins = append(ins, l.f16(l.float(op, "constant_val", 0)))
		}
		set(b.NamedOp("Pad", "opset1", name, []string{"pad_mode", ovMode}, elem, shape, ins...))
	case "layer_norm":
		x := l.in(op, "x")
		eps := l.float(op, "epsilon", 1e-5)
		attrs := []string{"eps", strconv.FormatFloat(float64(eps), 'g', -1, 32), "normalize_variance", "true", "eps_mode", "inside_sqrt"}
		hasG, hasB := l.has(op, "gamma"), l.has(op, "beta")
		mvnName := ""
		if !hasG && !hasB {
			mvnName = name
		}
		y := b.NamedOp("MVN", "opset6", mvnName, attrs, elem, shape, x, l.i64s(l.ints(op, "axes")...))
		if hasG {
			gName := ""
			if !hasB {
				gName = name
			}
			y = b.NamedOp("Multiply", "opset1", gName, numpy, elem, shape, y, l.in(op, "gamma"))
		}
		if hasB {
			y = b.NamedOp("Add", "opset1", name, numpy, elem, shape, y, l.in(op, "beta"))
		}
		set(y)
	case "conv":
		set(l.conv(op, name, elem, shape))
	case "constexpr_blockwise_shift_scale":
		set(l.shiftScale(op, name, elem, shape))
	case "constexpr_lut_to_dense":
		l.fail("%s: lookup table weights (lutN) do not run on Intel's NPU; convert with -format int4 or int8", outName)
	default:
		l.fail("operation %s (%s) is not supported", typ, outName)
	}
}

// conv lowers a convolution: Convolution, or GroupConvolution with the weight (Cout, Cin/g,
// k...) seen as (g, Cout/g, Cin/g, k...), and the bias added over the channel axis.
func (l *lowerer) conv(op *milspec.Operation, name, elem string, shape []int) ovir.Value {
	b := l.b
	x := l.in(op, "x")
	spatial := len(x.Shape) - 2
	ones := make([]int64, spatial)
	for i := range ones {
		ones[i] = 1
	}
	strides, dil := l.ints(op, "strides", ones...), l.ints(op, "dilations", ones...)
	groups := l.ints(op, "groups", 1)[0]
	pb, pe := make([]int64, spatial), make([]int64, spatial)
	autoPad := "explicit"
	switch pt := l.str(op, "pad_type", "valid"); pt {
	case "valid":
		autoPad = "valid"
	case "same":
		autoPad = "same_upper"
	case "same_lower":
		autoPad = "same_lower"
	case "custom":
		pad := l.ints(op, "pad")
		if len(pad) != 2*spatial {
			l.fail("conv %s: %d pads for %d spatial axes", opName(op), len(pad), spatial)
		}
		for i := range spatial {
			pb[i], pe[i] = pad[2*i], pad[2*i+1]
		}
	default:
		l.fail("conv %s: pad_type %q", opName(op), pt)
	}
	attrs := []string{"strides", join(strides), "dilations", join(dil), "pads_begin", join(pb), "pads_end", join(pe), "auto_pad", autoPad}
	var w ovir.Value
	typ := "Convolution"
	if groups == 1 {
		w = l.in(op, "weight")
	} else {
		typ = "GroupConvolution"
		w = l.in(op, "weight")
		g := int(groups)
		ws := append([]int{g, w.Shape[0] / g}, w.Shape[1:]...)
		w = b.Op("Reshape", "opset1", []string{"special_zero", "false"}, w.Elem, ws, w, l.i64s(ints64(ws)...))
	}
	if !l.has(op, "bias") {
		return b.NamedOp(typ, "opset1", name, attrs, elem, shape, x, w)
	}
	y := b.Op(typ, "opset1", attrs, elem, shape, x, w)
	bias := l.in(op, "bias")
	bs := make([]int, len(shape))
	for i := range bs {
		bs[i] = 1
	}
	bs[1] = shape[1]
	bias = b.Op("Reshape", "opset1", []string{"special_zero", "false"}, bias.Elem, bs, bias, l.i64s(ints64(bs)...))
	return b.NamedOp("Add", "opset1", name, numpy, elem, shape, y, bias)
}

// shiftScale lowers a linearly quantized constant to the pattern Intel's NPU compiler reads
// compressed (NNCF's): the integers as a constant, Convert to fp16, Multiply by the scales. With
// a scale per block of columns, the last axis is split into (blocks, block) for the Multiply and
// joined again after it.
func (l *lowerer) shiftScale(op *milspec.Operation, name, elem string, shape []int) ovir.Value {
	b := l.b
	if l.has(op, "offset") {
		l.fail("%s: quantized weights with an offset are not supported (the intN formats are symmetric)", opName(op))
	}
	data, scale := l.constOf(op, "data"), l.constOf(op, "scale")
	if data == nil || scale == nil {
		l.fail("%s: data and scale must be constants", opName(op))
	}
	ds, ss := shapeOf(data.GetType()), shapeOf(scale.GetType())
	n := len(ds)
	if n == 0 || len(ss) != n {
		l.fail("%s: data %v and scale %v", opName(op), ds, ss)
	}
	for i := 0; i < n-1; i++ {
		if ss[i] != ds[i] {
			l.fail("%s: scales over blocks of %v rows; only blocks along the last axis are supported", opName(op), ds[i]/ss[i])
		}
	}
	if ss[n-1] == 1 {
		q := l.constant(data, nil)
		c := b.Op("Convert", "opset1", []string{"destination_type", elem}, elem, ds, q)
		return b.NamedOp("Multiply", "opset1", name, numpy, elem, ds, c, l.constant(scale, nil))
	}
	blocks := ss[n-1]
	if ds[n-1]%blocks != 0 {
		l.fail("%s: %d columns in %d blocks", opName(op), ds[n-1], blocks)
	}
	gs := append(append([]int{}, ds[:n-1]...), blocks, ds[n-1]/blocks)
	sc := append(append([]int{}, ds[:n-1]...), blocks, 1)
	q := l.constant(data, gs)
	c := b.Op("Convert", "opset1", []string{"destination_type", elem}, elem, gs, q)
	m := b.Op("Multiply", "opset1", numpy, elem, gs, c, l.constant(scale, sc))
	return b.NamedOp("Reshape", "opset1", name, []string{"special_zero", "false"}, elem, shape, m, l.i64s(ints64(shape)...))
}

func opName(op *milspec.Operation) string {
	if outs := op.GetOutputs(); len(outs) > 0 {
		return outs[0].GetName()
	}
	return ""
}

func shapeOf(t *milspec.ValueType) []int {
	tt := t.GetTensorType()
	shape := make([]int, len(tt.GetDimensions()))
	for i, d := range tt.GetDimensions() {
		shape[i] = int(d.GetConstant().GetSize())
	}
	return shape
}

func elemOf(dt milspec.DataType) string {
	switch dt {
	case milspec.DataType_FLOAT16:
		return "f16"
	case milspec.DataType_FLOAT32:
		return "f32"
	case milspec.DataType_INT32:
		return "i32"
	case milspec.DataType_INT8:
		return "i8"
	case milspec.DataType_INT4:
		return "i4"
	case milspec.DataType_BOOL:
		return "boolean"
	}
	panic(lowerError{fmt.Errorf("ovlower: values of type %v are not supported", dt)})
}

func f32ToF16(raw []byte) []byte {
	out := make([]byte, len(raw)/2)
	for i := 0; i < len(raw)/4; i++ {
		binary.LittleEndian.PutUint16(out[2*i:], mlpackage.Float16Bits(math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))))
	}
	return out
}

func mask(v []bool) string {
	s := make([]string, len(v))
	for i, b := range v {
		s[i] = "0"
		if b {
			s[i] = "1"
		}
	}
	return strings.Join(s, ",")
}

func itoa(v int64) string { return strconv.FormatInt(v, 10) }

func join(v []int64) string {
	s := make([]string, len(v))
	for i, x := range v {
		s[i] = itoa(x)
	}
	return strings.Join(s, ",")
}

func ints64(v []int) []int64 {
	out := make([]int64, len(v))
	for i, x := range v {
		out[i] = int64(x)
	}
	return out
}

// rtKey makes a metadata key an XML name: letters, digits, '_', '-' and '.' are kept.
func rtKey(k string) string {
	r := []byte(k)
	for i, c := range r {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_' || c == '-' || c == '.') {
			r[i] = '_'
		}
	}
	if len(r) == 0 || r[0] >= '0' && r[0] <= '9' || r[0] == '-' || r[0] == '.' {
		r = append([]byte("_"), r...)
	}
	return string(r)
}

// WriteFiles lowers fn as Lower does to the IR at path (.xml), with its constants in the .bin
// next to it, and returns the size of the .bin.
func WriteFiles(path, name string, fn *milspec.Function, weights io.ReaderAt, meta map[string]string) (int64, error) {
	binPath := strings.TrimSuffix(path, filepath.Ext(path)) + ".bin"
	f, err := os.Create(binPath)
	if err != nil {
		return 0, err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	xml, err := Lower(name, fn, weights, w, meta)
	if err == nil {
		err = w.Flush()
	}
	var size int64
	if info, serr := f.Stat(); serr == nil {
		size = info.Size()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.WriteFile(path, xml, 0o644)
	}
	if err != nil {
		os.Remove(binPath)
		return 0, err
	}
	return size, nil
}
