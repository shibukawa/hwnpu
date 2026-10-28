// Package mlpackage writes Core ML model packages (.mlpackage) holding an ML Program, without
// coremltools: the MIL program as protobuf, the weights in the MILBlob storage format and the
// package manifest. It covers the operations hwnpu's converters emit, not the whole MIL spec.
package mlpackage

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/shibukawa/hwnpu/internal/milspec"
)

// Data types of program values, in MIL's numbering.
const (
	Float16 = milspec.DataType_FLOAT16
	Float32 = milspec.DataType_FLOAT32
	Int32   = milspec.DataType_INT32
	UInt4   = milspec.DataType_UINT4
	Bool    = milspec.DataType_BOOL
	String  = milspec.DataType_STRING
)

// Var is a value in the program: a function input, a constant or an operation output.
type Var struct {
	Name  string
	DType milspec.DataType
	Shape []int // nil for a scalar
}

// Builder assembles one MIL function (main) and its constants.
type Builder struct {
	opset  string
	blobs  *BlobWriter
	inputs []*milspec.NamedValueType
	ops    []*milspec.Operation
	names  map[string]bool
	seq    int
	// ImmediateLimit is the largest constant, in bytes, kept inline in the program; larger ones go
	// to the weight file. coremltools uses a similar threshold.
	ImmediateLimit int
}

// NewBuilder starts a program for the given opset ("CoreML8" is iOS 18 / macOS 15). Constants
// larger than ImmediateLimit are written to blobs, which must not be nil in that case.
func NewBuilder(opset string, blobs *BlobWriter) *Builder {
	return &Builder{opset: opset, blobs: blobs, names: map[string]bool{}, ImmediateLimit: 256}
}

// unique makes a value name the loader accepts (letters, digits and underscores; a name with
// dots crashes Core ML's loader) and distinct from the names used so far.
func (b *Builder) unique(name string) string {
	clean := []byte(name)
	for i, c := range clean {
		if !(c >= 'a' && c <= 'z' || c >= 'A' && c <= 'Z' || c >= '0' && c <= '9' || c == '_') {
			clean[i] = '_'
		}
	}
	name = string(clean)
	if !b.names[name] {
		b.names[name] = true
		return name
	}
	for {
		b.seq++
		n := fmt.Sprintf("%s_%d", name, b.seq)
		if !b.names[n] {
			b.names[n] = true
			return n
		}
	}
}

func tensorType(dt milspec.DataType, shape []int) *milspec.ValueType {
	t := &milspec.TensorType{DataType: dt, Rank: int64(len(shape))}
	for _, d := range shape {
		t.Dimensions = append(t.Dimensions, &milspec.Dimension{Dimension: &milspec.Dimension_Constant{Constant: &milspec.Dimension_ConstantDimension{Size: uint64(d)}}})
	}
	return &milspec.ValueType{Type: &milspec.ValueType_TensorType{TensorType: t}}
}

func stringValue(s string) *milspec.Value {
	return &milspec.Value{Type: tensorType(String, nil), Value: &milspec.Value_ImmediateValue_{ImmediateValue: &milspec.Value_ImmediateValue{
		Value: &milspec.Value_ImmediateValue_Tensor{Tensor: &milspec.TensorValue{Value: &milspec.TensorValue_Strings{Strings: &milspec.TensorValue_RepeatedStrings{Values: []string{s}}}}}}}}
}

// Input declares a function input.
func (b *Builder) Input(name string, dt milspec.DataType, shape ...int) *Var {
	b.names[name] = true
	b.inputs = append(b.inputs, &milspec.NamedValueType{Name: name, Type: tensorType(dt, shape)})
	return &Var{Name: name, DType: dt, Shape: shape}
}

// Op appends an operation. Inputs are bound by name; an input may be one Var or a []*Var (for
// operations such as concat that take a list). The output type is given by the caller.
func (b *Builder) Op(typ string, inputs map[string]any, outDT milspec.DataType, outShape ...int) *Var {
	name := b.unique(typ)
	return b.op(typ, name, inputs, outDT, outShape)
}

// NamedOp is Op with an explicit output name.
func (b *Builder) NamedOp(typ, name string, inputs map[string]any, outDT milspec.DataType, outShape ...int) *Var {
	return b.op(typ, b.unique(name), inputs, outDT, outShape)
}

func (b *Builder) op(typ, name string, inputs map[string]any, outDT milspec.DataType, outShape []int) *Var {
	op := &milspec.Operation{Type: typ, Inputs: map[string]*milspec.Argument{}, Attributes: map[string]*milspec.Value{"name": stringValue(name)}}
	for k, v := range inputs {
		arg := &milspec.Argument{}
		switch v := v.(type) {
		case *Var:
			arg.Arguments = append(arg.Arguments, &milspec.Argument_Binding{Binding: &milspec.Argument_Binding_Name{Name: v.Name}})
		case []*Var:
			for _, x := range v {
				arg.Arguments = append(arg.Arguments, &milspec.Argument_Binding{Binding: &milspec.Argument_Binding_Name{Name: x.Name}})
			}
		case *milspec.Value:
			arg.Arguments = append(arg.Arguments, &milspec.Argument_Binding{Binding: &milspec.Argument_Binding_Value{Value: v}})
		default:
			panic(fmt.Sprintf("mlpackage: %s input %s: unsupported %T", typ, k, v))
		}
		op.Inputs[k] = arg
	}
	if outShape == nil {
		outShape = []int{}
	}
	op.Outputs = []*milspec.NamedValueType{{Name: name, Type: tensorType(outDT, outShape)}}
	b.ops = append(b.ops, op)
	return &Var{Name: name, DType: outDT, Shape: outShape}
}

// constValue builds a Value of the given type from little endian bytes, inline or as a blob.
func (b *Builder) constValue(dt milspec.DataType, shape []int, raw []byte, blobType BlobDType) (*milspec.Value, error) {
	v := &milspec.Value{Type: tensorType(dt, shape)}
	if len(raw) > b.ImmediateLimit || dt == UInt4 || dt == milspec.DataType_INT4 {
		if b.blobs == nil {
			return nil, fmt.Errorf("mlpackage: a %d byte constant needs a weight file", len(raw))
		}
		off, err := b.blobs.Add(blobType, raw)
		if err != nil {
			return nil, err
		}
		v.Value = &milspec.Value_BlobFileValue_{BlobFileValue: &milspec.Value_BlobFileValue{FileName: "@model_path/weights/weight.bin", Offset: off}}
		return v, nil
	}
	tv := &milspec.TensorValue{}
	switch dt {
	case Float16:
		tv.Value = &milspec.TensorValue_Bytes{Bytes: &milspec.TensorValue_RepeatedBytes{Values: raw}}
	case Float32:
		f := make([]float32, len(raw)/4)
		for i := range f {
			f[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		tv.Value = &milspec.TensorValue_Floats{Floats: &milspec.TensorValue_RepeatedFloats{Values: f}}
	case Int32:
		n := make([]int32, len(raw)/4)
		for i := range n {
			n[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
		}
		tv.Value = &milspec.TensorValue_Ints{Ints: &milspec.TensorValue_RepeatedInts{Values: n}}
	case Bool:
		bs := make([]bool, len(raw))
		for i := range bs {
			bs[i] = raw[i] != 0
		}
		tv.Value = &milspec.TensorValue_Bools{Bools: &milspec.TensorValue_RepeatedBools{Values: bs}}
	default:
		return nil, fmt.Errorf("mlpackage: immediate constants of type %v are not supported", dt)
	}
	v.Value = &milspec.Value_ImmediateValue_{ImmediateValue: &milspec.Value_ImmediateValue{Value: &milspec.Value_ImmediateValue_Tensor{Tensor: tv}}}
	return v, nil
}

func (b *Builder) constOp(name string, dt milspec.DataType, shape []int, raw []byte, blobType BlobDType) (*Var, error) {
	v, err := b.constValue(dt, shape, raw, blobType)
	if err != nil {
		return nil, err
	}
	name = b.unique(name)
	b.ops = append(b.ops, &milspec.Operation{Type: "const", Inputs: map[string]*milspec.Argument{},
		Attributes: map[string]*milspec.Value{"name": stringValue(name), "val": v},
		Outputs:    []*milspec.NamedValueType{{Name: name, Type: tensorType(dt, shape)}}})
	return &Var{Name: name, DType: dt, Shape: shape}, nil
}

// F16 adds an fp16 constant from already encoded halves. shape nil makes a scalar.
func (b *Builder) F16(name string, shape []int, halves []uint16) (*Var, error) {
	raw := make([]byte, 2*len(halves))
	for i, h := range halves {
		binary.LittleEndian.PutUint16(raw[2*i:], h)
	}
	return b.constOp(name, Float16, shape, raw, BlobFloat16)
}

// F16s converts float32 values to fp16 and adds them as a constant.
func (b *Builder) F16s(name string, shape []int, vals []float32) (*Var, error) {
	h := make([]uint16, len(vals))
	for i, v := range vals {
		h[i] = Float16Bits(v)
	}
	return b.F16(name, shape, h)
}

// F32 adds an fp32 constant.
func (b *Builder) F32(name string, shape []int, vals []float32) (*Var, error) {
	raw := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[4*i:], math.Float32bits(v))
	}
	return b.constOp(name, Float32, shape, raw, BlobFloat32)
}

// I32 adds an int32 constant.
func (b *Builder) I32(name string, shape []int, vals []int32) (*Var, error) {
	raw := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[4*i:], uint32(v))
	}
	return b.constOp(name, Int32, shape, raw, BlobInt32)
}

// Bools adds a bool constant.
func (b *Builder) Bools(name string, shape []int, vals []bool) (*Var, error) {
	raw := make([]byte, len(vals))
	for i, v := range vals {
		if v {
			raw[i] = 1
		}
	}
	return b.constOp(name, Bool, shape, raw, BlobUInt8)
}

// Str adds a string constant (operation modes such as pad_type).
func (b *Builder) Str(name, s string) *Var {
	name = b.unique(name)
	b.ops = append(b.ops, &milspec.Operation{Type: "const", Inputs: map[string]*milspec.Argument{},
		Attributes: map[string]*milspec.Value{"name": stringValue(name), "val": stringValue(s)},
		Outputs:    []*milspec.NamedValueType{{Name: name, Type: tensorType(String, nil)}}})
	return &Var{Name: name, DType: String}
}

// Scalar helpers for operation parameters.

func (b *Builder) ScalarF16(v float32) *Var { x, _ := b.F16s("c", nil, []float32{v}); return x }
func (b *Builder) ScalarF32(v float32) *Var { x, _ := b.F32("c", nil, []float32{v}); return x }
func (b *Builder) ScalarI32(v int32) *Var   { x, _ := b.I32("c", nil, []int32{v}); return x }
func (b *Builder) ScalarBool(v bool) *Var   { x, _ := b.Bools("c", nil, []bool{v}); return x }
func (b *Builder) Ints(v ...int32) *Var     { x, _ := b.I32("c", []int{len(v)}, v); return x }

// LUT4 adds a 4 bit palettized fp16 tensor: constexpr_lut_to_dense with one 16 entry table per
// block of the dense tensor. block gives the block size along each axis of shape (per_grouped_
// channel with group 16 in coremltools is block [16, cols]). indices holds one 4 bit entry per
// element, row major, two per byte with the low nibble first; lut holds 16 fp16 halves per block,
// blocks in row major order. shape is the dense shape, [rows, cols...].
func (b *Builder) LUT4(name string, shape, block []int, indices []byte, lut []uint16) (*Var, error) {
	if len(shape) < 1 || len(block) != len(shape) {
		return nil, fmt.Errorf("mlpackage: %s: block %v does not match shape %v", name, block, shape)
	}
	n, tables := 1, 1
	lutShape := make([]int, len(shape)+2)
	for i, d := range shape {
		if block[i] <= 0 || d%block[i] != 0 {
			return nil, fmt.Errorf("mlpackage: %s: axis %d of %v not a multiple of block %v", name, i, shape, block)
		}
		n *= d
		lutShape[i] = d / block[i]
		tables *= lutShape[i]
	}
	if len(indices) != (n+1)/2 || len(lut) != tables*16 {
		return nil, fmt.Errorf("mlpackage: %s: %d index bytes and %d lut entries for shape %v block %v", name, len(indices), len(lut), shape, block)
	}
	idx, err := b.constValue(UInt4, shape, indices, BlobUInt4)
	if err != nil {
		return nil, err
	}
	lutShape[len(shape)] = 16
	lutShape[len(shape)+1] = 1
	raw := make([]byte, 2*len(lut))
	for i, h := range lut {
		binary.LittleEndian.PutUint16(raw[2*i:], h)
	}
	save := b.ImmediateLimit
	b.ImmediateLimit = 0 // coremltools always puts the table in the weight file
	lv, err := b.constValue(Float16, lutShape, raw, BlobFloat16)
	b.ImmediateLimit = save
	if err != nil {
		return nil, err
	}
	return b.op("constexpr_lut_to_dense", b.unique(name), map[string]any{"indices": idx, "lut": lv}, Float16, shape), nil
}

// Int4Scaled adds a linearly quantized fp16 tensor: constexpr_blockwise_shift_scale over int4
// data with one fp16 scale per block (block gives the block size along each axis of shape), the
// layout of llama.cpp's Q4_0 when block is [1, 32]. data holds two's complement 4 bit values, two
// per byte with the low nibble first; scales are fp16 halves, blocks in row major order.
func (b *Builder) Int4Scaled(name string, shape, block []int, data []byte, scales []uint16) (*Var, error) {
	return b.scaled(name, 4, shape, block, data, scales)
}

// Int8Scaled is Int4Scaled for int8 data, one byte per element (coremltools' linear symmetric
// int8 per output channel when block is [1, cols]).
func (b *Builder) Int8Scaled(name string, shape, block []int, data []byte, scales []uint16) (*Var, error) {
	return b.scaled(name, 8, shape, block, data, scales)
}

func (b *Builder) scaled(name string, bits int, shape, block []int, data []byte, scales []uint16) (*Var, error) {
	if len(shape) < 1 || len(block) != len(shape) {
		return nil, fmt.Errorf("mlpackage: %s: block %v does not match shape %v", name, block, shape)
	}
	n, tables := 1, 1
	scaleShape := make([]int, len(shape))
	for i, d := range shape {
		if block[i] <= 0 || d%block[i] != 0 {
			return nil, fmt.Errorf("mlpackage: %s: axis %d of %v not a multiple of block %v", name, i, shape, block)
		}
		n *= d
		scaleShape[i] = d / block[i]
		tables *= scaleShape[i]
	}
	dt, bt, size := milspec.DataType_INT4, BlobInt4, (n+1)/2
	if bits == 8 {
		dt, bt, size = milspec.DataType_INT8, BlobInt8, n
	}
	if len(data) != size || len(scales) != tables {
		return nil, fmt.Errorf("mlpackage: %s: %d data bytes and %d scales for shape %v block %v", name, len(data), len(scales), shape, block)
	}
	dv, err := b.constValue(dt, shape, data, bt)
	if err != nil {
		return nil, err
	}
	raw := make([]byte, 2*len(scales))
	for i, h := range scales {
		binary.LittleEndian.PutUint16(raw[2*i:], h)
	}
	save := b.ImmediateLimit
	b.ImmediateLimit = 0
	sv, err := b.constValue(Float16, scaleShape, raw, BlobFloat16)
	b.ImmediateLimit = save
	if err != nil {
		return nil, err
	}
	return b.op("constexpr_blockwise_shift_scale", b.unique(name), map[string]any{"data": dv, "scale": sv}, Float16, shape), nil
}

// Finish returns the program with the given outputs.
func (b *Builder) Finish(outputs ...*Var) *milspec.Program {
	block := &milspec.Block{Operations: b.ops}
	for _, o := range outputs {
		block.Outputs = append(block.Outputs, o.Name)
	}
	fn := &milspec.Function{Inputs: b.inputs, Opset: b.opset, BlockSpecializations: map[string]*milspec.Block{b.opset: block}}
	return &milspec.Program{Version: 1, Functions: map[string]*milspec.Function{"main": fn}}
}

// Float16Bits converts a float32 to IEEE half precision, rounding to nearest even.
func Float16Bits(f float32) uint16 {
	x := math.Float32bits(f)
	sign := uint16(x>>16) & 0x8000
	exp := int((x>>23)&0xff) - 127 + 15
	mant := x & 0x7fffff
	switch {
	case (x>>23)&0xff == 0xff: // inf or nan
		if mant != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp >= 0x1f:
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		h := uint16(mant >> shift)
		rem := mant & (1<<shift - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && h&1 == 1) {
			h++
		}
		return sign | h
	}
	h := sign | uint16(exp)<<10 | uint16(mant>>13)
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && h&1 == 1) {
		h++
	}
	return h
}

// Float16Value converts IEEE half precision bits to float32.
func Float16Value(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	mant := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if mant == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 15 + 1)
		for mant&0x400 == 0 {
			mant <<= 1
			e--
		}
		mant &= 0x3ff
		return math.Float32frombits(sign | e<<23 | mant<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | mant<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | mant<<13)
}
