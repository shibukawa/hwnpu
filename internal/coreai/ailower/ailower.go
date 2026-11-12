// Package ailower writes a MIL program built with internal/mlpackage as a Core AI program (the
// aicode dialect's MLIR bytecode of an .aimodel, internal/coreai/mlirbc), as internal/ovlower does
// for Intel's NPU: the architectures describe their models once, as MIL, and each MIL operation
// they use becomes a few of Core AI's primitive operations (aicode.primitive.*_v1), in the forms
// coreai-core writes them (tools/coreai/probes: what each torch operation becomes).
//
// The translation is strict: an operation, an input or a setting it does not know stops it with
// an error. Two rewrites matter for the Neural Engine: Core AI turns matmul, softmax, matmul (and
// exp(x - max) / sum) into one attention operation that it does not place on the Neural Engine,
// so softmax is written as exp(x - max) times the reciprocal of its sum; and every binary
// operation takes operands of one shape, so broadcasts are explicit (broadcast_in_dims).
package ailower

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/coreai/mlirbc"
	"github.com/shibukawa/hwnpu/internal/milspec"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Producer is the producer string of the bytecode hwnpu writes.
const Producer = "hwnpu-ailower"

// assetVersion is the aicode serialization asset's version coreai-core 1.0.0b3 writes.
const assetVersion = 20260204

// inlineBytes is the largest constant written inline (dense_elements); larger ones are resources.
const inlineBytes = 256

// Feature is an input or output of the program.
type Feature struct {
	Name  string
	Elem  string // "f16", "f32", "i32"
	Shape []int
}

// Program is a lowered program.
type Program struct {
	Bytecode []byte
	Inputs   []Feature
	Outputs  []Feature
}

// Lower lowers the MIL function fn, whose blob values point into weights (weight.bin).
func Lower(fn *milspec.Function, weights io.ReaderAt) (*Program, error) {
	block := fn.GetBlockSpecializations()[fn.GetOpset()]
	if block == nil {
		return nil, fmt.Errorf("ailower: the function has no block for opset %q", fn.GetOpset())
	}
	w := mlirbc.NewWriter(Producer, map[string][]byte{"aicode": {0x05}})
	if os.Getenv("HWNPU_AILOWER_TRACE") != "" {
		w.Trace = func(id int, name string, operands, results []int) {
			var o, r []string
			for _, t := range operands {
				o = append(o, w.TypeText(t))
			}
			for _, t := range results {
				r = append(r, w.TypeText(t))
			}
			fmt.Fprintf(os.Stderr, "op_id %d %s (%s) -> (%s)\n", id, name, strings.Join(o, ", "), strings.Join(r, ", "))
		}
	}
	l := &lowerer{
		w:       w,
		weights: weights,
		vals:    map[string]val{},
		consts:  map[string]*milspec.Value{},
		alias:   map[string]string{},
	}
	return l.lower(fn, block)
}

// LowerPackage lowers an .mlpackage's main function.
func LowerPackage(dir string) (*Program, error) {
	m, err := mlpackage.ReadModel(dir)
	if err != nil {
		return nil, err
	}
	prog := m.GetMlProgram()
	if prog == nil {
		return nil, fmt.Errorf("ailower: %s is not an ML Program", dir)
	}
	fn := prog.GetFunctions()["main"]
	if fn == nil {
		return nil, fmt.Errorf("ailower: %s has no main function", dir)
	}
	f, err := os.Open(filepath.Join(dir, "Data", "com.apple.CoreML", "weights", "weight.bin"))
	if err != nil && !os.IsNotExist(err) {
		return nil, err
	}
	var weights io.ReaderAt
	if f != nil {
		defer f.Close()
		weights = f
	}
	p, err := Lower(fn, weights)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", filepath.Base(dir), err)
	}
	return p, nil
}

// WriteAsset writes the program as an .aimodel directory (main.mlirb, main.hash,
// metadata.json) at dir.
func (p *Program) WriteAsset(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	sum := sha256.Sum256(p.Bytecode)
	meta := fmt.Sprintf("{\n  \"assetVersion\" : \"2.0\",\n  \"producer\" : %q,\n  \"creationDate\" : %q\n}", Producer, time.Now().UTC().Format("20060102T150405Z"))
	io, err := json.MarshalIndent(struct{ Inputs, Outputs []Feature }{p.Inputs, p.Outputs}, "", "  ")
	if err != nil {
		return err
	}
	for name, b := range map[string][]byte{"main.mlirb": p.Bytecode, "main.hash": sum[:], "metadata.json": []byte(meta), IOFile: io} {
		if err := os.WriteFile(filepath.Join(dir, name), b, 0o644); err != nil {
			return err
		}
	}
	return nil
}

// IOFile is the file of an asset hwnpu writes that names its inputs and outputs (ReadIO).
const IOFile = "hwnpu-io.json"

// ReadIO reads the inputs and outputs of an asset written by WriteAsset.
func ReadIO(dir string) (inputs, outputs []Feature, err error) {
	b, err := os.ReadFile(filepath.Join(dir, IOFile))
	if err != nil {
		return nil, nil, err
	}
	var io struct{ Inputs, Outputs []Feature }
	if err := json.Unmarshal(b, &io); err != nil {
		return nil, nil, fmt.Errorf("ailower: %s: %w", dir, err)
	}
	return io.Inputs, io.Outputs, nil
}

// Hash is the hex sha256 of the bytecode (main.hash, the name of Core AI's cache entry).
func (p *Program) Hash() string {
	sum := sha256.Sum256(p.Bytecode)
	return hex.EncodeToString(sum[:])
}

type val struct {
	v     mlirbc.Value
	elem  string // aicode element: float16, float32, sint32, uint32, bool, ...
	shape []int
}

type lowerer struct {
	w       *mlirbc.Writer
	g       *mlirbc.BlockW
	weights io.ReaderAt
	vals    map[string]val
	consts  map[string]*milspec.Value
	alias   map[string]string
	blobs   int
}

type lowerError struct{ err error }

func (l *lowerer) fail(format string, args ...any) {
	panic(lowerError{fmt.Errorf("ailower: "+format, args...)})
}

func (l *lowerer) lower(fn *milspec.Function, block *milspec.Block) (p *Program, err error) {
	defer func() {
		if r := recover(); r != nil {
			if e, ok := r.(lowerError); ok {
				err = e.err
				return
			}
			panic(r)
		}
	}()
	p = &Program{}
	var argTypes []int
	var argNames []string
	var ins []val
	for _, in := range fn.GetInputs() {
		t := in.GetType().GetTensorType()
		if t == nil {
			l.fail("input %s is not a tensor (states are not supported)", in.GetName())
		}
		elem := elemOf(t.GetDataType())
		shape := shapeOf(in.GetType())
		argTypes = append(argTypes, l.ttype(elem, shape))
		argNames = append(argNames, in.GetName())
		ins = append(ins, val{elem: elem, shape: shape})
		p.Inputs = append(p.Inputs, Feature{in.GetName(), short(elem), shape})
	}
	g, args := l.w.NewBlock(argTypes...)
	l.g = g
	for i, a := range args {
		ins[i].v = a
		l.vals[argNames[i]] = ins[i]
	}
	outputs := block.GetOutputs()
	cut := 0 // HWNPU_AILOWER_CUT=N: only the first N operations, the last one's value the output
	fmt.Sscan(os.Getenv("HWNPU_AILOWER_CUT"), &cut)
	for i, op := range block.GetOperations() {
		if cut > 0 && i >= cut {
			break
		}
		l.op(op)
		if cut > 0 && op.GetType() != "const" {
			outputs = []string{op.GetOutputs()[0].GetName()}
		}
	}
	var outs []mlirbc.Value
	var outTypes []string
	for _, o := range outputs {
		v := l.named(o)
		outs = append(outs, v.v)
		outTypes = append(outTypes, l.ttext(v.elem, v.shape))
		p.Outputs = append(p.Outputs, Feature{o, short(v.elem), v.shape})
	}
	g.Op("aicode.dataflow.output_v1", nil, outs, nil)

	var inTypes []string
	for _, v := range ins {
		inTypes = append(inTypes, l.ttext(v.elem, v.shape))
	}
	names := func(ns []string) string {
		var parts []string
		for _, n := range ns {
			parts = append(parts, fmt.Sprintf("#aicode.serialization.array<[#aicode.serialization.opaque<\"aicode.name\" = [#aicode.serialization.string<%q>]>]>", n))
		}
		return "#aicode.serialization.array<[" + strings.Join(parts, ", ") + "]>"
	}
	var outNames []string
	for _, f := range p.Outputs {
		outNames = append(outNames, f.Name)
	}
	zero1 := l.prop("#aicode.serialization.integer<(1bit unsigned) 0>")
	absent := mlirbc.Prop{Attr: -1, Optional: true}
	// graph_v1's properties in their order: argAttrs, (absent), externalize, functionType,
	// isPrivate, name, noInline, paramDecls, resAttrs
	gp := []mlirbc.Prop{l.prop(names(argNames)), absent, zero1,
		l.prop("#aicode.serialization.type<!aicode.function_v1<(operands " + strings.Join(inTypes, ", ") + ") -> (results " + strings.Join(outTypes, ", ") + ")>>"),
		zero1, l.prop(`#aicode.serialization.string<"main">`), zero1, l.prop("#aicode.serialization.array<>"), l.prop(names(outNames))}
	asset, _ := l.w.NewBlock()
	asset.Op("aicode.dataflow.graph_v1", gp, nil, nil, g)
	top, _ := l.w.NewBlock()
	ver := l.w.Attr("aicode", fmt.Sprintf("#aicode.serialization.integer<(32bit unsigned) %d>", assetVersion))
	top.Op("aicode.serialization.asset", []mlirbc.Prop{absent, {Attr: ver, Optional: true}}, nil, nil, asset)
	p.Bytecode = l.w.Bytes(top)
	return p, nil
}

func (l *lowerer) prop(text string) mlirbc.Prop { return mlirbc.Prop{Attr: l.w.Attr("aicode", text)} }

func short(elem string) string {
	switch elem {
	case "float16":
		return "f16"
	case "float32":
		return "f32"
	case "sint32":
		return "i32"
	}
	return elem
}

func elemOf(dt milspec.DataType) string {
	switch dt {
	case milspec.DataType_FLOAT16:
		return "float16"
	case milspec.DataType_FLOAT32:
		return "float32"
	case milspec.DataType_INT32:
		return "sint32"
	case milspec.DataType_BOOL:
		return "bool"
	}
	panic(lowerError{fmt.Errorf("ailower: data type %v is not supported", dt)})
}

func shapeOf(t *milspec.ValueType) []int {
	tt := t.GetTensorType()
	if tt == nil {
		return nil
	}
	out := make([]int, len(tt.GetDimensions()))
	for i, d := range tt.GetDimensions() {
		c := d.GetConstant()
		if c == nil {
			panic(lowerError{fmt.Errorf("ailower: a dimension that is not constant")})
		}
		out[i] = int(c.GetSize())
	}
	return out
}

func (l *lowerer) ttext(elem string, shape []int) string {
	if len(shape) == 0 {
		return "!aicode.tensor_v1<!aicode." + elem + "_v1>"
	}
	d := make([]string, len(shape))
	for i, n := range shape {
		d[i] = fmt.Sprint(n)
	}
	return fmt.Sprintf("!aicode.tensor_v1<[%s] : !aicode.%s_v1>", strings.Join(d, ", "), elem)
}

func (l *lowerer) ttype(elem string, shape []int) int {
	return l.w.Type("aicode", l.ttext(elem, shape))
}

// emit appends a primitive operation with one result.
func (l *lowerer) emit(name string, elem string, shape []int, props []mlirbc.Prop, operands ...val) val {
	ops := make([]mlirbc.Value, len(operands))
	for i, o := range operands {
		ops[i] = o.v
	}
	r := l.g.Op("aicode.primitive."+name, props, ops, []int{l.ttype(elem, shape)})[0]
	return val{v: r, elem: elem, shape: slices.Clone(shape)}
}

// constRaw is a constant of the given element type, shape and little endian bytes.
func (l *lowerer) constRaw(elem string, shape []int, raw []byte) val {
	n := 1
	for _, d := range shape {
		n *= d
	}
	splat := n > 1 && allSame(raw, len(raw)/n)
	var text string
	switch {
	case splat:
		text = fmt.Sprintf("#aicode.serialization.dense_elements<[splat = true] %q>", base64.StdEncoding.EncodeToString(raw[:len(raw)/n]))
	case len(raw) <= inlineBytes:
		text = fmt.Sprintf("#aicode.serialization.dense_elements<[splat = false] %q>", base64.StdEncoding.EncodeToString(raw))
	default:
		key := fmt.Sprintf("hwnpu_%d", l.blobs)
		l.blobs++
		l.w.Blob("aicode", key, 16, raw)
		text = "#aicode.serialization.dense_resource<" + key + ">"
	}
	return l.emit("constant_v1", elem, shape, []mlirbc.Prop{l.prop(text)})
}

func allSame(raw []byte, size int) bool {
	if size == 0 {
		return false
	}
	for i := size; i < len(raw); i += size {
		for j := 0; j < size; j++ {
			if raw[i+j] != raw[j] {
				return false
			}
		}
	}
	return true
}

func le32(v ...int) []byte {
	b := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(b[4*i:], uint32(int32(x)))
	}
	return b
}

func (l *lowerer) i32s(v ...int) val { return l.constRaw("sint32", []int{len(v)}, le32(v...)) }
func (l *lowerer) u32s(v ...int) val { return l.constRaw("uint32", []int{len(v)}, le32(v...)) }
func (l *lowerer) i32(v int) val     { return l.constRaw("sint32", nil, le32(v)) }
func (l *lowerer) u32(v int) val     { return l.constRaw("uint32", nil, le32(v)) }

// scalar is a scalar of x's element type.
func (l *lowerer) scalar(elem string, f float32) val {
	switch elem {
	case "float16":
		b := make([]byte, 2)
		binary.LittleEndian.PutUint16(b, mlpackage.Float16Bits(f))
		return l.constRaw(elem, nil, b)
	case "float32":
		b := make([]byte, 4)
		binary.LittleEndian.PutUint32(b, math.Float32bits(f))
		return l.constRaw(elem, nil, b)
	}
	l.fail("a float scalar of %s", elem)
	return val{}
}

// --- MIL values ---

func (l *lowerer) resolve(name string) string {
	for {
		a, ok := l.alias[name]
		if !ok {
			return name
		}
		name = a
	}
}

func (l *lowerer) named(name string) val {
	name = l.resolve(name)
	if v, ok := l.vals[name]; ok {
		return v
	}
	c, ok := l.consts[name]
	if !ok {
		l.fail("value %s is not defined", name)
	}
	v := l.constant(c)
	l.vals[name] = v
	return v
}

func (l *lowerer) binding(b *milspec.Argument_Binding) val {
	if n := b.GetName(); n != "" {
		return l.named(n)
	}
	return l.constant(b.GetValue())
}

func (l *lowerer) in(op *milspec.Operation, key string) val {
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

func opName(op *milspec.Operation) string {
	if len(op.GetOutputs()) > 0 {
		return op.GetOutputs()[0].GetName()
	}
	return "?"
}

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

func (l *lowerer) ints(op *milspec.Operation, key string, def ...int) []int {
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
	out := make([]int, len(tv.GetInts().GetValues()))
	for i, v := range tv.GetInts().GetValues() {
		out[i] = int(v)
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
	case elem == "float16" && len(raw) == 2:
		return mlpackage.Float16Value(binary.LittleEndian.Uint16(raw))
	case elem == "float32" && len(raw) == 4:
		return math.Float32frombits(binary.LittleEndian.Uint32(raw))
	}
	l.fail("%s %s: input %s is not a float scalar", op.GetType(), opName(op), key)
	return 0
}

// decode returns a constant's aicode element type, shape and little endian bytes; fp32 becomes
// fp16 (the programs compute in fp16).
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
			return "float16", shape, data
		case mlpackage.BlobFloat32:
			return "float16", shape, f32ToF16(data)
		case mlpackage.BlobInt8:
			return "sint8", shape, data
		case mlpackage.BlobUInt8:
			return "uint8", shape, data
		case mlpackage.BlobInt4:
			return "sint4", shape, data
		case mlpackage.BlobUInt4:
			return "uint4", shape, data
		case mlpackage.BlobUInt6:
			return "uint6", shape, data
		case mlpackage.BlobInt32:
			return "sint32", shape, data
		}
		l.fail("weight blobs of type %d are not supported", bt)
	}
	tv := c.GetImmediateValue().GetTensor()
	if tv == nil {
		l.fail("unsupported constant %v", c)
	}
	switch dt {
	case milspec.DataType_FLOAT16:
		return "float16", shape, tv.GetBytes().GetValues()
	case milspec.DataType_FLOAT32:
		f := tv.GetFloats().GetValues()
		b := make([]byte, 4*len(f))
		for i, v := range f {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(v))
		}
		return "float16", shape, f32ToF16(b)
	case milspec.DataType_INT32:
		v := tv.GetInts().GetValues()
		b := make([]byte, 4*len(v))
		for i, x := range v {
			binary.LittleEndian.PutUint32(b[4*i:], uint32(x))
		}
		return "sint32", shape, b
	case milspec.DataType_UINT8, milspec.DataType_UINT4, milspec.DataType_INT8:
		return map[milspec.DataType]string{milspec.DataType_UINT8: "uint8", milspec.DataType_UINT4: "uint4", milspec.DataType_INT8: "sint8"}[dt], shape, tv.GetBytes().GetValues()
	case milspec.DataType_BOOL:
		v := tv.GetBools().GetValues()
		b := make([]byte, len(v))
		for i, x := range v {
			if x {
				b[i] = 1
			}
		}
		return "bool", shape, b
	}
	l.fail("constants of type %v are not supported", dt)
	return
}

func (l *lowerer) constant(c *milspec.Value) val {
	elem, shape, raw := l.decode(c)
	return l.constRaw(elem, shape, raw)
}

func f32ToF16(b []byte) []byte {
	out := make([]byte, len(b)/2)
	for i := 0; i < len(b)/4; i++ {
		binary.LittleEndian.PutUint16(out[2*i:], mlpackage.Float16Bits(math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))))
	}
	return out
}
