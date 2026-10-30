// Package milcpu runs ML Programs on the CPU in float32, as a reference for hwnpu's converters.
// It covers the operations that internal/mlpackage's builder emits, not the whole MIL spec, and
// favors being obviously correct over speed (only the matrix products are parallel).
//
// The point is to check that a converter's graph computes what the PyTorch reference computes,
// without a Neural Engine or a Mac: a port that matches here in fp32 and not on the Neural Engine
// has a precision problem, not a structural one. With RoundF16 every operation's result is
// rounded to fp16, which approximates the storage precision of the Neural Engine (not its
// arithmetic).
package milcpu

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"google.golang.org/protobuf/proto"

	"github.com/shibukawa/hwnpu/internal/milspec"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/mlspec"
)

// Tensor is a float32 tensor in row major order. A nil Shape is a scalar.
type Tensor struct {
	Shape []int
	Data  []float32
}

// value is a program value: a float tensor, or the int, bool or string parameters of operations.
type value struct {
	shape []int
	f     []float32
	i     []int32
	b     []bool
	s     string
	str   bool
}

func (v *value) tensor() Tensor { return Tensor{Shape: v.shape, Data: v.f} }

// Program is a loaded ML Program function.
type Program struct {
	fn      *milspec.Function
	ops     []*milspec.Operation
	outputs []string
	weights io.ReaderAt
	// lastUse[name] is the index of the last operation reading name, to free values early.
	lastUse map[string]int
}

// Options change how Run computes.
type Options struct {
	// RoundF16 rounds every operation's result to fp16.
	RoundF16 bool
}

// Load prepares the main function of prog. weights is the weight file the program's blob values
// point into (nil if it has none).
func Load(prog *milspec.Program, weights io.ReaderAt) (*Program, error) {
	return LoadFunction(prog, "main", weights)
}

// LoadFunction prepares the named function of a program with several, as Load does main.
func LoadFunction(prog *milspec.Program, name string, weights io.ReaderAt) (*Program, error) {
	fn, ok := prog.GetFunctions()[name]
	if !ok {
		return nil, fmt.Errorf("milcpu: no function %s", name)
	}
	block, ok := fn.GetBlockSpecializations()[fn.GetOpset()]
	if !ok {
		return nil, fmt.Errorf("milcpu: no block for opset %s", fn.GetOpset())
	}
	p := &Program{fn: fn, ops: block.GetOperations(), outputs: block.GetOutputs(), weights: weights, lastUse: map[string]int{}}
	for i, op := range p.ops {
		for _, arg := range op.GetInputs() {
			for _, b := range arg.GetArguments() {
				if n := b.GetName(); n != "" {
					p.lastUse[n] = i
				}
			}
		}
	}
	for _, o := range p.outputs {
		p.lastUse[o] = len(p.ops)
	}
	return p, nil
}

// Inputs returns the shapes of the program's inputs by name.
func (p *Program) Inputs() map[string][]int {
	in := map[string][]int{}
	for _, v := range p.fn.GetInputs() {
		in[v.GetName()] = typeShape(v.GetType())
	}
	return in
}

// OpenPackage loads the program of an .mlpackage directory. Close the returned closer when done.
func OpenPackage(dir string) (*Program, io.Closer, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "Data", "com.apple.CoreML", "model.mlmodel"))
	if err != nil {
		return nil, nil, err
	}
	var m mlspec.Model
	if err := proto.Unmarshal(raw, &m); err != nil {
		return nil, nil, fmt.Errorf("milcpu: %s: %w", dir, err)
	}
	if m.GetMlProgram() == nil {
		return nil, nil, fmt.Errorf("milcpu: %s is not an ML Program", dir)
	}
	f, err := os.Open(filepath.Join(dir, "Data", "com.apple.CoreML", "weights", "weight.bin"))
	if err != nil {
		return nil, nil, err
	}
	p, err := Load(m.GetMlProgram(), f)
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	return p, f, nil
}

// Run evaluates the program on named inputs and returns its outputs.
func (p *Program) Run(inputs map[string]Tensor, opts Options) (map[string]Tensor, error) {
	vals := map[string]*value{}
	for _, in := range p.fn.GetInputs() {
		t, ok := inputs[in.GetName()]
		if !ok {
			return nil, fmt.Errorf("milcpu: input %s missing", in.GetName())
		}
		shape := typeShape(in.GetType())
		if numel(shape) != len(t.Data) {
			return nil, fmt.Errorf("milcpu: input %s has %d values, shape %v", in.GetName(), len(t.Data), shape)
		}
		vals[in.GetName()] = &value{shape: shape, f: t.Data}
	}
	for i, op := range p.ops {
		out, err := p.eval(op, vals)
		if err != nil {
			return nil, fmt.Errorf("milcpu: %s %s: %w", op.GetType(), opName(op), err)
		}
		o := op.GetOutputs()[0]
		if want := typeShape(o.GetType()); out.f != nil && !sameShape(want, out.shape) {
			return nil, fmt.Errorf("milcpu: %s %s: computed shape %v, declared %v", op.GetType(), o.GetName(), out.shape, want)
		}
		if opts.RoundF16 && out.f != nil && op.GetType() != "const" {
			for j, x := range out.f {
				out.f[j] = mlpackage.Float16Value(mlpackage.Float16Bits(x))
			}
		}
		vals[o.GetName()] = out
		for _, arg := range op.GetInputs() {
			for _, b := range arg.GetArguments() {
				if n := b.GetName(); n != "" && p.lastUse[n] == i {
					delete(vals, n)
				}
			}
		}
	}
	res := map[string]Tensor{}
	for _, o := range p.outputs {
		v, ok := vals[o]
		if !ok || v.f == nil {
			return nil, fmt.Errorf("milcpu: output %s not computed", o)
		}
		res[o] = v.tensor()
	}
	return res, nil
}

func opName(op *milspec.Operation) string {
	if outs := op.GetOutputs(); len(outs) > 0 {
		return outs[0].GetName()
	}
	return ""
}

func typeShape(t *milspec.ValueType) []int {
	tt := t.GetTensorType()
	if tt == nil || tt.GetRank() == 0 {
		return nil
	}
	shape := make([]int, len(tt.GetDimensions()))
	for i, d := range tt.GetDimensions() {
		shape[i] = int(d.GetConstant().GetSize())
	}
	return shape
}

func numel(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func sameShape(a, b []int) bool {
	if numel(a) != numel(b) || len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// decode turns a constant Value (immediate or in the weight file) into a value.
func (p *Program) decode(v *milspec.Value) (*value, error) {
	shape := typeShape(v.GetType())
	dt := v.GetType().GetTensorType().GetDataType()
	n := numel(shape)
	if bf := v.GetBlobFileValue(); bf != nil {
		if p.weights == nil {
			return nil, fmt.Errorf("blob value without a weight file")
		}
		bt, raw, err := mlpackage.ReadBlob(p.weights, bf.GetOffset())
		if err != nil {
			return nil, err
		}
		return decodeRaw(bt, dt, raw, shape, n)
	}
	tv := v.GetImmediateValue().GetTensor()
	if tv == nil {
		return nil, fmt.Errorf("unsupported value %v", v)
	}
	out := &value{shape: shape}
	switch {
	case dt == milspec.DataType_STRING:
		out.str = true
		if s := tv.GetStrings().GetValues(); len(s) > 0 {
			out.s = s[0]
		}
	case dt == milspec.DataType_FLOAT16:
		raw := tv.GetBytes().GetValues()
		out.f = make([]float32, len(raw)/2)
		for i := range out.f {
			out.f[i] = mlpackage.Float16Value(binary.LittleEndian.Uint16(raw[2*i:]))
		}
	case dt == milspec.DataType_FLOAT32:
		out.f = append([]float32(nil), tv.GetFloats().GetValues()...)
	case dt == milspec.DataType_INT32:
		out.i = append([]int32(nil), tv.GetInts().GetValues()...)
	case dt == milspec.DataType_BOOL:
		out.b = append([]bool(nil), tv.GetBools().GetValues()...)
	default:
		return nil, fmt.Errorf("immediate values of type %v are not supported", dt)
	}
	return out, nil
}

// decodeRaw decodes blob data. Integer weight data (the indices and quantized values of the
// constexpr operations) is kept in i.
func decodeRaw(bt mlpackage.BlobDType, dt milspec.DataType, raw []byte, shape []int, n int) (*value, error) {
	out := &value{shape: shape}
	switch bt {
	case mlpackage.BlobFloat16:
		out.f = make([]float32, len(raw)/2)
		for i := range out.f {
			out.f[i] = mlpackage.Float16Value(binary.LittleEndian.Uint16(raw[2*i:]))
		}
	case mlpackage.BlobFloat32:
		out.f = make([]float32, len(raw)/4)
		for i := range out.f {
			out.f[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	case mlpackage.BlobInt32:
		out.i = make([]int32, len(raw)/4)
		for i := range out.i {
			out.i[i] = int32(binary.LittleEndian.Uint32(raw[4*i:]))
		}
	case mlpackage.BlobUInt8:
		if dt != milspec.DataType_BOOL {
			// uint8 data: the indices of an 8 bit table
			out.i = make([]int32, len(raw))
			for i, c := range raw {
				out.i[i] = int32(c)
			}
			break
		}
		out.b = make([]bool, len(raw))
		for i, c := range raw {
			out.b[i] = c != 0
		}
	case mlpackage.BlobUInt6:
		// packed 6 bits at a time, low bits first
		out.i = make([]int32, n)
		for i := range out.i {
			bit := 6 * i
			v := uint32(raw[bit/8])
			if bit/8+1 < len(raw) {
				v |= uint32(raw[bit/8+1]) << 8
			}
			out.i[i] = int32(v>>(bit%8)) & 63
		}
	case mlpackage.BlobInt8:
		out.i = make([]int32, len(raw))
		for i, c := range raw {
			out.i[i] = int32(int8(c))
		}
	case mlpackage.BlobUInt4, mlpackage.BlobInt4:
		out.i = make([]int32, n)
		for i := range out.i {
			nib := int32(raw[i/2]>>(4*(i%2))) & 15
			if bt == mlpackage.BlobInt4 && nib >= 8 {
				nib -= 16
			}
			out.i[i] = nib
		}
	default:
		return nil, fmt.Errorf("blob type %d is not supported", bt)
	}
	if out.f != nil && len(out.f) != n || out.i != nil && len(out.i) != n {
		return nil, fmt.Errorf("blob holds %d values for shape %v", max(len(out.f), len(out.i)), shape)
	}
	return out, nil
}
