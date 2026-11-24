// Package milcpu runs ML Programs on the CPU in float32, as a reference for bonsaiai's converters.
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
	"strings"
	"sync"

	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"

	"google.golang.org/protobuf/proto"

	"github.com/shibukawa/bonsaiai/internal/milspec"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
	"github.com/shibukawa/bonsaiai/internal/mlspec"
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
	// q is integer weight data of 8 bits or fewer (the indices and quantized values of the
	// constexpr operations), a byte each, signed (int8, int4) when qSigned: not 4 bytes each in i
	q       []byte
	qSigned bool
	// lut is a weight kept as its 8 bit table (constexpr_lut_to_dense that only linear reads):
	// f is nil, linear reads it as it is
	lut *lutWeight
	// f16 is a weight kept in fp16 (a const that only linear reads): f is nil, linear converts it
	// as it packs it
	f16 *f16Weight
	b   []bool
	s   string
	str bool
}

// ints are the integer values of v: i, or q's.
func (v *value) ints() []int32 {
	if v.i != nil || v.q == nil {
		return v.i
	}
	out := make([]int32, len(v.q))
	for k, c := range v.q {
		if v.qSigned {
			out[k] = int32(int8(c))
		} else {
			out[k] = int32(c)
		}
	}
	return out
}

// qAt is v's integer value k (i's or q's).
func (v *value) qAt(k int) int32 {
	if v.q == nil {
		return v.i[k]
	}
	if v.qSigned {
		return int32(int8(v.q[k]))
	}
	return int32(v.q[k])
}

// qLen is how many integer values v holds.
func (v *value) qLen() int {
	if v.q != nil {
		return len(v.q)
	}
	return len(v.i)
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
	// consts are the const operations' values, decoded on the first run (no operation writes
	// into its inputs; rounding to fp16 in place leaves fp16 weights as they are)
	consts []*value
	// pool keeps the buffers of the values for the next ones (pool.go)
	pool pool
	// reluAfter[i]: operation i is a convolution whose value only the ReLU right after it
	// reads, which it then computes itself (the ReLU passing its value on); reluNext tells
	// eval so for the operation it runs
	reluAfter []bool
	reluNext  bool
	// inline are the operations' inline constants, decoded at their first use
	inline map[*milspec.Value]*value
	// constOut are the values of the const operations (and constexpr_*), which stay
	constOut map[string]bool
	// lutLinear are the outputs of constexpr_lut_to_dense that only linears read as their
	// weight: kept as their tables (lutWeight), not in float32; f16Linear those of fp16 consts,
	// kept in fp16 (f16Weight)
	lutLinear, f16Linear map[string]bool
	mu                   sync.Mutex
}

// Options change how Run computes.
type Options struct {
	// RoundF16 rounds every operation's result to fp16.
	RoundF16 bool
	// FlushSubnormals, with RoundF16, rounds fp16 subnormals (under 6.1e-5) to zero, as the
	// Neural Engine may.
	FlushSubnormals bool
	// Trace, when set, is called with each operation's type, output name and largest absolute
	// value (before RoundF16), to find what overflows fp16 (65504).
	Trace func(typ, name string, maxAbs, rms float32)
	// Value, when set, is called with each operation's output name and values (after
	// RoundF16), to compare a value inside the program (aicheck's BONSAIAI_DEV=ailower.cut).
	Value func(name string, data []float32)
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
	p.consts = make([]*value, len(p.ops))
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
	p.constOut = map[string]bool{}
	for _, op := range p.ops {
		if t := op.GetType(); t == "const" || strings.HasPrefix(t, "constexpr_") {
			p.constOut[op.GetOutputs()[0].GetName()] = true
		}
	}
	p.lutLinear = linearWeights(p.ops, p.outputs, func(op *milspec.Operation) bool {
		return op.GetType() == "constexpr_lut_to_dense"
	})
	p.f16Linear = linearWeights(p.ops, p.outputs, func(op *milspec.Operation) bool {
		v := op.GetAttributes()["val"]
		return op.GetType() == "const" && v.GetBlobFileValue() != nil && v.GetType().GetTensorType().GetDataType() == milspec.DataType_FLOAT16
	})
	p.reluAfter = make([]bool, len(p.ops))
	for i := 0; i+1 < len(p.ops); i++ {
		o := p.ops[i].GetOutputs()[0].GetName()
		next := p.ops[i+1]
		if p.ops[i].GetType() == "conv" && next.GetType() == "relu" && p.lastUse[o] == i+1 && len(next.GetInputs()) == 1 {
			if a := next.GetInputs()["x"].GetArguments(); len(a) == 1 && a[0].GetName() == o {
				p.reluAfter[i] = true
			}
		}
	}
	return p, nil
}

// linearWeights are the outputs of the operations of ops that kind takes, of rank 2, that only
// linears read, as their weight (not an output of the function).
func linearWeights(ops []*milspec.Operation, outputs []string, kind func(*milspec.Operation) bool) map[string]bool {
	out := map[string]bool{}
	for _, op := range ops {
		if kind(op) && len(typeShape(op.GetOutputs()[0].GetType())) == 2 {
			out[op.GetOutputs()[0].GetName()] = true
		}
	}
	for _, o := range outputs {
		delete(out, o)
	}
	for _, op := range ops {
		for k, arg := range op.GetInputs() {
			for _, b := range arg.GetArguments() {
				if n := b.GetName(); out[n] && (op.GetType() != "linear" || k != "weight") {
					delete(out, n)
				}
			}
		}
	}
	return out
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
	p.mu.Lock()
	defer p.mu.Unlock()
	// a convolution and the ReLU after it in one pass, unless each value is looked at
	fuse := opts.Trace == nil && opts.Value == nil && !opts.RoundF16
	for i, op := range p.ops {
		var out *value
		var err error
		switch {
		case p.consts[i] != nil:
			out = p.consts[i]
		case fuse && i > 0 && p.reluAfter[i-1]:
			out = vals[op.GetInputs()["x"].GetArguments()[0].GetName()] // relu'd already
		default:
			p.reluNext = fuse && p.reluAfter[i]
			out, err = p.eval(op, vals)
			p.reluNext = false
			if err == nil && (op.GetType() == "const" || strings.HasPrefix(op.GetType(), "constexpr_")) {
				p.consts[i] = out
			}
		}
		if err != nil {
			return nil, fmt.Errorf("milcpu: %s %s: %w", op.GetType(), opName(op), err)
		}
		o := op.GetOutputs()[0]
		if want := typeShape(o.GetType()); out.f != nil && !sameShape(want, out.shape) {
			return nil, fmt.Errorf("milcpu: %s %s: computed shape %v, declared %v", op.GetType(), o.GetName(), out.shape, want)
		}
		if opts.Trace != nil && out.f != nil && op.GetType() != "const" {
			var m float32
			var ss float64
			for _, x := range out.f {
				ss += float64(x) * float64(x)
				if x < 0 {
					x = -x
				}
				if x > m || x != x {
					m = x
				}
			}
			opts.Trace(op.GetType(), o.GetName(), m, float32(math.Sqrt(ss/float64(max(1, len(out.f))))))
		}
		if opts.RoundF16 && out.f != nil && op.GetType() != "const" {
			for j, x := range out.f {
				v := mlpackage.Float16Value(mlpackage.Float16Bits(x))
				if opts.FlushSubnormals && v > -6.103515625e-05 && v < 6.103515625e-05 {
					v = 0
				}
				out.f[j] = v
			}
		}
		if opts.Value != nil && out.f != nil && op.GetType() != "const" {
			opts.Value(o.GetName(), out.f)
		}
		if p.consts[i] == nil {
			var from []*value
			for _, arg := range op.GetInputs() {
				for _, b := range arg.GetArguments() {
					if n := b.GetName(); n != "" {
						from = append(from, vals[n])
					}
				}
			}
			p.pool.hold(out.f, from)
		}
		vals[o.GetName()] = out
		for _, arg := range op.GetInputs() {
			for _, b := range arg.GetArguments() {
				if n := b.GetName(); n != "" && p.lastUse[n] == i {
					if v, ok := vals[n]; ok {
						delete(vals, n)
						p.pool.drop(v.f)
					}
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
	for _, v := range vals {
		p.pool.give(v.f) // the caller's now (the outputs, and what nothing read)
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
func (p *Program) decode(v *milspec.Value) (*value, error) { return p.decodeAs(v, false) }

// decodeAs is decode, a fp16 blob of rank 2 kept in fp16 (f16Weight) when asF16.
func (p *Program) decodeAs(v *milspec.Value, asF16 bool) (*value, error) {
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
		if asF16 && bt == mlpackage.BlobFloat16 && len(shape) == 2 && len(raw) == 2*n {
			return &value{shape: shape, f16: &f16Weight{n: shape[0], k: shape[1], h: raw}}, nil
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
			out.q = raw
			break
		}
		out.b = make([]bool, len(raw))
		for i, c := range raw {
			out.b[i] = c != 0
		}
	case mlpackage.BlobUInt6:
		// packed 6 bits at a time, low bits first
		out.q = make([]byte, n)
		for i := range out.q {
			bit := 6 * i
			v := uint32(raw[bit/8])
			if bit/8+1 < len(raw) {
				v |= uint32(raw[bit/8+1]) << 8
			}
			out.q[i] = byte(v>>(bit%8)) & 63
		}
	case mlpackage.BlobInt8:
		out.q, out.qSigned = raw, true
	case mlpackage.BlobUInt4, mlpackage.BlobInt4:
		out.q, out.qSigned = make([]byte, n), bt == mlpackage.BlobInt4
		for i := range out.q {
			nib := raw[i/2] >> (4 * (i % 2)) & 15
			if out.qSigned && nib >= 8 {
				nib |= 0xf0 // as int8
			}
			out.q[i] = nib
		}
	default:
		return nil, fmt.Errorf("blob type %d is not supported", bt)
	}
	if out.f != nil && len(out.f) != n || out.i != nil && len(out.i) != n || out.q != nil && len(out.q) != n {
		return nil, fmt.Errorf("blob holds %d values for shape %v", max(len(out.f), len(out.i), len(out.q)), shape)
	}
	return out, nil
}
