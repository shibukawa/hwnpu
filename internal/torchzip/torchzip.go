// Package torchzip reads the tensors of a PyTorch checkpoint saved by torch.save (a zip of a
// pickle, PREFIX/data.pkl, and the storages, PREFIX/data/KEY): the tensors anywhere in the
// pickled object (a state dict, or a dict holding one), by their dotted path. Only the pickle
// opcodes torch.save writes are run, and no Python class is built: a class the pickle names is
// a placeholder that keeps its arguments and its state, so a checkpoint can name classes of any
// package.
package torchzip

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"slices"
	"sort"
)

// Tensor is a tensor of the checkpoint.
type Tensor struct {
	Name   string
	DType  string // F32, F16, BF16, F64, I64, I32, I16, I8, U8, BOOL
	Shape  []int
	stride []int
	offset int // in elements of the storage
	key    string
}

// Len is the number of elements.
func (t Tensor) Len() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

// File is an open checkpoint.
type File struct {
	f       *os.File
	z       *zip.Reader
	prefix  string
	entries map[string]*zip.File
	tensors map[string]Tensor
	names   []string
	// Root is the unpickled object, with tensors as Tensor values, dicts as map[any]any and
	// lists and tuples as []any.
	Root any
}

// Open reads the pickle of the checkpoint at path.
func Open(p string) (*File, error) {
	f, err := os.Open(p)
	if err != nil {
		return nil, err
	}
	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	z, err := zip.NewReader(f, st.Size())
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("torchzip: %s: %w", p, err)
	}
	t := &File{f: f, z: z, entries: map[string]*zip.File{}, tensors: map[string]Tensor{}}
	var pkl *zip.File
	for _, e := range z.File {
		t.entries[e.Name] = e
		if path.Base(e.Name) == "data.pkl" && pkl == nil {
			pkl = e
			t.prefix = path.Dir(e.Name)
		}
	}
	if pkl == nil {
		f.Close()
		return nil, fmt.Errorf("torchzip: %s: no data.pkl", p)
	}
	r, err := pkl.Open()
	if err != nil {
		f.Close()
		return nil, err
	}
	data, err := io.ReadAll(r)
	r.Close()
	if err != nil {
		f.Close()
		return nil, err
	}
	root, err := unpickle(data)
	if err != nil {
		f.Close()
		return nil, fmt.Errorf("torchzip: %s: %w", p, err)
	}
	t.Root = root
	t.collect("", root)
	sort.Strings(t.names)
	return t, nil
}

// collect names the tensors under v by their dotted path (a state dict under "state_dict"
// keeps its keys: the path of a tensor is its key with the keys of the dicts holding it before).
func (t *File) collect(prefix string, v any) {
	switch v := v.(type) {
	case Tensor:
		v.Name = prefix
		t.tensors[prefix] = v
		t.names = append(t.names, prefix)
	case map[any]any:
		for k, x := range v {
			ks, ok := k.(string)
			if !ok {
				ks = fmt.Sprint(k)
			}
			if prefix != "" {
				ks = prefix + "." + ks
			}
			t.collect(ks, x)
		}
	case []any:
		for i, x := range v {
			ks := fmt.Sprint(i)
			if prefix != "" {
				ks = prefix + "." + ks
			}
			t.collect(ks, x)
		}
	}
}

// Close closes the file.
func (t *File) Close() error { return t.f.Close() }

// Names are the tensors' paths, sorted.
func (t *File) Names() []string { return slices.Clone(t.names) }

// Lookup finds a tensor by its path.
func (t *File) Lookup(name string) (Tensor, bool) {
	x, ok := t.tensors[name]
	return x, ok
}

var dtypeSize = map[string]int{"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}

// Float32 reads a tensor as float32, in row-major order of its shape.
func (t *File) Float32(name string) ([]float32, error) {
	x, ok := t.tensors[name]
	if !ok {
		return nil, fmt.Errorf("torchzip: no tensor %s", name)
	}
	e := t.entries[t.prefix+"/data/"+x.key]
	if e == nil {
		return nil, fmt.Errorf("torchzip: %s: no storage %s", name, x.key)
	}
	if e.Method != zip.Store {
		return nil, fmt.Errorf("torchzip: %s: storage %s is compressed", name, x.key)
	}
	off, err := e.DataOffset()
	if err != nil {
		return nil, err
	}
	size := dtypeSize[x.DType]
	n := x.Len()
	// the elements of the storage the tensor spans
	last := x.offset
	for i, d := range x.Shape {
		if d > 0 {
			last += (d - 1) * x.stride[i]
		}
	}
	if n == 0 {
		return nil, nil
	}
	span := last - x.offset + 1
	buf := make([]byte, span*size)
	if _, err := t.f.ReadAt(buf, off+int64(x.offset*size)); err != nil {
		return nil, fmt.Errorf("torchzip: %s: %w", name, err)
	}
	elem := func(i int) float32 {
		b := buf[i*size:]
		switch x.DType {
		case "F32":
			return math.Float32frombits(binary.LittleEndian.Uint32(b))
		case "F16":
			return f16(binary.LittleEndian.Uint16(b))
		case "BF16":
			return math.Float32frombits(uint32(binary.LittleEndian.Uint16(b)) << 16)
		case "F64":
			return float32(math.Float64frombits(binary.LittleEndian.Uint64(b)))
		case "I64":
			return float32(int64(binary.LittleEndian.Uint64(b)))
		case "I32":
			return float32(int32(binary.LittleEndian.Uint32(b)))
		case "I16":
			return float32(int16(binary.LittleEndian.Uint16(b)))
		case "I8":
			return float32(int8(b[0]))
		default:
			return float32(b[0])
		}
	}
	out := make([]float32, n)
	idx := make([]int, len(x.Shape))
	for i := range out {
		at := 0
		for a, j := range idx {
			at += j * x.stride[a]
		}
		out[i] = elem(at)
		for a := len(idx) - 1; a >= 0; a-- {
			idx[a]++
			if idx[a] < x.Shape[a] {
				break
			}
			idx[a] = 0
		}
	}
	return out, nil
}

func f16(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := int(h>>10) & 0x1f
	man := uint32(h & 0x3ff)
	switch {
	case exp == 0 && man == 0:
		return math.Float32frombits(sign)
	case exp == 0:
		f := float32(man) / 1024 * float32(math.Pow(2, -14))
		if sign != 0 {
			return -f
		}
		return f
	case exp == 31:
		return math.Float32frombits(sign | 0x7f800000 | man<<13)
	}
	return math.Float32frombits(sign | uint32(exp-15+127)<<23 | man<<13)
}

// Object is an instance of a class the pickle names, not built: its class, the arguments it was
// made with and the state it was given.
type Object struct {
	Class string
	Args  []any
	State any
}

type global struct{ module, name string }

type mark struct{}

var storageTypes = map[string]string{
	"FloatStorage": "F32", "HalfStorage": "F16", "BFloat16Storage": "BF16", "DoubleStorage": "F64",
	"LongStorage": "I64", "IntStorage": "I32", "ShortStorage": "I16", "CharStorage": "I8",
	"ByteStorage": "U8", "BoolStorage": "BOOL",
}

type storage struct {
	dtype, key string
}

// unpickle runs the pickle in data.
func unpickle(data []byte) (any, error) {
	var stack []any
	memo := map[int]any{}
	pos := 0
	need := func(n int) ([]byte, error) {
		if pos+n > len(data) {
			return nil, fmt.Errorf("pickle: truncated at %d", pos)
		}
		b := data[pos : pos+n]
		pos += n
		return b, nil
	}
	pop := func() any {
		v := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		return v
	}
	popMark := func() []any {
		for i := len(stack) - 1; i >= 0; i-- {
			if _, ok := stack[i].(mark); ok {
				items := slices.Clone(stack[i+1:])
				stack = stack[:i]
				return items
			}
		}
		return nil
	}
	line := func() string {
		i := pos
		for i < len(data) && data[i] != '\n' {
			i++
		}
		s := string(data[pos:i])
		pos = i + 1
		return s
	}
	for pos < len(data) {
		op := data[pos]
		pos++
		switch op {
		case 0x80: // PROTO
			pos++
		case 0x95: // FRAME
			pos += 8
		case '.': // STOP
			if len(stack) == 0 {
				return nil, fmt.Errorf("pickle: empty stack at STOP")
			}
			return pop(), nil
		case '}': // EMPTY_DICT
			stack = append(stack, map[any]any{})
		case ']', ')': // EMPTY_LIST, EMPTY_TUPLE
			stack = append(stack, []any{})
		case '(': // MARK
			stack = append(stack, mark{})
		case 'X': // BINUNICODE
			b, err := need(4)
			if err != nil {
				return nil, err
			}
			s, err := need(int(binary.LittleEndian.Uint32(b)))
			if err != nil {
				return nil, err
			}
			stack = append(stack, string(s))
		case 0x8c: // SHORT_BINUNICODE
			b, err := need(1)
			if err != nil {
				return nil, err
			}
			s, err := need(int(b[0]))
			if err != nil {
				return nil, err
			}
			stack = append(stack, string(s))
		case 0x8d: // BINUNICODE8
			b, err := need(8)
			if err != nil {
				return nil, err
			}
			s, err := need(int(binary.LittleEndian.Uint64(b)))
			if err != nil {
				return nil, err
			}
			stack = append(stack, string(s))
		case 'U', 'C': // SHORT_BINSTRING, SHORT_BINBYTES
			b, err := need(1)
			if err != nil {
				return nil, err
			}
			s, err := need(int(b[0]))
			if err != nil {
				return nil, err
			}
			stack = append(stack, string(s))
		case 'T', 'B': // BINSTRING, BINBYTES
			b, err := need(4)
			if err != nil {
				return nil, err
			}
			s, err := need(int(binary.LittleEndian.Uint32(b)))
			if err != nil {
				return nil, err
			}
			stack = append(stack, string(s))
		case 'J': // BININT
			b, err := need(4)
			if err != nil {
				return nil, err
			}
			stack = append(stack, int(int32(binary.LittleEndian.Uint32(b))))
		case 'K': // BININT1
			b, err := need(1)
			if err != nil {
				return nil, err
			}
			stack = append(stack, int(b[0]))
		case 'M': // BININT2
			b, err := need(2)
			if err != nil {
				return nil, err
			}
			stack = append(stack, int(binary.LittleEndian.Uint16(b)))
		case 0x8a: // LONG1
			b, err := need(1)
			if err != nil {
				return nil, err
			}
			v, err := need(int(b[0]))
			if err != nil {
				return nil, err
			}
			var x int64
			for i := len(v) - 1; i >= 0; i-- {
				x = x<<8 | int64(v[i])
			}
			if len(v) > 0 && len(v) < 8 && v[len(v)-1]&0x80 != 0 {
				x -= 1 << (8 * len(v))
			}
			stack = append(stack, int(x))
		case 'G': // BINFLOAT
			b, err := need(8)
			if err != nil {
				return nil, err
			}
			stack = append(stack, math.Float64frombits(binary.BigEndian.Uint64(b)))
		case 'N':
			stack = append(stack, nil)
		case 0x88:
			stack = append(stack, true)
		case 0x89:
			stack = append(stack, false)
		case 't': // TUPLE
			stack = append(stack, popMark())
		case 0x85: // TUPLE1
			a := pop()
			stack = append(stack, []any{a})
		case 0x86: // TUPLE2
			b, a := pop(), pop()
			stack = append(stack, []any{a, b})
		case 0x87: // TUPLE3
			c, b, a := pop(), pop(), pop()
			stack = append(stack, []any{a, b, c})
		case 'l': // LIST
			stack = append(stack, popMark())
		case 'd': // DICT
			items := popMark()
			m := map[any]any{}
			for i := 0; i+1 < len(items); i += 2 {
				m[key(items[i])] = items[i+1]
			}
			stack = append(stack, m)
		case 'c': // GLOBAL
			mod := line()
			name := line()
			stack = append(stack, global{mod, name})
		case 0x93: // STACK_GLOBAL
			name, mod := pop(), pop()
			stack = append(stack, global{fmt.Sprint(mod), fmt.Sprint(name)})
		case 'R': // REDUCE
			args, _ := pop().([]any)
			fn := pop()
			v, err := reduce(fn, args)
			if err != nil {
				return nil, err
			}
			stack = append(stack, v)
		case 0x81: // NEWOBJ
			args, _ := pop().([]any)
			cls := pop()
			v, err := reduce(cls, args)
			if err != nil {
				return nil, err
			}
			stack = append(stack, v)
		case 'b': // BUILD
			state := pop()
			switch o := stack[len(stack)-1].(type) {
			case *Object:
				o.State = state
			case map[any]any:
				if m, ok := state.(map[any]any); ok {
					for k, v := range m {
						o[k] = v
					}
				}
			}
		case 's': // SETITEM
			v, k := pop(), pop()
			if m, ok := stack[len(stack)-1].(map[any]any); ok {
				m[key(k)] = v
			}
		case 'u': // SETITEMS
			items := popMark()
			if m, ok := stack[len(stack)-1].(map[any]any); ok {
				for i := 0; i+1 < len(items); i += 2 {
					m[key(items[i])] = items[i+1]
				}
			}
		case 'a': // APPEND
			v := pop()
			if l, ok := stack[len(stack)-1].([]any); ok {
				stack[len(stack)-1] = append(l, v)
			}
		case 'e': // APPENDS
			items := popMark()
			if l, ok := stack[len(stack)-1].([]any); ok {
				stack[len(stack)-1] = append(l, items...)
			}
		case 'q': // BINPUT
			b, err := need(1)
			if err != nil {
				return nil, err
			}
			memo[int(b[0])] = stack[len(stack)-1]
		case 'r': // LONG_BINPUT
			b, err := need(4)
			if err != nil {
				return nil, err
			}
			memo[int(binary.LittleEndian.Uint32(b))] = stack[len(stack)-1]
		case 0x94: // MEMOIZE
			memo[len(memo)] = stack[len(stack)-1]
		case 'h': // BINGET
			b, err := need(1)
			if err != nil {
				return nil, err
			}
			stack = append(stack, memo[int(b[0])])
		case 'j': // LONG_BINGET
			b, err := need(4)
			if err != nil {
				return nil, err
			}
			stack = append(stack, memo[int(binary.LittleEndian.Uint32(b))])
		case 'Q': // BINPERSID
			pid, _ := pop().([]any)
			if len(pid) < 3 || pid[0] != "storage" {
				return nil, fmt.Errorf("pickle: persistent id %v is not a storage", pid)
			}
			cls, _ := pid[1].(global)
			dt, ok := storageTypes[cls.name]
			if !ok {
				return nil, fmt.Errorf("pickle: storage type %s.%s", cls.module, cls.name)
			}
			stack = append(stack, storage{dtype: dt, key: fmt.Sprint(pid[2])})
		case '0': // POP
			pop()
		case '1': // POP_MARK
			popMark()
		case '2': // DUP
			stack = append(stack, stack[len(stack)-1])
		case 0x8f: // EMPTY_SET
			stack = append(stack, []any{})
		case 0x90: // ADDITEMS
			items := popMark()
			if l, ok := stack[len(stack)-1].([]any); ok {
				stack[len(stack)-1] = append(l, items...)
			}
		case 0x91: // FROZENSET
			stack = append(stack, popMark())
		default:
			return nil, fmt.Errorf("pickle: opcode 0x%02x at %d", op, pos-1)
		}
	}
	return nil, fmt.Errorf("pickle: no STOP")
}

// key is a dict key as a comparable value.
func key(k any) any {
	switch k := k.(type) {
	case []any, map[any]any:
		return fmt.Sprint(k)
	}
	return k
}

func ints(v any) ([]int, bool) {
	l, ok := v.([]any)
	if !ok {
		return nil, false
	}
	out := make([]int, len(l))
	for i, x := range l {
		n, ok := x.(int)
		if !ok {
			return nil, false
		}
		out[i] = n
	}
	return out, true
}

// reduce calls fn with args: the tensor rebuilders and OrderedDict are run, other classes are
// kept as Objects.
func reduce(fn any, args []any) (any, error) {
	g, ok := fn.(global)
	if !ok {
		return &Object{Class: fmt.Sprint(fn), Args: args}, nil
	}
	full := g.module + "." + g.name
	switch {
	case full == "torch._utils._rebuild_tensor_v2" || full == "torch._utils._rebuild_tensor":
		if len(args) < 4 {
			return nil, fmt.Errorf("pickle: %s with %d arguments", full, len(args))
		}
		st, ok := args[0].(storage)
		if !ok {
			return nil, fmt.Errorf("pickle: %s of %T", full, args[0])
		}
		off, _ := args[1].(int)
		shape, ok1 := ints(args[2])
		stride, ok2 := ints(args[3])
		if !ok1 || !ok2 {
			return nil, fmt.Errorf("pickle: %s: shape %v, stride %v", full, args[2], args[3])
		}
		return Tensor{DType: st.dtype, Shape: shape, stride: stride, offset: off, key: st.key}, nil
	case full == "torch._utils._rebuild_parameter" || full == "torch._utils._rebuild_parameter_with_state":
		if len(args) == 0 {
			return nil, fmt.Errorf("pickle: %s without arguments", full)
		}
		return args[0], nil
	case full == "collections.OrderedDict":
		m := map[any]any{}
		if len(args) > 0 {
			if l, ok := args[0].([]any); ok {
				for _, kv := range l {
					if p, ok := kv.([]any); ok && len(p) == 2 {
						m[key(p[0])] = p[1]
					}
				}
			}
		}
		return m, nil
	}
	return &Object{Class: full, Args: args}, nil
}
