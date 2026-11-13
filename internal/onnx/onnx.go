// Package onnx reads ONNX models without dependencies (the protobuf wire format through
// protowire), evaluates graphs on the host (Eval: the shapes a graph computes, and the small
// dynamic parts of a model such as top-k and non-maximum suppression), and lowers the static
// parts of a graph to MIL programs for the Neural Engine and Intel's NPU (Program).
//
// It is for models distributed as ONNX only (RTMO, Basic Pitch): their graphs are the source of
// truth, and the converter keeps to their computation op by op.
package onnx

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"google.golang.org/protobuf/encoding/protowire"
)

// Element types (TensorProto.DataType).
const (
	Float   = 1
	Uint8   = 2
	Int8    = 3
	Int32   = 6
	Int64   = 7
	Bool    = 9
	Float16 = 10
	Double  = 11
	// Complex64 values are kept as (real, imaginary) pairs in F, twice the shape's length.
	Complex64 = 14
)

// Tensor is a value: its shape and its elements as float64, whatever its element type (integers
// up to 2^53 are exact; larger ones, such as the int64 maximum of an open slice, are only large).
type Tensor struct {
	Shape []int
	F     []float64
	// Type is the element type: Float, Double, Int64, Int32, Bool...
	Type int
}

// Len is the number of elements of a shape.
func Len(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// IsInt reports whether the tensor's type is an integer or bool type.
func (t *Tensor) IsInt() bool {
	return t.Type != Float && t.Type != Double && t.Type != Float16
}

// Ints returns the elements as ints.
func (t *Tensor) Ints() []int {
	out := make([]int, len(t.F))
	for i, v := range t.F {
		switch {
		case v > math.MaxInt64/2:
			out[i] = math.MaxInt64 / 2
		case v < math.MinInt64/2:
			out[i] = math.MinInt64 / 2
		default:
			out[i] = int(v)
		}
	}
	return out
}

// Attr is a node attribute.
type Attr struct {
	F      float64
	I      int64
	S      string
	T      *Tensor
	Floats []float64
	Ints   []int64
	// G is a graph attribute's (If's branches).
	G *Graph
	// Kind is AttributeProto.type: 1 FLOAT, 2 INT, 3 STRING, 4 TENSOR, 5 GRAPH, 6 FLOATS, 7 INTS.
	Kind int
}

// Node is a node of a graph.
type Node struct {
	Op, Name, Domain string
	In, Out          []string
	Attrs            map[string]Attr
	// Graphs are the subgraphs of a loop (While: its condition and its body) or of an If (its
	// then and else branches; the outer values they read are the If's inputs after the
	// condition).
	Graphs []*Graph
}

// Int returns an int attribute, or def.
func (n *Node) Int(name string, def int64) int64 {
	if a, ok := n.Attrs[name]; ok {
		return a.I
	}
	return def
}

// Float returns a float attribute, or def.
func (n *Node) Float(name string, def float64) float64 {
	if a, ok := n.Attrs[name]; ok {
		return a.F
	}
	return def
}

// IntList returns an ints attribute, or nil.
func (n *Node) IntList(name string) []int {
	a, ok := n.Attrs[name]
	if !ok {
		return nil
	}
	out := make([]int, len(a.Ints))
	for i, v := range a.Ints {
		out[i] = int(v)
	}
	return out
}

// Str returns a string attribute, or def.
func (n *Node) Str(name, def string) string {
	if a, ok := n.Attrs[name]; ok {
		return a.S
	}
	return def
}

// ValueInfo is a graph input or output: its name, element type and shape (-1 for a dimension
// given by name, or not given).
type ValueInfo struct {
	Name  string
	Type  int
	Shape []int
}

// Graph is a model's main graph.
type Graph struct {
	Nodes   []*Node
	Init    map[string]*Tensor
	Inputs  []ValueInfo
	Outputs []ValueInfo
	// Opset is the default domain's opset version.
	Opset int
}

// Load reads an ONNX model file.
func Load(path string) (*Graph, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	g, err := Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("onnx: %s: %w", path, err)
	}
	return g, nil
}

// Parse reads an ONNX model (ModelProto).
func Parse(raw []byte) (*Graph, error) {
	g := &Graph{Init: map[string]*Tensor{}}
	var graph []byte
	err := each(raw, func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error {
		switch {
		case num == 7 && typ == protowire.BytesType:
			graph = b
		case num == 8 && typ == protowire.BytesType: // opset_import
			var domain string
			var version int
			err := each(b, func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error {
				switch num {
				case 1:
					domain = string(b)
				case 2:
					version = int(v)
				}
				return nil
			})
			if err != nil {
				return err
			}
			if domain == "" || domain == "ai.onnx" {
				g.Opset = version
			}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if graph == nil {
		return nil, fmt.Errorf("no graph")
	}
	if err := parseGraph(graph, g); err != nil {
		return nil, err
	}
	return g, nil
}

// parseGraph reads a GraphProto into g.
func parseGraph(graph []byte, g *Graph) error {
	err := each(graph, func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error {
		if typ != protowire.BytesType {
			return nil
		}
		switch num {
		case 1:
			n, err := parseNode(b)
			if err != nil {
				return err
			}
			g.Nodes = append(g.Nodes, n)
		case 5:
			name, t, err := parseTensor(b)
			if err != nil {
				return err
			}
			g.Init[name] = t
		case 11, 12:
			vi, err := parseValueInfo(b)
			if err != nil {
				return err
			}
			if num == 11 {
				g.Inputs = append(g.Inputs, vi)
			} else {
				g.Outputs = append(g.Outputs, vi)
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	// initializers may also be listed as inputs (older exporters)
	inputs := g.Inputs[:0]
	for _, in := range g.Inputs {
		if _, ok := g.Init[in.Name]; !ok {
			inputs = append(inputs, in)
		}
	}
	g.Inputs = inputs
	return nil
}

// outer are the values a graph's nodes read that neither its nodes, its initializers nor its
// inputs give: a branch's reads of the graph around it.
func (g *Graph) outer() []string {
	have := map[string]bool{"": true}
	for k := range g.Init {
		have[k] = true
	}
	for _, in := range g.Inputs {
		have[in.Name] = true
	}
	var out []string
	for _, n := range g.Nodes {
		for _, x := range n.In {
			if !have[x] {
				have[x] = true
				out = append(out, x)
			}
		}
		for _, sub := range n.Graphs {
			for _, x := range sub.outer() {
				if !have[x] {
					have[x] = true
					out = append(out, x)
				}
			}
		}
		for _, o := range n.Out {
			have[o] = true
		}
	}
	return out
}

// each calls f with every field of a message: the bytes of a length-delimited field, the value
// of a varint, or the bits of a fixed 32 or 64 bit field (in v).
func each(msg []byte, f func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error) error {
	for len(msg) > 0 {
		num, typ, n := protowire.ConsumeTag(msg)
		if n < 0 {
			return protowire.ParseError(n)
		}
		msg = msg[n:]
		var b []byte
		var v uint64
		switch typ {
		case protowire.BytesType:
			b, n = protowire.ConsumeBytes(msg)
		case protowire.VarintType:
			v, n = protowire.ConsumeVarint(msg)
		case protowire.Fixed32Type:
			var x uint32
			x, n = protowire.ConsumeFixed32(msg)
			v = uint64(x)
		case protowire.Fixed64Type:
			v, n = protowire.ConsumeFixed64(msg)
		default:
			n = protowire.ConsumeFieldValue(num, typ, msg)
		}
		if n < 0 {
			return protowire.ParseError(n)
		}
		msg = msg[n:]
		if err := f(num, typ, b, v); err != nil {
			return err
		}
	}
	return nil
}

func parseNode(msg []byte) (*Node, error) {
	n := &Node{Attrs: map[string]Attr{}}
	err := each(msg, func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error {
		switch num {
		case 1:
			n.In = append(n.In, string(b))
		case 2:
			n.Out = append(n.Out, string(b))
		case 3:
			n.Name = string(b)
		case 4:
			n.Op = string(b)
		case 7:
			n.Domain = string(b)
		case 5:
			name, a, err := parseAttr(b)
			if err != nil {
				return err
			}
			n.Attrs[name] = a
		}
		return nil
	})
	if err == nil && n.Op == "If" {
		th, el := n.Attrs["then_branch"].G, n.Attrs["else_branch"].G
		if th == nil || el == nil {
			return nil, fmt.Errorf("If %s without its branches", n.Name)
		}
		n.Graphs = []*Graph{th, el}
		seen := map[string]bool{}
		for _, b := range n.Graphs {
			for _, x := range b.outer() {
				if !seen[x] {
					seen[x] = true
					n.In = append(n.In, x)
				}
			}
		}
	}
	return n, err
}

func parseAttr(msg []byte) (string, Attr, error) {
	var a Attr
	var name string
	err := each(msg, func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error {
		switch num {
		case 1:
			name = string(b)
		case 2:
			a.F = float64(math.Float32frombits(uint32(v)))
		case 3:
			a.I = int64(v)
		case 4:
			a.S = string(b)
		case 5:
			_, t, err := parseTensor(b)
			if err != nil {
				return err
			}
			a.T = t
		case 6:
			a.G = &Graph{Init: map[string]*Tensor{}}
			return parseGraph(b, a.G)
		case 7: // floats, packed or one by one
			if typ == protowire.BytesType {
				for i := 0; i+4 <= len(b); i += 4 {
					a.Floats = append(a.Floats, float64(math.Float32frombits(binary.LittleEndian.Uint32(b[i:]))))
				}
			} else {
				a.Floats = append(a.Floats, float64(math.Float32frombits(uint32(v))))
			}
		case 8: // ints
			if typ == protowire.BytesType {
				for p := b; len(p) > 0; {
					x, k := protowire.ConsumeVarint(p)
					if k < 0 {
						return protowire.ParseError(k)
					}
					a.Ints = append(a.Ints, int64(x))
					p = p[k:]
				}
			} else {
				a.Ints = append(a.Ints, int64(v))
			}
		case 20:
			a.Kind = int(v)
		}
		return nil
	})
	return name, a, err
}

func parseValueInfo(msg []byte) (ValueInfo, error) {
	var vi ValueInfo
	err := each(msg, func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error {
		switch num {
		case 1:
			vi.Name = string(b)
		case 2: // TypeProto
			return each(b, func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error {
				if num != 1 { // tensor_type
					return nil
				}
				return each(b, func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error {
					switch num {
					case 1:
						vi.Type = int(v)
					case 2: // shape
						vi.Shape = []int{}
						return each(b, func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error {
							if num != 1 {
								return nil
							}
							d := -1
							err := each(b, func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error {
								if num == 1 {
									d = int(int64(v))
								}
								return nil
							})
							vi.Shape = append(vi.Shape, d)
							return err
						})
					}
					return nil
				})
			})
		}
		return nil
	})
	return vi, err
}

func parseTensor(msg []byte) (string, *Tensor, error) {
	t := &Tensor{}
	var name string
	var raw []byte
	var floats, doubles []float64
	var ints []int64
	err := each(msg, func(num protowire.Number, typ protowire.Type, b []byte, v uint64) error {
		switch num {
		case 1:
			if typ == protowire.BytesType {
				for p := b; len(p) > 0; {
					x, k := protowire.ConsumeVarint(p)
					if k < 0 {
						return protowire.ParseError(k)
					}
					t.Shape = append(t.Shape, int(int64(x)))
					p = p[k:]
				}
			} else {
				t.Shape = append(t.Shape, int(int64(v)))
			}
		case 2:
			t.Type = int(v)
		case 8:
			name = string(b)
		case 9:
			raw = b
		case 4: // float_data
			if typ == protowire.BytesType {
				for i := 0; i+4 <= len(b); i += 4 {
					floats = append(floats, float64(math.Float32frombits(binary.LittleEndian.Uint32(b[i:]))))
				}
			} else {
				floats = append(floats, float64(math.Float32frombits(uint32(v))))
			}
		case 10: // double_data
			if typ == protowire.BytesType {
				for i := 0; i+8 <= len(b); i += 8 {
					doubles = append(doubles, math.Float64frombits(binary.LittleEndian.Uint64(b[i:])))
				}
			} else {
				doubles = append(doubles, math.Float64frombits(v))
			}
		case 5, 7: // int32_data (also bool, int8, uint8, float16 bits), int64_data
			if typ == protowire.BytesType {
				for p := b; len(p) > 0; {
					x, k := protowire.ConsumeVarint(p)
					if k < 0 {
						return protowire.ParseError(k)
					}
					ints = append(ints, int64(x))
					p = p[k:]
				}
			} else {
				ints = append(ints, int64(v))
			}
		case 13:
			return fmt.Errorf("tensor %s: external data is not supported", name)
		}
		return nil
	})
	if err != nil {
		return "", nil, err
	}
	count := Len(t.Shape)
	t.F = make([]float64, 0, count)
	switch {
	case raw != nil:
		size := map[int]int{Float: 4, Double: 8, Int64: 8, Int32: 4, Bool: 1, Uint8: 1, Int8: 1, Float16: 2}[t.Type]
		if size == 0 || len(raw) != size*count {
			return "", nil, fmt.Errorf("tensor %s: %d bytes of type %d for shape %v", name, len(raw), t.Type, t.Shape)
		}
		for i := range count {
			p := raw[i*size:]
			var x float64
			switch t.Type {
			case Float:
				x = float64(math.Float32frombits(binary.LittleEndian.Uint32(p)))
			case Double:
				x = math.Float64frombits(binary.LittleEndian.Uint64(p))
			case Int64:
				x = float64(int64(binary.LittleEndian.Uint64(p)))
			case Int32:
				x = float64(int32(binary.LittleEndian.Uint32(p)))
			case Bool, Uint8:
				x = float64(p[0])
			case Int8:
				x = float64(int8(p[0]))
			case Float16:
				x = f16(binary.LittleEndian.Uint16(p))
			}
			t.F = append(t.F, x)
		}
	case floats != nil:
		t.F = floats
	case doubles != nil:
		t.F = doubles
	default:
		for _, x := range ints {
			switch t.Type {
			case Int32, Int8:
				t.F = append(t.F, float64(int32(x)))
			case Float16:
				t.F = append(t.F, f16(uint16(x)))
			default:
				t.F = append(t.F, float64(x))
			}
		}
	}
	if len(t.F) != count {
		return "", nil, fmt.Errorf("tensor %s: %d values for shape %v", name, len(t.F), t.Shape)
	}
	return name, t, nil
}

// f16 is the value of IEEE half precision bits.
func f16(h uint16) float64 {
	sign := 1.0
	if h&0x8000 != 0 {
		sign = -1
	}
	e, m := int(h>>10&0x1f), float64(h&0x3ff)
	switch e {
	case 0:
		return sign * m * math.Pow(2, -24)
	case 31:
		if m == 0 {
			return sign * math.Inf(1)
		}
		return math.NaN()
	}
	return sign * (1 + m/1024) * math.Pow(2, float64(e-15))
}
