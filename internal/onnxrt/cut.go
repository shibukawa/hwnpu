package onnxrt

import (
	"context"
	"fmt"

	"google.golang.org/protobuf/encoding/protowire"

	"github.com/shibukawa/hwnpu/internal/onnx"
)

// The fields of onnx.proto Cut reads and writes.
const (
	graphNode     = 1 // GraphProto.node
	nodeInput     = 1 // NodeProto.input
	nodeOutput    = 2 // NodeProto.output
	valueName     = 1 // ValueInfoProto.name
	tensorElem    = 1 // TypeProto.Tensor.elem_type
	elemTypeFloat = 1 // TensorProto.FLOAT
)

// Cut is the model's graph from the values inputs (float tensors of any shape, given in place
// of what computes them) to the values outputs: the nodes between them alone, the graph's
// inputs those of inputs, its outputs outputs, and no value_info. A cut from the graph's own
// inputs to values inside it is an onnx.Program's (Graph.Cut); from values inside it, what the
// host computes before a program (Basic Pitch's CQT) is left out.
func Cut(model []byte, inputs, outputs []string) ([]byte, error) {
	return editField(model, modelGraph, func(graph []byte) ([]byte, error) {
		type node struct {
			field    []byte
			ins, out []string
		}
		var nodes []node
		var rest []byte
		for b := graph; len(b) > 0; {
			num, typ, n := protowire.ConsumeTag(b)
			if n < 0 {
				return nil, protowire.ParseError(n)
			}
			m := protowire.ConsumeFieldValue(num, typ, b[n:])
			if m < 0 {
				return nil, protowire.ParseError(m)
			}
			field := b[:n+m]
			b = b[n+m:]
			switch {
			case num == graphNode && typ == protowire.BytesType:
				v, _ := protowire.ConsumeBytes(field[n:])
				nd := node{field: field}
				for c := v; len(c) > 0; {
					fn, ft, k := protowire.ConsumeTag(c)
					if k < 0 {
						return nil, protowire.ParseError(k)
					}
					l := protowire.ConsumeFieldValue(fn, ft, c[k:])
					if l < 0 {
						return nil, protowire.ParseError(l)
					}
					if ft == protowire.BytesType && (fn == nodeInput || fn == nodeOutput) {
						str, _ := protowire.ConsumeBytes(c[k:])
						if fn == nodeInput {
							nd.ins = append(nd.ins, string(str))
						} else {
							nd.out = append(nd.out, string(str))
						}
					}
					c = c[k+l:]
				}
				nodes = append(nodes, nd)
			case num == graphInput || num == graphOutput || num == graphValueInfo:
			default:
				rest = append(rest, field...)
			}
		}
		stop := map[string]bool{}
		for _, in := range inputs {
			stop[in] = true
		}
		producer := map[string]int{}
		for i, nd := range nodes {
			for _, o := range nd.out {
				producer[o] = i
			}
		}
		keep := make([]bool, len(nodes))
		work := append([]string(nil), outputs...)
		for len(work) > 0 {
			v := work[len(work)-1]
			work = work[:len(work)-1]
			i, ok := producer[v]
			if stop[v] || !ok || keep[i] {
				continue
			}
			keep[i] = true
			work = append(work, nodes[i].ins...)
		}
		var out []byte
		for i, nd := range nodes {
			if keep[i] {
				out = append(out, nd.field...)
			}
		}
		out = append(out, rest...)
		for _, name := range inputs {
			out = appendValueInfo(out, graphInput, name)
		}
		for _, name := range outputs {
			out = appendValueInfo(out, graphOutput, name)
		}
		return out, nil
	})
}

// appendValueInfo appends a graph's field num (an input or an output): a ValueInfoProto of
// name, a float tensor of any shape.
func appendValueInfo(out []byte, num protowire.Number, name string) []byte {
	var tt []byte
	tt = protowire.AppendTag(tt, tensorElem, protowire.VarintType)
	tt = protowire.AppendVarint(tt, elemTypeFloat)
	var t []byte
	t = protowire.AppendTag(t, typeTensor, protowire.BytesType)
	t = protowire.AppendBytes(t, tt)
	var v []byte
	v = protowire.AppendTag(v, valueName, protowire.BytesType)
	v = protowire.AppendString(v, name)
	v = protowire.AppendTag(v, valueType, protowire.BytesType)
	v = protowire.AppendBytes(v, t)
	out = protowire.AppendTag(out, num, protowire.BytesType)
	return protowire.AppendBytes(out, v)
}

// Program runs an onnx.Program's cut of an ONNX graph (its inputs, of their fixed shapes, to its
// outputs) with ONNX Runtime on the CPU, as runner.Program runs a converted one: Run's y is laid
// out as the converted program's (Flat: each output size / rows columns of every row; else its
// one output), so the code that reads a Core ML program's y reads it.
type Program struct {
	s      *Session
	p      *onnx.Program
	names  map[string]string // the runner's input name (x, ...) to the graph's
	shapes [][]int           // the outputs', learned from a first run
	out    []int
}

// OpenProgram loads the ONNX file's bytes for p (whose G is the same model's graph) into ONNX
// Runtime on threads threads (0 or 1: the calling thread alone).
func OpenProgram(model []byte, p *onnx.Program, threads int) (*Program, error) {
	if !p.Flat && len(p.Outputs) != 1 {
		return nil, fmt.Errorf("onnxrt: a program of %d outputs that is not Flat", len(p.Outputs))
	}
	rt, err := Load()
	if err != nil {
		return nil, err
	}
	var ins []string
	for _, in := range p.Inputs {
		ins = append(ins, in.Name)
	}
	b, err := Cut(model, ins, p.Outputs)
	if err != nil {
		return nil, err
	}
	s, err := rt.Open(b, Options{Threads: threads})
	if err != nil {
		return nil, err
	}
	q := &Program{s: s, p: p, names: map[string]string{}}
	for i, in := range p.Inputs {
		name := "x"
		if i > 0 {
			if name = p.Names[in.Name]; name == "" {
				name = in.Name
			}
		}
		q.names[name] = in.Name
	}
	// a first run on zeros gives the outputs' shapes
	zeros := map[string][]float32{}
	for name, shape := range q.Inputs() {
		zeros[name] = make([]float32, onnx.Len(shape))
	}
	if _, err := q.Run(context.Background(), zeros); err != nil {
		s.Close()
		return nil, err
	}
	return q, nil
}

// Inputs are the program's inputs and their shapes, under the runner's names (x first).
func (q *Program) Inputs() map[string][]int {
	out := map[string][]int{}
	for name, g := range q.names {
		for _, in := range q.p.Inputs {
			if in.Name == g {
				out[name] = in.Shape
			}
		}
	}
	return out
}

// Output is y's shape: (1, rows, columns) for a Flat program, else the output's own.
func (q *Program) Output() []int { return q.out }

// Shapes are the outputs' shapes (after OpenProgram).
func (q *Program) Shapes() [][]int { return q.shapes }

func (q *Program) Metadata() map[string]string { return nil }

// OnNPU is false: the program runs on the CPU.
func (q *Program) OnNPU() bool { return false }

func (q *Program) Close() error { return q.s.Close() }

// Run computes y from the inputs, by the runner's names.
func (q *Program) Run(ctx context.Context, in map[string][]float32) ([]float32, error) {
	ins := map[string]Tensor{}
	for name, g := range q.names {
		x, ok := in[name]
		if !ok {
			return nil, fmt.Errorf("onnxrt: no input %s", name)
		}
		var shape []int64
		for _, in := range q.p.Inputs {
			if in.Name == g {
				for _, d := range in.Shape {
					shape = append(shape, int64(d))
				}
			}
		}
		ins[g] = Tensor{Shape: shape, Data: x}
	}
	outs, err := q.s.Run(ins)
	if err != nil {
		return nil, err
	}
	shapes := make([][]int, len(q.p.Outputs))
	data := make([][]float32, len(q.p.Outputs))
	for i, name := range q.p.Outputs {
		t, ok := outs[name]
		if !ok {
			return nil, fmt.Errorf("onnxrt: no output %s", name)
		}
		for _, d := range t.Shape {
			shapes[i] = append(shapes[i], int(d))
		}
		data[i] = t.Data
	}
	q.shapes = shapes
	if !q.p.Flat {
		q.out = shapes[0]
		return data[0], nil
	}
	rows := onnx.FlatRows(shapes)
	width := 0
	for _, d := range data {
		width += len(d) / rows
	}
	y := make([]float32, rows*width)
	at := 0
	for _, d := range data {
		w := len(d) / rows
		for r := range rows {
			copy(y[r*width+at:][:w], d[r*w:(r+1)*w])
		}
		at += w
	}
	q.out = []int{1, rows, width}
	return y, nil
}
