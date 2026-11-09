package tflite

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"slices"

	"github.com/shibukawa/hwnpu/internal/onnx"
)

// The builtin operators the converter takes (schema.fbs's BuiltinOperator).
const (
	opAdd             = 0
	opConcatenation   = 2
	opEmbeddingLookup = 7
	opFullyConnected  = 9
	opLogistic        = 14
	opMul             = 18
	opRelu            = 19
	opReshape         = 22
	opSoftmax         = 25
	opTanh            = 28
	opGather          = 36
	opTranspose       = 39
	opMean            = 40
	opSub             = 41
	opDiv             = 42
	opSqueeze         = 43
	opStridedSlice    = 45
	opExp             = 47
	opCast            = 53
	opMaximum         = 55
	opArgMax          = 56
	opMinimum         = 57
	opLess            = 58
	opNeg             = 59
	opGreater         = 61
	opGreaterEqual    = 62
	opLessEqual       = 63
	opSelect          = 64
	opSlice           = 65
	opSin             = 66
	opCos             = 108
	opCumsum          = 128
	opExpandDims      = 70
	opEqual           = 71
	opLog             = 73
	opSum             = 74
	opSqrt            = 75
	opRsqrt           = 76
	opPow             = 78
	opArgMin          = 79
	opReduceMax       = 82
	opPack            = 83
	opOneHot          = 85
	opLogicalAnd      = 86
	opSquare          = 92
	opSquaredDiff     = 99
	opAbs             = 101
	opWhile           = 119
	opSelectV2        = 123
	opBatchMatMul     = 126
	opGelu            = 150
	opPad             = 34
	opComplexAbs      = 135
	opRFFT2D          = 131
)

// tensor types (schema.fbs's TensorType) as onnx's element types
var types = map[int]int{8: onnx.Complex64, 0: onnx.Float, 1: onnx.Float16, 2: onnx.Int32, 3: onnx.Uint8, 4: onnx.Int64, 6: onnx.Bool, 9: onnx.Int8, 10: onnx.Double}

// Load reads a .tflite file and converts its main subgraph (and the subgraphs its loops run) to
// an onnx.Graph: the tensors are named by their index ("t12"), the graph's inputs and outputs keep
// theirs as ValueInfo names mapped by Names.
func Load(path string) (*Model, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m, err := Parse(b)
	if err != nil {
		return nil, fmt.Errorf("tflite: %s: %w", path, err)
	}
	return m, nil
}

// Model is a converted model: its main graph and the TFLite names of its inputs and outputs.
type Model struct {
	Graph        *onnx.Graph
	Inputs       []string // the main graph's inputs as TFLite names them, in order
	Outputs      []string
	InputValues  []string // and as the graph names them ("t3")
	OutputValues []string
}

// Parse converts a TFLite model.
func Parse(b []byte) (m *Model, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("malformed model: %v", r)
		}
	}()
	root := table{b, u32(b, 0)}
	var codes []int
	for _, oc := range root.tables(1) {
		c := oc.int32(3, 0)
		if dep := oc.uint8(0, 0); c < dep {
			c = dep
		}
		codes = append(codes, c)
	}
	buffers := root.tables(4)
	subgraphs := root.tables(2)
	if len(subgraphs) == 0 {
		return nil, fmt.Errorf("no subgraphs")
	}
	c := &converter{b: b, codes: codes, buffers: buffers, subgraphs: subgraphs, graphs: map[int]*onnx.Graph{}}
	g, err := c.graph(0)
	if err != nil {
		return nil, err
	}
	sg := subgraphs[0]
	tensors := sg.tables(0)
	m = &Model{Graph: g}
	for _, i := range sg.ints(1) {
		m.Inputs = append(m.Inputs, tensors[i].string(3))
		m.InputValues = append(m.InputValues, name(i))
	}
	for _, i := range sg.ints(2) {
		m.Outputs = append(m.Outputs, tensors[i].string(3))
		m.OutputValues = append(m.OutputValues, name(i))
	}
	return m, nil
}

func name(i int) string { return fmt.Sprintf("t%d", i) }

type converter struct {
	b         []byte
	codes     []int
	buffers   []table
	subgraphs []table
	graphs    map[int]*onnx.Graph
}

// graph converts subgraph k.
func (c *converter) graph(k int) (*onnx.Graph, error) {
	if g, ok := c.graphs[k]; ok {
		return g, nil
	}
	sg := c.subgraphs[k]
	tensors := sg.tables(0)
	g := &onnx.Graph{Init: map[string]*onnx.Tensor{}}
	c.graphs[k] = g
	shapes := make([][]int, len(tensors))
	ttypes := make([]int, len(tensors))
	for i, t := range tensors {
		shapes[i] = t.ints(0)
		ttypes[i] = types[t.uint8(1, 0)]
		buf := t.int32(2, 0)
		if buf > 0 && buf < len(c.buffers) {
			data := c.buffers[buf].bytes(0)
			if off := c.buffers[buf].uint64(1, 0); len(data) == 0 && off > 1 {
				size := c.buffers[buf].uint64(2, 0)
				data = c.b[off : off+size]
			}
			if len(data) > 0 {
				v, err := decode(data, t.uint8(1, 0), shapes[i])
				if err != nil {
					return nil, fmt.Errorf("tensor %d (%s): %w", i, t.string(3), err)
				}
				g.Init[name(i)] = v
			}
		}
	}
	for _, i := range sg.ints(1) {
		g.Inputs = append(g.Inputs, onnx.ValueInfo{Name: name(i), Type: ttypes[i], Shape: shapes[i]})
	}
	for _, i := range sg.ints(2) {
		g.Outputs = append(g.Outputs, onnx.ValueInfo{Name: name(i), Type: ttypes[i], Shape: shapes[i]})
	}
	for oi, op := range sg.tables(3) {
		code := c.codes[op.int32(0, 0)]
		var ins, outs []string
		for _, i := range op.ints(1) {
			if i < 0 {
				ins = append(ins, "")
			} else {
				ins = append(ins, name(i))
			}
		}
		for _, i := range op.ints(2) {
			outs = append(outs, name(i))
		}
		opts, _ := op.sub(4)
		nodes, err := c.node(g, code, opts, ins, outs, fmt.Sprintf("op%d_%d", k, oi), shapes, op.ints(1))
		if err != nil {
			return nil, fmt.Errorf("subgraph %d op %d (builtin %d): %w", k, oi, code, err)
		}
		g.Nodes = append(g.Nodes, nodes...)
	}
	return g, nil
}

func decode(data []byte, typ int, shape []int) (*onnx.Tensor, error) {
	t := &onnx.Tensor{Shape: shape, Type: types[typ]}
	n := onnx.Len(shape)
	size := map[int]int{0: 4, 1: 2, 2: 4, 3: 1, 4: 8, 6: 1, 9: 1, 10: 8}[typ]
	if size == 0 {
		return nil, fmt.Errorf("type %d", typ)
	}
	if len(data) < n*size {
		return nil, fmt.Errorf("%d bytes for %v of type %d", len(data), shape, typ)
	}
	t.F = make([]float64, n)
	for i := range t.F {
		p := data[i*size:]
		switch typ {
		case 0:
			t.F[i] = float64(math.Float32frombits(binary.LittleEndian.Uint32(p)))
		case 2:
			t.F[i] = float64(int32(binary.LittleEndian.Uint32(p)))
		case 4:
			t.F[i] = float64(int64(binary.LittleEndian.Uint64(p)))
		case 3, 6:
			t.F[i] = float64(p[0])
		case 9:
			t.F[i] = float64(int8(p[0]))
		case 10:
			t.F[i] = math.Float64frombits(binary.LittleEndian.Uint64(p))
		case 1:
			t.F[i] = f16(binary.LittleEndian.Uint16(p))
		}
	}
	return t, nil
}

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
		return sign * math.Inf(1)
	}
	return sign * (1 + m/1024) * math.Pow(2, float64(e-15))
}

func ints64(v ...int) []int64 {
	out := make([]int64, len(v))
	for i, x := range v {
		out[i] = int64(x)
	}
	return out
}

// node converts one operator to onnx nodes.
func (c *converter) node(g *onnx.Graph, code int, opts table, ins, outs []string, nm string, shapes [][]int, idx []int) ([]*onnx.Node, error) {
	n := func(op string, in, out []string, attrs map[string]onnx.Attr) *onnx.Node {
		if attrs == nil {
			attrs = map[string]onnx.Attr{}
		}
		return &onnx.Node{Op: op, Name: nm, In: in, Out: out, Attrs: attrs}
	}
	hasOpts := opts.b != nil
	act := func(i int) int {
		if !hasOpts {
			return 0
		}
		return opts.int8(i)
	}
	// a fused activation after the node: the node writes tmp, the activation out
	fused := func(node *onnx.Node, a int) ([]*onnx.Node, error) {
		switch a {
		case 0:
			return []*onnx.Node{node}, nil
		case 1:
			tmp := nm + "_pre"
			out := node.Out[0]
			node.Out = []string{tmp}
			return []*onnx.Node{node, n("Relu", []string{tmp}, []string{out}, nil)}, nil
		case 4:
			tmp := nm + "_pre"
			out := node.Out[0]
			node.Out = []string{tmp}
			return []*onnx.Node{node, n("Tanh", []string{tmp}, []string{out}, nil)}, nil
		}
		return nil, fmt.Errorf("fused activation %d", a)
	}
	rank := func(i int) int { return len(shapes[idx[i]]) }
	switch code {
	case opAdd, opSub, opMul, opDiv:
		op := map[int]string{opAdd: "Add", opSub: "Sub", opMul: "Mul", opDiv: "Div"}[code]
		return fused(n(op, ins, outs, nil), act(0))
	case opMaximum, opMinimum:
		return []*onnx.Node{n(map[int]string{opMaximum: "Max", opMinimum: "Min"}[code], ins, outs, nil)}, nil
	case opSquaredDiff:
		d := nm + "_d"
		return []*onnx.Node{n("Sub", ins, []string{d}, nil), n("Mul", []string{d, d}, outs, nil)}, nil
	case opSquare:
		return []*onnx.Node{n("Mul", []string{ins[0], ins[0]}, outs, nil)}, nil
	case opPow:
		return []*onnx.Node{n("Pow", ins, outs, nil)}, nil
	case opLogistic, opTanh, opExp, opLog, opSqrt, opRsqrt, opAbs, opNeg, opRelu:
		op := map[int]string{opLogistic: "Sigmoid", opTanh: "Tanh", opExp: "Exp", opLog: "Log", opSqrt: "Sqrt", opRsqrt: "Rsqrt", opAbs: "Abs", opNeg: "Neg", opRelu: "Relu"}[code]
		return []*onnx.Node{n(op, ins, outs, nil)}, nil
	case opGelu:
		return []*onnx.Node{n("Gelu", ins, outs, map[string]onnx.Attr{"approximate": {I: int64(opts.uint8(0, 0))}})}, nil
	case opLess, opGreater, opGreaterEqual, opLessEqual, opEqual, opLogicalAnd:
		op := map[int]string{opLess: "Less", opGreater: "Greater", opGreaterEqual: "GreaterOrEqual", opLessEqual: "LessOrEqual", opEqual: "Equal", opLogicalAnd: "And"}[code]
		return []*onnx.Node{n(op, ins, outs, nil)}, nil
	case opSelect, opSelectV2:
		return []*onnx.Node{n("Where", ins, outs, nil)}, nil
	case opCast:
		return []*onnx.Node{n("Identity", ins[:1], outs, nil)}, nil
	case opSoftmax:
		if beta := opts.float32(0, 1); hasOpts && beta != 1 {
			return nil, fmt.Errorf("softmax beta %g", beta)
		}
		return []*onnx.Node{n("Softmax", ins, outs, map[string]onnx.Attr{"axis": {I: -1}})}, nil
	case opReshape:
		if len(ins) > 1 && ins[1] != "" {
			return []*onnx.Node{n("Reshape", ins[:2], outs, map[string]onnx.Attr{"allowzero": {I: 1}})}, nil
		}
		shape := opts.ints(0)
		cst := nm + "_shape"
		g.Init[cst] = &onnx.Tensor{Shape: []int{len(shape)}, F: toF(shape), Type: onnx.Int64}
		return []*onnx.Node{n("Reshape", []string{ins[0], cst}, outs, map[string]onnx.Attr{"allowzero": {I: 1}})}, nil
	case opSqueeze:
		dims := opts.ints(0)
		cst := nm + "_axes"
		g.Init[cst] = &onnx.Tensor{Shape: []int{len(dims)}, F: toF(dims), Type: onnx.Int64}
		if len(dims) == 0 {
			return []*onnx.Node{n("Squeeze", ins[:1], outs, nil)}, nil
		}
		return []*onnx.Node{n("Squeeze", []string{ins[0], cst}, outs, nil)}, nil
	case opExpandDims:
		return []*onnx.Node{n("Unsqueeze", ins, outs, nil)}, nil
	case opTranspose:
		p, ok := g.Init[ins[1]]
		if !ok {
			return nil, fmt.Errorf("transpose by a computed permutation")
		}
		return []*onnx.Node{n("Transpose", ins[:1], outs, map[string]onnx.Attr{"perm": {Ints: ints64(p.Ints()...)}})}, nil
	case opSum, opMean, opReduceMax:
		op := map[int]string{opSum: "ReduceSum", opMean: "ReduceMean", opReduceMax: "ReduceMax"}[code]
		keep := int64(0)
		if hasOpts && opts.bool(0) {
			keep = 1
		}
		return []*onnx.Node{n(op, ins, outs, map[string]onnx.Attr{"keepdims": {I: keep}})}, nil
	case opConcatenation:
		return fused(n("Concat", ins, outs, map[string]onnx.Attr{"axis": {I: int64(opts.int32(0, 0))}}), act(1))
	case opEmbeddingLookup:
		return []*onnx.Node{n("Gather", []string{ins[1], ins[0]}, outs, map[string]onnx.Attr{"axis": {I: 0}})}, nil
	case opGather:
		if hasOpts && opts.int32(1, 0) != 0 {
			return nil, fmt.Errorf("gather with batch_dims")
		}
		return []*onnx.Node{n("Gather", ins, outs, map[string]onnx.Attr{"axis": {I: int64(opts.int32(0, 0))}})}, nil
	case opFullyConnected:
		// y = x (..., in) W(out, in)ᵀ + b; without keep_num_dims, x is flattened to (-1, in)
		var nodes []*onnx.Node
		wt := nm + "_wT"
		var in int
		if w, ok := g.Init[ins[1]]; ok && len(w.Shape) == 2 {
			in = w.Shape[1]
			g.Init[wt] = transpose2(w)
		} else if ws := shapes[idx[1]]; len(ws) == 2 {
			// a computed weight: transposed on the way
			in = ws[1]
			nodes = append(nodes, n("Transpose", ins[1:2], []string{wt}, map[string]onnx.Attr{"perm": {Ints: []int64{1, 0}}}))
		} else {
			return nil, fmt.Errorf("a fully connected layer of a weight of shape %v", ws)
		}
		x := ins[0]
		keep := hasOpts && opts.bool(2)
		if !keep || rank(0) != 2 {
			if !keep {
				shp := nm + "_flat"
				g.Init[shp] = &onnx.Tensor{Shape: []int{2}, F: []float64{-1, float64(in)}, Type: onnx.Int64}
				nodes = append(nodes, n("Reshape", []string{x, shp}, []string{nm + "_x"}, map[string]onnx.Attr{"allowzero": {I: 1}}))
				x = nm + "_x"
			}
		}
		y := outs[0]
		hasBias := len(ins) > 2 && ins[2] != ""
		mm := y
		if hasBias {
			mm = nm + "_mm"
		}
		nodes = append(nodes, n("MatMul", []string{x, wt}, []string{mm}, nil))
		if hasBias {
			nodes = append(nodes, n("Add", []string{mm, ins[2]}, []string{y}, nil))
		}
		last := nodes[len(nodes)-1]
		rest, err := fused(last, act(0))
		if err != nil {
			return nil, err
		}
		return append(nodes[:len(nodes)-1], rest...), nil
	case opBatchMatMul:
		a, bname := ins[0], ins[1]
		var nodes []*onnx.Node
		tr := func(v string, r int, tag string) string {
			perm := make([]int, r)
			for i := range perm {
				perm[i] = i
			}
			perm[r-1], perm[r-2] = perm[r-2], perm[r-1]
			out := nm + tag
			nodes = append(nodes, n("Transpose", []string{v}, []string{out}, map[string]onnx.Attr{"perm": {Ints: ints64(perm...)}}))
			return out
		}
		if opts.bool(0) {
			a = tr(a, rank(0), "_a")
		}
		if opts.bool(1) {
			bname = tr(bname, rank(1), "_b")
		}
		return append(nodes, n("MatMul", []string{a, bname}, outs, nil)), nil
	case opStridedSlice:
		attrs := map[string]onnx.Attr{}
		for i, k := range []string{"begin_mask", "end_mask", "ellipsis_mask", "new_axis_mask", "shrink_axis_mask"} {
			attrs[k] = onnx.Attr{I: int64(opts.int32(i, 0))}
		}
		return []*onnx.Node{n("StridedSlice", ins, outs, attrs)}, nil
	case opSlice:
		return []*onnx.Node{n("TFLSlice", ins, outs, nil)}, nil
	case opSin, opCos:
		return []*onnx.Node{n(map[int]string{opSin: "Sin", opCos: "Cos"}[code], ins, outs, nil)}, nil
	case opCumsum:
		attrs := map[string]onnx.Attr{"exclusive": {I: int64(opts.uint8(0, 0))}, "reverse": {I: int64(opts.uint8(1, 0))}}
		return []*onnx.Node{n("CumSum", ins, outs, attrs)}, nil
	case opArgMax, opArgMin:
		return []*onnx.Node{n(map[int]string{opArgMax: "TFLArgMax", opArgMin: "TFLArgMin"}[code], ins, outs, nil)}, nil
	case opOneHot:
		return []*onnx.Node{n("OneHot", ins, outs, map[string]onnx.Attr{"axis": {I: int64(opts.int32(0, -1))}})}, nil
	case opPack:
		return []*onnx.Node{n("Pack", ins, outs, map[string]onnx.Attr{"axis": {I: int64(opts.int32(1, 0))}})}, nil
	case opPad:
		p, ok := g.Init[ins[1]]
		if !ok {
			return nil, fmt.Errorf("pad by computed paddings")
		}
		pv := p.Ints()
		r := len(pv) / 2
		pads := make([]int, 2*r)
		for i := range r {
			pads[i], pads[r+i] = pv[2*i], pv[2*i+1]
		}
		cst := nm + "_pads"
		g.Init[cst] = &onnx.Tensor{Shape: []int{2 * r}, F: toF(pads), Type: onnx.Int64}
		return []*onnx.Node{n("Pad", []string{ins[0], cst}, outs, nil)}, nil
	case opRFFT2D:
		return []*onnx.Node{n("TFLRFFT2D", ins, outs, nil)}, nil
	case opComplexAbs:
		return []*onnx.Node{n("TFLComplexAbs", ins, outs, nil)}, nil
	case opWhile:
		cond, err := c.graph(opts.int32(0, 0))
		if err != nil {
			return nil, err
		}
		body, err := c.graph(opts.int32(1, 0))
		if err != nil {
			return nil, err
		}
		node := n("While", ins, outs, nil)
		node.Graphs = []*onnx.Graph{cond, body}
		return []*onnx.Node{node}, nil
	}
	return nil, fmt.Errorf("builtin operator %d is not supported", code)
}

func (t table) int8(i int) int { return int(int8(t.uint8(i, 0))) }

func toF(v []int) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = float64(x)
	}
	return out
}

func transpose2(w *onnx.Tensor) *onnx.Tensor {
	r, c := w.Shape[0], w.Shape[1]
	out := &onnx.Tensor{Shape: []int{c, r}, F: make([]float64, r*c), Type: w.Type}
	for i := range r {
		for j := range c {
			out.F[j*r+i] = w.F[i*c+j]
		}
	}
	return out
}

var _ = slices.Clone[[]int]
