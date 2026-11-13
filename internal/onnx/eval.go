package onnx

import (
	"fmt"
	"math"
	"slices"
	"sort"

	"github.com/shibukawa/hwnpu/internal/sgemm"
)

// Needed returns the nodes of g that the outputs depend on, in the graph's (topological) order,
// stopping at the names in stop, which are given, and at the Shape of a value in shapes.
func (g *Graph) Needed(outputs []string, stop map[string]bool) []*Node {
	return g.needed(outputs, stop, nil)
}

func (g *Graph) needed(outputs []string, stop map[string]bool, shapes map[string][]int) []*Node {
	producer := map[string]int{}
	for i, n := range g.Nodes {
		for _, o := range n.Out {
			producer[o] = i
		}
	}
	need := map[int]bool{}
	var visit func(name string)
	visit = func(name string) {
		if stop[name] {
			return
		}
		i, ok := producer[name]
		if !ok || need[i] {
			return
		}
		need[i] = true
		n := g.Nodes[i]
		if n.Op == "Shape" && len(n.In) > 0 {
			if _, known := shapes[n.In[0]]; known {
				return
			}
		}
		for _, in := range n.In {
			if in != "" {
				visit(in)
			}
		}
	}
	for _, o := range outputs {
		visit(o)
	}
	var out []*Node
	for i, n := range g.Nodes {
		if need[i] {
			out = append(out, n)
		}
	}
	return out
}

// Eval runs the nodes the outputs depend on, on the host in float64, from the given values (the
// graph's inputs, or any of its values: the nodes before them are not run). It returns the
// outputs.
func (g *Graph) Eval(given map[string]*Tensor, outputs []string) (map[string]*Tensor, error) {
	return g.EvalShapes(given, nil, outputs)
}

// EvalShapes is Eval with the static shapes of values that are not given (a program's, Program.
// Shapes): the Shape of one is its shape, without running the nodes before it.
func (g *Graph) EvalShapes(given map[string]*Tensor, shapes map[string][]int, outputs []string) (map[string]*Tensor, error) {
	stop := map[string]bool{}
	vals := map[string]*Tensor{}
	for k, v := range given {
		stop[k], vals[k] = true, v
	}
	for k, v := range g.Init {
		if _, ok := vals[k]; !ok {
			vals[k] = v
		}
	}
	nodes := g.needed(outputs, stop, shapes)
	// last is the node reading a value last; a value made by a node is dropped after it, unless
	// it is an output, so that the host keeps only what is still to be read.
	last, keep := map[string]int{}, map[string]bool{}
	for i, n := range nodes {
		for _, name := range n.In {
			last[name] = i
		}
	}
	for _, o := range outputs {
		keep[o] = true
	}
	for k, n := range nodes {
		if n.Op == "Shape" && len(n.In) > 0 {
			if shape, ok := shapes[n.In[0]]; ok {
				if _, have := vals[n.In[0]]; !have {
					vals[n.In[0]] = &Tensor{Shape: shape, Type: Float} // its shape alone: Shape reads nothing else
				}
			}
		}
		ins := make([]*Tensor, len(n.In))
		for i, name := range n.In {
			if name == "" {
				continue
			}
			v, ok := vals[name]
			if !ok {
				return nil, fmt.Errorf("onnx: %s %s: no value for its input %s", n.Op, n.Name, name)
			}
			ins[i] = v
		}
		var outs []*Tensor
		var err error
		if n.Op == "If" {
			outs, err = evalIf(n, ins)
		} else {
			outs, err = EvalNode(n, ins)
		}
		if err != nil {
			return nil, fmt.Errorf("onnx: %s %s: %w", n.Op, n.Name, err)
		}
		for i, o := range n.Out {
			if i < len(outs) && o != "" {
				vals[o] = outs[i]
			}
		}
		for _, name := range n.In {
			if last[name] == k && !keep[name] && !stop[name] && g.Init[name] == nil {
				delete(vals, name)
			}
		}
	}
	out := map[string]*Tensor{}
	for _, o := range outputs {
		v, ok := vals[o]
		if !ok {
			return nil, fmt.Errorf("onnx: no value %s", o)
		}
		out[o] = v
	}
	return out, nil
}

func scalar(x float64, typ int) *Tensor { return &Tensor{Shape: []int{}, F: []float64{x}, Type: typ} }

// strides of a shape, row major.
func strides(shape []int) []int {
	s := make([]int, len(shape))
	n := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = n
		n *= shape[i]
	}
	return s
}

// Broadcast is numpy's broadcast of two shapes.
func Broadcast(a, b []int) ([]int, error) {
	n := max(len(a), len(b))
	out := make([]int, n)
	for i := range n {
		da, db := 1, 1
		if j := i - (n - len(a)); j >= 0 {
			da = a[j]
		}
		if j := i - (n - len(b)); j >= 0 {
			db = b[j]
		}
		switch {
		case da == db, db == 1:
			out[i] = da
		case da == 1:
			out[i] = db
		default:
			return nil, fmt.Errorf("shapes %v and %v do not broadcast", a, b)
		}
	}
	return out, nil
}

// index maps each element of the shape out to its element of x broadcast to it.
func index(x, out []int) []int {
	n := len(out)
	xs := strides(x)
	idx := make([]int, Len(out))
	coord := make([]int, n)
	for k := range idx {
		off := 0
		for i := range n {
			if j := i - (n - len(x)); j >= 0 && x[j] != 1 {
				off += coord[i] * xs[j]
			}
		}
		idx[k] = off
		for i := n - 1; i >= 0; i-- {
			coord[i]++
			if coord[i] < out[i] {
				break
			}
			coord[i] = 0
		}
	}
	return idx
}

func binop(a, b *Tensor, typ int, f func(x, y float64) float64) (*Tensor, error) {
	shape, err := Broadcast(a.Shape, b.Shape)
	if err != nil {
		return nil, err
	}
	out := &Tensor{Shape: shape, F: make([]float64, Len(shape)), Type: typ}
	// without a table of indices where an operand is the result's shape, or repeats whole (a
	// scalar, a bias over the last axes): the tables were a third of EfficientSAM's decoder
	ma, mb := repeats(a.Shape, shape), repeats(b.Shape, shape)
	switch {
	case ma == len(out.F) && mb == len(out.F):
		for k := range out.F {
			out.F[k] = f(a.F[k], b.F[k])
		}
	case ma == len(out.F) && mb > 0:
		for k := 0; k < len(out.F); k += mb {
			for j, y := range b.F[:mb] {
				out.F[k+j] = f(a.F[k+j], y)
			}
		}
	case mb == len(out.F) && ma > 0:
		for k := 0; k < len(out.F); k += ma {
			for j, x := range a.F[:ma] {
				out.F[k+j] = f(x, b.F[k+j])
			}
		}
	default:
		ia, ib := index(a.Shape, shape), index(b.Shape, shape)
		for k := range out.F {
			out.F[k] = f(a.F[ia[k]], b.F[ib[k]])
		}
	}
	return out, nil
}

// repeats is the length of x when x broadcast to out is x repeated whole, one copy after another
// (x is out's last axes, any axes before them 1 or absent; out itself counts), else 0.
func repeats(x, out []int) int {
	n := 1
	i, j := len(x)-1, len(out)-1
	for ; i >= 0 && x[i] == out[j]; i, j = i-1, j-1 {
		n *= x[i]
	}
	for ; i >= 0; i-- {
		if x[i] != 1 {
			return 0
		}
	}
	if n == 0 {
		return 0
	}
	return n
}

func unary(a *Tensor, f func(x float64) float64) *Tensor {
	out := &Tensor{Shape: slices.Clone(a.Shape), F: make([]float64, len(a.F)), Type: a.Type}
	for i, x := range a.F {
		out.F[i] = f(x)
	}
	return out
}

func bool2f(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// axis normalizes a possibly negative axis.
func axis(a, rank int) int {
	if a < 0 {
		a += rank
	}
	return a
}

// EvalNode runs one node.
func EvalNode(n *Node, in []*Tensor) ([]*Tensor, error) {
	one := func(t *Tensor, err error) ([]*Tensor, error) {
		if err != nil {
			return nil, err
		}
		return []*Tensor{t}, nil
	}
	arg := func(i int) *Tensor {
		if i < len(in) {
			return in[i]
		}
		return nil
	}
	switch n.Op {
	case "Identity":
		return []*Tensor{in[0]}, nil
	case "Constant":
		if a, ok := n.Attrs["value"]; ok && a.T != nil {
			return []*Tensor{a.T}, nil
		}
		if a, ok := n.Attrs["value_float"]; ok {
			return []*Tensor{scalar(a.F, Float)}, nil
		}
		if a, ok := n.Attrs["value_int"]; ok {
			return []*Tensor{scalar(float64(a.I), Int64)}, nil
		}
		if a, ok := n.Attrs["value_ints"]; ok {
			t := &Tensor{Shape: []int{len(a.Ints)}, Type: Int64}
			for _, v := range a.Ints {
				t.F = append(t.F, float64(v))
			}
			return []*Tensor{t}, nil
		}
		return nil, fmt.Errorf("a constant of no value")
	case "Add":
		return one(binop(in[0], in[1], in[0].Type, func(x, y float64) float64 { return x + y }))
	case "Sub":
		return one(binop(in[0], in[1], in[0].Type, func(x, y float64) float64 { return x - y }))
	case "Mul":
		return one(binop(in[0], in[1], in[0].Type, func(x, y float64) float64 { return x * y }))
	case "Div":
		if in[0].IsInt() {
			return one(binop(in[0], in[1], in[0].Type, func(x, y float64) float64 { return math.Trunc(x / y) }))
		}
		return one(binop(in[0], in[1], in[0].Type, func(x, y float64) float64 { return x / y }))
	case "Pow":
		return one(binop(in[0], in[1], in[0].Type, math.Pow))
	case "Max":
		t := in[0]
		for _, u := range in[1:] {
			var err error
			if t, err = binop(t, u, t.Type, math.Max); err != nil {
				return nil, err
			}
		}
		return []*Tensor{t}, nil
	case "Min":
		t := in[0]
		for _, u := range in[1:] {
			var err error
			if t, err = binop(t, u, t.Type, math.Min); err != nil {
				return nil, err
			}
		}
		return []*Tensor{t}, nil
	case "Equal":
		return one(binop(in[0], in[1], Bool, func(x, y float64) float64 { return bool2f(x == y) }))
	case "Less":
		return one(binop(in[0], in[1], Bool, func(x, y float64) float64 { return bool2f(x < y) }))
	case "Greater":
		return one(binop(in[0], in[1], Bool, func(x, y float64) float64 { return bool2f(x > y) }))
	case "LessOrEqual":
		return one(binop(in[0], in[1], Bool, func(x, y float64) float64 { return bool2f(x <= y) }))
	case "GreaterOrEqual":
		return one(binop(in[0], in[1], Bool, func(x, y float64) float64 { return bool2f(x >= y) }))
	case "And":
		return one(binop(in[0], in[1], Bool, func(x, y float64) float64 { return bool2f(x != 0 && y != 0) }))
	case "Or":
		return one(binop(in[0], in[1], Bool, func(x, y float64) float64 { return bool2f(x != 0 || y != 0) }))
	case "Not":
		t := unary(in[0], func(x float64) float64 { return bool2f(x == 0) })
		t.Type = Bool
		return []*Tensor{t}, nil
	case "Sqrt":
		return []*Tensor{unary(in[0], math.Sqrt)}, nil
	case "Exp":
		return []*Tensor{unary(in[0], math.Exp)}, nil
	case "Log":
		return []*Tensor{unary(in[0], math.Log)}, nil
	case "Abs":
		return []*Tensor{unary(in[0], math.Abs)}, nil
	case "Neg":
		return []*Tensor{unary(in[0], func(x float64) float64 { return -x })}, nil
	case "Erf":
		return []*Tensor{unary(in[0], math.Erf)}, nil
	case "Cos":
		return []*Tensor{unary(in[0], math.Cos)}, nil
	case "Sin":
		return []*Tensor{unary(in[0], math.Sin)}, nil
	case "TFLRFFT2D":
		return one(rfft2d(in[0], in[1].Ints()))
	case "TFLComplexAbs":
		if in[0].Type != Complex64 {
			return nil, fmt.Errorf("complex abs of type %d", in[0].Type)
		}
		out := &Tensor{Shape: in[0].Shape, Type: Float, F: make([]float64, len(in[0].F)/2)}
		for i := range out.F {
			out.F[i] = math.Hypot(in[0].F[2*i], in[0].F[2*i+1])
		}
		return []*Tensor{out}, nil
	case "Floor":
		return []*Tensor{unary(in[0], math.Floor)}, nil
	case "Ceil":
		return []*Tensor{unary(in[0], math.Ceil)}, nil
	case "Round":
		return []*Tensor{unary(in[0], math.RoundToEven)}, nil
	case "Sigmoid":
		return []*Tensor{unary(in[0], func(x float64) float64 { return 1 / (1 + math.Exp(-x)) })}, nil
	case "Relu":
		return []*Tensor{unary(in[0], func(x float64) float64 { return math.Max(x, 0) })}, nil
	case "Tanh":
		return []*Tensor{unary(in[0], math.Tanh)}, nil
	case "Clip":
		lo, hi := math.Inf(-1), math.Inf(1)
		if a, ok := n.Attrs["min"]; ok {
			lo = a.F
		}
		if a, ok := n.Attrs["max"]; ok {
			hi = a.F
		}
		if t := arg(1); t != nil {
			lo = t.F[0]
		}
		if t := arg(2); t != nil {
			hi = t.F[0]
		}
		return []*Tensor{unary(in[0], func(x float64) float64 { return math.Min(math.Max(x, lo), hi) })}, nil
	case "Cast":
		to := int(n.Int("to", Float))
		t := unary(in[0], func(x float64) float64 {
			switch to {
			case Float:
				return float64(float32(x))
			case Double, Float16:
				return x
			case Bool:
				return bool2f(x != 0)
			default:
				return math.Trunc(x)
			}
		})
		t.Type = to
		return []*Tensor{t}, nil
	case "Where":
		shape, err := Broadcast(in[0].Shape, in[1].Shape)
		if err == nil {
			shape, err = Broadcast(shape, in[2].Shape)
		}
		if err != nil {
			return nil, err
		}
		ic, ia, ib := index(in[0].Shape, shape), index(in[1].Shape, shape), index(in[2].Shape, shape)
		out := &Tensor{Shape: shape, F: make([]float64, Len(shape)), Type: in[1].Type}
		for k := range out.F {
			if in[0].F[ic[k]] != 0 {
				out.F[k] = in[1].F[ia[k]]
			} else {
				out.F[k] = in[2].F[ib[k]]
			}
		}
		return []*Tensor{out}, nil
	case "Shape":
		t := &Tensor{Shape: []int{len(in[0].Shape)}, Type: Int64}
		for _, d := range in[0].Shape {
			t.F = append(t.F, float64(d))
		}
		start, end := int(n.Int("start", 0)), int(n.Int("end", int64(len(in[0].Shape))))
		start, end = clampAxis(start, len(in[0].Shape)), clampAxis(end, len(in[0].Shape))
		t.F, t.Shape = t.F[start:end], []int{end - start}
		return []*Tensor{t}, nil
	case "Size":
		return []*Tensor{scalar(float64(len(in[0].F)), Int64)}, nil
	case "ConstantOfShape":
		v := 0.0
		typ := Float
		if a, ok := n.Attrs["value"]; ok && a.T != nil {
			v, typ = a.T.F[0], a.T.Type
		}
		shape := in[0].Ints()
		t := &Tensor{Shape: shape, F: make([]float64, Len(shape)), Type: typ}
		for i := range t.F {
			t.F[i] = v
		}
		return []*Tensor{t}, nil
	case "Range":
		start, limit, delta := in[0].F[0], in[1].F[0], in[2].F[0]
		count := max(0, int(math.Ceil((limit-start)/delta)))
		t := &Tensor{Shape: []int{count}, Type: in[0].Type}
		for i := range count {
			t.F = append(t.F, start+float64(i)*delta)
		}
		return []*Tensor{t}, nil
	case "Unsqueeze":
		axes := n.IntList("axes")
		if t := arg(1); t != nil {
			axes = t.Ints()
		}
		r := len(in[0].Shape) + len(axes)
		shape := []int{}
		set := map[int]bool{}
		for _, a := range axes {
			set[axis(a, r)] = true
		}
		j := 0
		for i := range r {
			if set[i] {
				shape = append(shape, 1)
			} else {
				shape = append(shape, in[0].Shape[j])
				j++
			}
		}
		return []*Tensor{{Shape: shape, F: in[0].F, Type: in[0].Type}}, nil
	case "Squeeze":
		axes := n.IntList("axes")
		if t := arg(1); t != nil {
			axes = t.Ints()
		}
		set := map[int]bool{}
		for _, a := range axes {
			set[axis(a, len(in[0].Shape))] = true
		}
		shape := []int{}
		for i, d := range in[0].Shape {
			if (len(axes) == 0 && d == 1) || set[i] {
				continue
			}
			shape = append(shape, d)
		}
		return []*Tensor{{Shape: shape, F: in[0].F, Type: in[0].Type}}, nil
	case "Reshape":
		shape, err := ReshapeTo(in[0].Shape, in[1].Ints(), n.Int("allowzero", 0) != 0)
		if err != nil {
			return nil, err
		}
		return []*Tensor{{Shape: shape, F: in[0].F, Type: in[0].Type}}, nil
	case "Flatten":
		a := axis(int(n.Int("axis", 1)), len(in[0].Shape))
		return []*Tensor{{Shape: []int{Len(in[0].Shape[:a]), Len(in[0].Shape[a:])}, F: in[0].F, Type: in[0].Type}}, nil
	case "Concat":
		return one(concat(in, int(n.Int("axis", 0))))
	case "Gather":
		return one(gather(in[0], in[1], int(n.Int("axis", 0))))
	case "Slice":
		starts, ends := in[1].Ints(), in[2].Ints()
		var axes, steps []int
		if t := arg(3); t != nil {
			axes = t.Ints()
		}
		if t := arg(4); t != nil {
			steps = t.Ints()
		}
		begin, end, step := SliceParams(in[0].Shape, starts, ends, axes, steps)
		return []*Tensor{slice(in[0], begin, end, step)}, nil
	case "Transpose":
		perm := n.IntList("perm")
		if perm == nil {
			for i := len(in[0].Shape) - 1; i >= 0; i-- {
				perm = append(perm, i)
			}
		}
		return []*Tensor{transpose(in[0], perm)}, nil
	case "Expand":
		shape, err := Broadcast(in[0].Shape, in[1].Ints())
		if err != nil {
			return nil, err
		}
		idx := index(in[0].Shape, shape)
		out := &Tensor{Shape: shape, F: make([]float64, len(idx)), Type: in[0].Type}
		for k, i := range idx {
			out.F[k] = in[0].F[i]
		}
		return []*Tensor{out}, nil
	case "Tile":
		return []*Tensor{tile(in[0], in[1].Ints())}, nil
	case "Split":
		a := axis(int(n.Int("axis", 0)), len(in[0].Shape))
		sizes := n.IntList("split")
		if t := arg(1); t != nil {
			sizes = t.Ints()
		}
		if sizes == nil {
			k := len(n.Out)
			for range k {
				sizes = append(sizes, in[0].Shape[a]/k)
			}
		}
		var outs []*Tensor
		at := 0
		for _, s := range sizes {
			begin, end, step := make([]int, len(in[0].Shape)), slices.Clone(in[0].Shape), make([]int, len(in[0].Shape))
			for i := range step {
				step[i] = 1
			}
			begin[a], end[a] = at, at+s
			outs = append(outs, slice(in[0], begin, end, step))
			at += s
		}
		return outs, nil
	case "MatMul":
		return one(matmul(in[0], in[1]))
	case "Gemm":
		a, b := in[0], in[1]
		if n.Int("transA", 0) != 0 {
			a = transpose(a, []int{1, 0})
		}
		if n.Int("transB", 0) != 0 {
			b = transpose(b, []int{1, 0})
		}
		y, err := matmul(a, b)
		if err != nil {
			return nil, err
		}
		alpha, beta := n.Float("alpha", 1), n.Float("beta", 1)
		y = unary(y, func(x float64) float64 { return alpha * x })
		if c := arg(2); c != nil {
			return one(binop(y, c, y.Type, func(x, z float64) float64 { return x + beta*z }))
		}
		return []*Tensor{y}, nil
	case "Softmax":
		return []*Tensor{softmax(in[0], axis(int(n.Int("axis", -1)), len(in[0].Shape)))}, nil
	case "ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd":
		axes := n.IntList("axes")
		if t := arg(1); t != nil {
			axes = t.Ints()
		}
		return []*Tensor{reduce(in[0], n.Op, axes, n.Int("keepdims", 1) != 0, n.Int("noop_with_empty_axes", 0) != 0)}, nil
	case "ArgMax":
		return []*Tensor{argmax(in[0], axis(int(n.Int("axis", 0)), len(in[0].Shape)), n.Int("keepdims", 1) != 0)}, nil
	case "TopK":
		return topk(in[0], int(in[1].F[0]), axis(int(n.Int("axis", -1)), len(in[0].Shape)), n.Int("largest", 1) != 0)
	case "NonMaxSuppression":
		return one(nms(in, int(n.Int("center_point_box", 0))))
	case "Conv":
		return one(conv(n, in[0], in[1], arg(2)))
	case "Rsqrt":
		return []*Tensor{unary(in[0], func(x float64) float64 { return 1 / math.Sqrt(x) })}, nil
	case "Gelu":
		if n.Int("approximate", 0) != 0 {
			return []*Tensor{unary(in[0], func(x float64) float64 {
				return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
			})}, nil
		}
		return []*Tensor{unary(in[0], func(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) })}, nil
	case "StridedSlice":
		return one(stridedSlice(n, in[0], in[1].Ints(), in[2].Ints(), in[3].Ints()))
	case "TFLArgMax", "TFLArgMin":
		a := axis(in[1].Ints()[0], len(in[0].Shape))
		t := in[0]
		if n.Op == "TFLArgMin" {
			t = unary(t, func(x float64) float64 { return -x })
		}
		return []*Tensor{argmax(t, a, false)}, nil
	case "OneHot":
		depth := int(in[1].F[0])
		var on, off float64
		if len(in) == 3 { // ONNX: values [off, on]
			off, on = in[2].F[0], in[2].F[1]
		} else { // TFLite: on, off
			on, off = in[2].F[0], in[3].F[0]
		}
		idx := in[0]
		a := axis(int(n.Int("axis", -1)), len(idx.Shape)+1)
		shape := slices.Insert(slices.Clone(idx.Shape), a, depth)
		out := &Tensor{Shape: shape, F: make([]float64, Len(shape)), Type: in[2].Type}
		outer, inner := Len(idx.Shape[:a]), Len(idx.Shape[a:])
		for o := range outer {
			for d := range depth {
				for i := range inner {
					v := off
					if int(idx.F[o*inner+i]) == d {
						v = on
					}
					out.F[(o*depth+d)*inner+i] = v
				}
			}
		}
		return []*Tensor{out}, nil
	case "Pack":
		a := axis(int(n.Int("axis", 0)), len(in[0].Shape)+1)
		var parts []*Tensor
		for _, t := range in {
			parts = append(parts, &Tensor{Shape: slices.Insert(slices.Clone(t.Shape), a, 1), F: t.F, Type: t.Type})
		}
		return one(concat(parts, a))
	case "While":
		return while(n, in)
	case "CumSum":
		t := in[0]
		a := axis(in[1].Ints()[0], len(t.Shape))
		excl, rev := n.Int("exclusive", 0) != 0, n.Int("reverse", 0) != 0
		d, inner := t.Shape[a], Len(t.Shape[a+1:])
		out := &Tensor{Shape: slices.Clone(t.Shape), F: make([]float64, len(t.F)), Type: t.Type}
		for o := range Len(t.Shape[:a]) {
			for i := range inner {
				var s float64
				for k := range d {
					j := k
					if rev {
						j = d - 1 - k
					}
					at := (o*d+j)*inner + i
					if excl {
						out.F[at] = s
						s += t.F[at]
					} else {
						s += t.F[at]
						out.F[at] = s
					}
				}
			}
		}
		return []*Tensor{out}, nil
	case "TFLSlice": // begin and size (-1: to the end) per axis
		begin, size := in[1].Ints(), in[2].Ints()
		r := len(in[0].Shape)
		b, e, st := make([]int, r), slices.Clone(in[0].Shape), make([]int, r)
		for i := range r {
			st[i] = 1
			if i < len(begin) {
				b[i] = begin[i]
				if size[i] >= 0 {
					e[i] = begin[i] + size[i]
				}
			}
		}
		return []*Tensor{slice(in[0], b, e, st)}, nil
	case "ScatterND":
		return one(scatterND(in[0], in[1], in[2], n.Str("reduction", "none")))
	case "ConvTranspose":
		return one(convTranspose(n, in[0], in[1], arg(2)))
	case "InstanceNormalization":
		return []*Tensor{instanceNorm(in[0], in[1], in[2], n.Float("epsilon", 1e-5))}, nil
	case "LayerNormalization":
		return one(layerNorm(in[0], in[1], arg(2), int(n.Int("axis", -1)), n.Float("epsilon", 1e-5)))
	case "Resize":
		return one(resize(n, in[0], arg(2), arg(3)))
	case "Pad":
		pads := n.IntList("pads")
		if t := arg(1); t != nil {
			pads = t.Ints()
		}
		c := n.Float("value", 0)
		if t := arg(2); t != nil && len(t.F) > 0 {
			c = t.F[0]
		}
		var axes []int
		if t := arg(3); t != nil {
			axes = t.Ints()
		}
		return one(pad(in[0], FullPads(len(in[0].Shape), pads, axes), n.Str("mode", "constant"), c))
	}
	return nil, fmt.Errorf("the host does not evaluate %s", n.Op)
}

func clampAxis(a, rank int) int {
	if a < 0 {
		a += rank
	}
	return min(max(a, 0), rank)
}

// ReshapeTo is ONNX's Reshape of a shape to a target with 0 (copy, unless allowzero) and -1.
func ReshapeTo(shape, target []int, allowzero bool) ([]int, error) {
	out := slices.Clone(target)
	infer := -1
	known := 1
	for i, d := range out {
		switch {
		case d == 0 && !allowzero:
			out[i] = shape[i]
			known *= out[i]
		case d == -1:
			infer = i
		default:
			known *= d
		}
	}
	if infer >= 0 {
		if known == 0 {
			return nil, fmt.Errorf("reshape %v to %v", shape, target)
		}
		out[infer] = Len(shape) / known
	}
	if Len(out) != Len(shape) {
		return nil, fmt.Errorf("reshape %v to %v", shape, target)
	}
	return out, nil
}

// SliceParams turns ONNX Slice's starts, ends, axes and steps into a begin, end and step per
// axis of the shape, clamped as ONNX clamps them.
func SliceParams(shape, starts, ends, axes, steps []int) (begin, end, step []int) {
	r := len(shape)
	begin, end, step = make([]int, r), slices.Clone(shape), make([]int, r)
	for i := range step {
		step[i] = 1
	}
	if axes == nil {
		for i := range starts {
			axes = append(axes, i)
		}
	}
	for k, a := range axes {
		a = axis(a, r)
		d := shape[a]
		s := 1
		if steps != nil {
			s = steps[k]
		}
		b, e := starts[k], ends[k]
		if b < 0 {
			b += d
		}
		if e < 0 {
			e += d
		}
		if s > 0 {
			b, e = min(max(b, 0), d), min(max(e, 0), d)
		} else {
			b, e = min(max(b, -1), d-1), min(max(e, -1), d-1)
		}
		begin[a], end[a], step[a] = b, e, s
	}
	return begin, end, step
}

func slice(t *Tensor, begin, end, step []int) *Tensor {
	r := len(t.Shape)
	shape := make([]int, r)
	for i := range r {
		if step[i] > 0 {
			shape[i] = max(0, (end[i]-begin[i]+step[i]-1)/step[i])
		} else {
			shape[i] = max(0, (begin[i]-end[i]-step[i]-1)/(-step[i]))
		}
	}
	st := strides(t.Shape)
	out := &Tensor{Shape: shape, F: make([]float64, Len(shape)), Type: t.Type}
	coord := make([]int, r)
	for k := range out.F {
		off := 0
		for i := range r {
			off += (begin[i] + coord[i]*step[i]) * st[i]
		}
		out.F[k] = t.F[off]
		for i := r - 1; i >= 0; i-- {
			coord[i]++
			if coord[i] < shape[i] {
				break
			}
			coord[i] = 0
		}
	}
	return out
}

func transpose(t *Tensor, perm []int) *Tensor {
	r := len(t.Shape)
	shape := make([]int, r)
	for i, p := range perm {
		shape[i] = t.Shape[p]
	}
	st := strides(t.Shape)
	out := &Tensor{Shape: shape, F: make([]float64, len(t.F)), Type: t.Type}
	coord := make([]int, r)
	for k := range out.F {
		off := 0
		for i := range r {
			off += coord[i] * st[perm[i]]
		}
		out.F[k] = t.F[off]
		for i := r - 1; i >= 0; i-- {
			coord[i]++
			if coord[i] < shape[i] {
				break
			}
			coord[i] = 0
		}
	}
	return out
}

func concat(ts []*Tensor, a int) (*Tensor, error) {
	var parts []*Tensor
	for _, t := range ts {
		if t != nil {
			parts = append(parts, t)
		}
	}
	r := len(parts[0].Shape)
	a = axis(a, r)
	shape := slices.Clone(parts[0].Shape)
	shape[a] = 0
	for _, t := range parts {
		if len(t.Shape) != r {
			return nil, fmt.Errorf("concat of ranks %d and %d", r, len(t.Shape))
		}
		shape[a] += t.Shape[a]
	}
	outer := Len(shape[:a])
	out := &Tensor{Shape: shape, Type: parts[0].Type}
	for o := range outer {
		for _, t := range parts {
			inner := Len(t.Shape[a:])
			out.F = append(out.F, t.F[o*inner:(o+1)*inner]...)
		}
	}
	return out, nil
}

func gather(t, idx *Tensor, a int) (*Tensor, error) {
	a = axis(a, len(t.Shape))
	d := t.Shape[a]
	outer, inner := Len(t.Shape[:a]), Len(t.Shape[a+1:])
	shape := append(append(slices.Clone(t.Shape[:a]), idx.Shape...), t.Shape[a+1:]...)
	out := &Tensor{Shape: shape, Type: t.Type}
	for o := range outer {
		for _, iv := range idx.F {
			i := int(iv)
			if i < 0 {
				i += d
			}
			if i < 0 || i >= d {
				return nil, fmt.Errorf("index %d out of %d", int(iv), d)
			}
			out.F = append(out.F, t.F[(o*d+i)*inner:(o*d+i+1)*inner]...)
		}
	}
	return out, nil
}

func tile(t *Tensor, reps []int) *Tensor {
	shape := make([]int, len(t.Shape))
	for i := range shape {
		shape[i] = t.Shape[i] * reps[i]
	}
	st := strides(t.Shape)
	out := &Tensor{Shape: shape, F: make([]float64, Len(shape)), Type: t.Type}
	coord := make([]int, len(shape))
	for k := range out.F {
		off := 0
		for i := range shape {
			off += coord[i] % t.Shape[i] * st[i]
		}
		out.F[k] = t.F[off]
		for i := len(shape) - 1; i >= 0; i-- {
			coord[i]++
			if coord[i] < shape[i] {
				break
			}
			coord[i] = 0
		}
	}
	return out
}

func matmul(a, b *Tensor) (*Tensor, error) {
	as, bs := slices.Clone(a.Shape), slices.Clone(b.Shape)
	if len(as) == 1 {
		as = []int{1, as[0]}
	}
	if len(bs) == 1 {
		bs = []int{bs[0], 1}
	}
	m, k, k2, nn := as[len(as)-2], as[len(as)-1], bs[len(bs)-2], bs[len(bs)-1]
	if k != k2 {
		return nil, fmt.Errorf("matmul of %v and %v", a.Shape, b.Shape)
	}
	batch, err := Broadcast(as[:len(as)-2], bs[:len(bs)-2])
	if err != nil {
		return nil, err
	}
	ia, ib := index(as[:len(as)-2], batch), index(bs[:len(bs)-2], batch)
	shape := append(slices.Clone(batch), m, nn)
	out := &Tensor{Shape: shape, F: make([]float64, Len(shape)), Type: a.Type}
	for bi := range Len(batch) {
		pa, pb, po := a.F[ia[bi]*m*k:], b.F[ib[bi]*k*nn:], out.F[bi*m*nn:]
		if m*k*nn >= bigMatMul {
			matmul32(m, k, nn, pa, pb, po)
			continue
		}
		for i := range m {
			for kk := range k {
				x := pa[i*k+kk]
				if x == 0 {
					continue
				}
				row := pb[kk*nn : (kk+1)*nn]
				o := po[i*nn : (i+1)*nn]
				for j, y := range row {
					o[j] += x * y
				}
			}
		}
	}
	// a vector operand loses its axis again
	switch {
	case len(a.Shape) == 1 && len(b.Shape) == 1:
		out.Shape = []int{}
	case len(a.Shape) == 1:
		out.Shape = append(slices.Clone(batch), nn)
	case len(b.Shape) == 1:
		out.Shape = append(slices.Clone(batch), m)
	}
	return out, nil
}

// bigMatMul is the products (m k n) from which matmul multiplies in float32 with internal/sgemm
// (as ONNX Runtime does) in place of float64 loops: EfficientSAM's mask decoder 1.4 -> 0.7 s.
const bigMatMul = 1 << 22

// matmul32 is c (m x n) = a (m x k) b (k x n), row major, through float32.
func matmul32(m, k, n int, a, b, c []float64) {
	a32, b32, c32 := make([]float32, m*k), make([]float32, k*n), make([]float32, m*n)
	for i := range a32 {
		a32[i] = float32(a[i])
	}
	for i := range b32 {
		b32[i] = float32(b[i])
	}
	sgemm.MulAdd(m, n, k, sgemm.A{Data: a32, Stride: k}, sgemm.Matrix{Data: b32, Stride: n, N: n}, c32, n, 1)
	for i, v := range c32 {
		c[i] = float64(v)
	}
}

func softmax(t *Tensor, a int) *Tensor {
	out := &Tensor{Shape: slices.Clone(t.Shape), F: make([]float64, len(t.F)), Type: t.Type}
	d, inner := t.Shape[a], Len(t.Shape[a+1:])
	for o := range Len(t.Shape[:a]) {
		for in := range inner {
			hi := math.Inf(-1)
			for i := range d {
				hi = math.Max(hi, t.F[(o*d+i)*inner+in])
			}
			var z float64
			for i := range d {
				e := math.Exp(t.F[(o*d+i)*inner+in] - hi)
				out.F[(o*d+i)*inner+in] = e
				z += e
			}
			for i := range d {
				out.F[(o*d+i)*inner+in] /= z
			}
		}
	}
	return out
}

func reduce(t *Tensor, op string, axes []int, keep, noop bool) *Tensor {
	r := len(t.Shape)
	set := make([]bool, r)
	if len(axes) == 0 {
		if noop {
			return t
		}
		for i := range set {
			set[i] = true
		}
	}
	for _, a := range axes {
		set[axis(a, r)] = true
	}
	var shape, kept []int
	for i, d := range t.Shape {
		if set[i] {
			kept = append(kept, 1)
			if keep {
				shape = append(shape, 1)
			}
		} else {
			kept = append(kept, d)
			shape = append(shape, d)
		}
	}
	init := 0.0
	switch op {
	case "ReduceMax":
		init = math.Inf(-1)
	case "ReduceMin":
		init = math.Inf(1)
	case "ReduceProd":
		init = 1
	}
	out := &Tensor{Shape: shape, F: make([]float64, Len(kept)), Type: t.Type}
	for i := range out.F {
		out.F[i] = init
	}
	ks := strides(kept)
	coord := make([]int, r)
	for _, x := range t.F {
		off := 0
		for i := range r {
			if !set[i] {
				off += coord[i] * ks[i]
			}
		}
		switch op {
		case "ReduceMax":
			out.F[off] = math.Max(out.F[off], x)
		case "ReduceMin":
			out.F[off] = math.Min(out.F[off], x)
		case "ReduceProd":
			out.F[off] *= x
		default:
			out.F[off] += x
		}
		for i := r - 1; i >= 0; i-- {
			coord[i]++
			if coord[i] < t.Shape[i] {
				break
			}
			coord[i] = 0
		}
	}
	if op == "ReduceMean" {
		n := float64(len(t.F)) / float64(len(out.F))
		for i := range out.F {
			out.F[i] /= n
		}
	}
	return out
}

func argmax(t *Tensor, a int, keep bool) *Tensor {
	d, inner := t.Shape[a], Len(t.Shape[a+1:])
	shape := slices.Clone(t.Shape)
	if keep {
		shape[a] = 1
	} else {
		shape = slices.Delete(shape, a, a+1)
	}
	out := &Tensor{Shape: shape, Type: Int64}
	for o := range Len(t.Shape[:a]) {
		for in := range inner {
			best := 0
			for i := 1; i < d; i++ {
				if t.F[(o*d+i)*inner+in] > t.F[(o*d+best)*inner+in] {
					best = i
				}
			}
			out.F = append(out.F, float64(best))
		}
	}
	return out
}

func topk(t *Tensor, k, a int, largest bool) ([]*Tensor, error) {
	d, inner := t.Shape[a], Len(t.Shape[a+1:])
	if k > d {
		return nil, fmt.Errorf("top %d of %d", k, d)
	}
	shape := slices.Clone(t.Shape)
	shape[a] = k
	vals := &Tensor{Shape: shape, F: make([]float64, Len(shape)), Type: t.Type}
	idxs := &Tensor{Shape: slices.Clone(shape), F: make([]float64, Len(shape)), Type: Int64}
	order := make([]int, d)
	for o := range Len(t.Shape[:a]) {
		for in := range inner {
			at := func(i int) float64 { return t.F[(o*d+i)*inner+in] }
			for i := range order {
				order[i] = i
			}
			// ties keep the lower index first, as onnxruntime does
			sort.SliceStable(order, func(x, y int) bool {
				if largest {
					return at(order[x]) > at(order[y])
				}
				return at(order[x]) < at(order[y])
			})
			for j := range k {
				vals.F[(o*k+j)*inner+in] = at(order[j])
				idxs.F[(o*k+j)*inner+in] = float64(order[j])
			}
		}
	}
	return []*Tensor{vals, idxs}, nil
}

// nms is NonMaxSuppression: boxes (batch, n, 4), scores (batch, classes, n), and optionally the
// most boxes to keep per class, the IoU threshold and the score threshold; it returns the kept
// (batch, class, box) triples, by class and then by score.
func nms(in []*Tensor, center int) (*Tensor, error) {
	boxes, scores := in[0], in[1]
	keep := 0
	iou, minScore := 0.0, math.Inf(-1)
	if len(in) > 2 && in[2] != nil && len(in[2].F) > 0 {
		keep = int(in[2].F[0])
	}
	if len(in) > 3 && in[3] != nil && len(in[3].F) > 0 {
		iou = in[3].F[0]
	}
	if len(in) > 4 && in[4] != nil && len(in[4].F) > 0 {
		minScore = in[4].F[0]
	}
	nb, nc, n := scores.Shape[0], scores.Shape[1], scores.Shape[2]
	out := &Tensor{Type: Int64}
	corners := func(b, i int) (x1, y1, x2, y2 float64) {
		p := boxes.F[(b*n+i)*4:]
		if center != 0 {
			return p[0] - p[2]/2, p[1] - p[3]/2, p[0] + p[2]/2, p[1] + p[3]/2
		}
		return math.Min(p[1], p[3]), math.Min(p[0], p[2]), math.Max(p[1], p[3]), math.Max(p[0], p[2])
	}
	count := 0
	for b := range nb {
		for c := range nc {
			var cand []int
			for i := range n {
				if scores.F[(b*nc+c)*n+i] > minScore {
					cand = append(cand, i)
				}
			}
			sort.SliceStable(cand, func(x, y int) bool {
				return scores.F[(b*nc+c)*n+cand[x]] > scores.F[(b*nc+c)*n+cand[y]]
			})
			var kept []int
			for _, i := range cand {
				if keep > 0 && len(kept) >= keep {
					break
				}
				ax1, ay1, ax2, ay2 := corners(b, i)
				ok := true
				for _, j := range kept {
					bx1, by1, bx2, by2 := corners(b, j)
					w := math.Max(0, math.Min(ax2, bx2)-math.Max(ax1, bx1))
					h := math.Max(0, math.Min(ay2, by2)-math.Max(ay1, by1))
					inter := w * h
					union := (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
					if union > 0 && inter/union > iou {
						ok = false
						break
					}
				}
				if ok {
					kept = append(kept, i)
					out.F = append(out.F, float64(b), float64(c), float64(i))
					count++
				}
			}
		}
	}
	out.Shape = []int{count, 3}
	return out, nil
}

// Cut splits a graph at its dynamic part: the nodes of the ops in dynamic (TopK,
// NonMaxSuppression) and every node that reads a value they compute, which a program of fixed
// shapes cannot hold. It returns the values computed from input by the rest that those nodes
// read, in the graph's order: the outputs of a program for the static part (Program), from which
// the host evaluates the dynamic part (EvalShapes). Shape reads no data, so a value whose shape
// alone the dynamic part reads is not among them.
func (g *Graph) Cut(input string, dynamic map[string]bool, outputs []string) []string {
	dep := map[string]bool{input: true}
	tail := map[string]bool{}
	for _, n := range g.Nodes {
		data := false
		for _, in := range n.In {
			if dep[in] {
				data = true
			}
		}
		if n.Op == "Shape" {
			continue
		}
		dyn := dynamic[n.Op]
		for _, in := range n.In {
			if tail[in] {
				dyn = true
			}
		}
		for _, o := range n.Out {
			if data {
				dep[o] = true
			}
			if dyn && data {
				tail[o] = true
			}
		}
	}
	var cut []string
	seen := map[string]bool{}
	for _, n := range g.needed(outputs, nil, nil) {
		isTail := false
		for _, o := range n.Out {
			isTail = isTail || tail[o]
		}
		if !isTail {
			continue
		}
		for _, in := range n.In {
			if in != "" && dep[in] && !tail[in] && !seen[in] && in != input {
				seen[in] = true
				cut = append(cut, in)
			}
		}
	}
	return cut
}

// FullPads turns ONNX Pad's pads (begins then ends, of the axes given or all) into a begin and an
// end for every axis: (b0, e0, b1, e1, ...).
func FullPads(rank int, pads, axes []int) []int {
	if axes == nil {
		for i := range len(pads) / 2 {
			axes = append(axes, i)
		}
	}
	out := make([]int, 2*rank)
	for k, a := range axes {
		a = axis(a, rank)
		out[2*a], out[2*a+1] = pads[k], pads[len(axes)+k]
	}
	return out
}

// pad pads t by (begin, end) per axis with a constant, or by reflection (the edge not repeated)
// or the edge's value; a negative pad crops.
func pad(t *Tensor, pads []int, mode string, c float64) (*Tensor, error) {
	r := len(t.Shape)
	shape := make([]int, r)
	for i := range r {
		shape[i] = t.Shape[i] + pads[2*i] + pads[2*i+1]
	}
	st := strides(t.Shape)
	out := &Tensor{Shape: shape, F: make([]float64, Len(shape)), Type: t.Type}
	coord := make([]int, r)
	for k := range out.F {
		off, inside := 0, true
		for i := range r {
			j := coord[i] - pads[2*i]
			d := t.Shape[i]
			if j < 0 || j >= d {
				switch mode {
				case "reflect":
					if d == 1 {
						j = 0
						break
					}
					for j < 0 || j >= d {
						if j < 0 {
							j = -j
						}
						if j >= d {
							j = 2*(d-1) - j
						}
					}
				case "edge":
					j = min(max(j, 0), d-1)
				default:
					inside = false
				}
			}
			off += j * st[i]
		}
		if inside {
			out.F[k] = t.F[off]
		} else {
			out.F[k] = c
		}
		for i := r - 1; i >= 0; i-- {
			coord[i]++
			if coord[i] < shape[i] {
				break
			}
			coord[i] = 0
		}
	}
	return out, nil
}

// conv is Conv over 1 or 2 spatial axes (a 1 D one as 2 D of height 1), with groups, strides,
// dilations and explicit pads.
func conv(n *Node, x, w, b *Tensor) (*Tensor, error) {
	k := len(w.Shape) - 2
	if k != 1 && k != 2 || len(x.Shape) != k+2 {
		return nil, fmt.Errorf("a convolution of %v by %v", x.Shape, w.Shape)
	}
	if ap := n.Str("auto_pad", "NOTSET"); ap != "NOTSET" && ap != "VALID" {
		return nil, fmt.Errorf("auto_pad %s", ap)
	}
	strides, dil, pads := n.IntList("strides"), n.IntList("dilations"), n.IntList("pads")
	if strides == nil {
		strides = slices.Repeat([]int{1}, k)
	}
	if dil == nil {
		dil = slices.Repeat([]int{1}, k)
	}
	if pads == nil {
		pads = make([]int, 2*k)
	}
	// as 2 D: (h, w) with h = 1 for one axis
	xh, xw, kh, kw := 1, x.Shape[2], 1, w.Shape[2]
	sh, sw, dh, dw, pt, pl, pb, pr := 1, strides[0], 1, dil[0], 0, pads[0], 0, pads[1]
	if k == 2 {
		xh, xw, kh, kw = x.Shape[2], x.Shape[3], w.Shape[2], w.Shape[3]
		sh, sw, dh, dw = strides[0], strides[1], dil[0], dil[1]
		pt, pl, pb, pr = pads[0], pads[1], pads[2], pads[3]
	}
	nb, cin, cout := x.Shape[0], x.Shape[1], w.Shape[0]
	g := int(n.Int("group", 1))
	cg := cin / g
	oh, ow := (xh+pt+pb-dh*(kh-1)-1)/sh+1, (xw+pl+pr-dw*(kw-1)-1)/sw+1
	shape := []int{nb, cout, oh, ow}
	if k == 1 {
		shape = []int{nb, cout, ow}
	}
	out := &Tensor{Shape: shape, F: make([]float64, nb*cout*oh*ow), Type: x.Type}
	per := cout / g
	for bi := range nb {
		for o := range cout {
			gi := o / per
			dst := out.F[(bi*cout+o)*oh*ow:][:oh*ow]
			if b != nil {
				for i := range dst {
					dst[i] = b.F[o]
				}
			}
			for c := range cg {
				src := x.F[(bi*cin+gi*cg+c)*xh*xw:][:xh*xw]
				for a := range kh {
					for e := range kw {
						wv := w.F[((o*cg+c)*kh+a)*kw+e]
						if wv == 0 {
							continue
						}
						for i := range oh {
							y := i*sh - pt + a*dh
							if y < 0 || y >= xh {
								continue
							}
							row := src[y*xw:][:xw]
							drow := dst[i*ow:][:ow]
							for j := range ow {
								xx := j*sw - pl + e*dw
								if xx >= 0 && xx < xw {
									drow[j] += wv * row[xx]
								}
							}
						}
					}
				}
			}
		}
	}
	return out, nil
}

// while runs a TFLite WHILE: the body over the loop's values for as long as the condition holds.
func while(n *Node, in []*Tensor) ([]*Tensor, error) {
	if len(n.Graphs) != 2 {
		return nil, fmt.Errorf("a loop without its condition and body")
	}
	cond, body := n.Graphs[0], n.Graphs[1]
	vars := in
	for step := 0; ; step++ {
		if step > 1<<20 {
			return nil, fmt.Errorf("the loop does not end")
		}
		given := map[string]*Tensor{}
		for i, vi := range cond.Inputs {
			given[vi.Name] = vars[i]
		}
		c, err := cond.Eval(given, []string{cond.Outputs[0].Name})
		if err != nil {
			return nil, fmt.Errorf("condition: %w", err)
		}
		if c[cond.Outputs[0].Name].F[0] == 0 {
			return vars, nil
		}
		given = map[string]*Tensor{}
		for i, vi := range body.Inputs {
			given[vi.Name] = vars[i]
		}
		var names []string
		for _, vo := range body.Outputs {
			names = append(names, vo.Name)
		}
		out, err := body.Eval(given, names)
		if err != nil {
			return nil, fmt.Errorf("body: %w", err)
		}
		next := make([]*Tensor, len(names))
		for i, nm := range names {
			next[i] = out[nm]
		}
		vars = next
	}
}

// stridedSlice is TFLite's STRIDED_SLICE (no ellipsis or new axes).
func stridedSlice(n *Node, t *Tensor, begin, end, strides []int) (*Tensor, error) {
	if n.Int("ellipsis_mask", 0) != 0 || n.Int("new_axis_mask", 0) != 0 {
		return nil, fmt.Errorf("strided slice with an ellipsis or new axes")
	}
	bm, em, sm := n.Int("begin_mask", 0), n.Int("end_mask", 0), n.Int("shrink_axis_mask", 0)
	r := len(t.Shape)
	b, e, st := make([]int, r), slices.Clone(t.Shape), make([]int, r)
	var keep []int
	for i := range r {
		st[i] = 1
		d := t.Shape[i]
		if i >= len(begin) {
			keep = append(keep, d)
			continue
		}
		s := strides[i]
		bi, ei := begin[i], end[i]
		if bi < 0 {
			bi += d
		}
		if ei < 0 {
			ei += d
		}
		if bm>>i&1 != 0 {
			if s > 0 {
				bi = 0
			} else {
				bi = d - 1
			}
		}
		if em>>i&1 != 0 {
			if s > 0 {
				ei = d
			} else {
				ei = -1
			}
		}
		if sm>>i&1 != 0 {
			ei, s = bi+1, 1
		}
		if s > 0 {
			bi, ei = min(max(bi, 0), d), min(max(ei, 0), d)
		}
		b[i], e[i], st[i] = bi, ei, s
		if sm>>i&1 == 0 {
			if s > 0 {
				keep = append(keep, max(0, (ei-bi+s-1)/s))
			} else {
				keep = append(keep, max(0, (bi-ei-s-1)/(-s)))
			}
		}
	}
	out := slice(t, b, e, st)
	out.Shape = keep
	return out, nil
}

// rfft2d is TFLite's RFFT2D over the last two axes with fft_length (m, n), for m 1 (each row
// alone; n a power of 2): the n/2+1 complex values of each row, zero padded or cropped to n.
func rfft2d(x *Tensor, length []int) (*Tensor, error) {
	r := len(x.Shape)
	if r < 2 || len(length) != 2 || length[0] != 1 || x.Shape[r-2] != 1 {
		return nil, fmt.Errorf("rfft2d of %v by %v: only one row at a time", x.Shape, length)
	}
	n := length[1]
	if n&(n-1) != 0 {
		return nil, fmt.Errorf("rfft2d of length %d: not a power of 2", n)
	}
	w := x.Shape[r-1]
	rows := Len(x.Shape) / w
	half := n/2 + 1
	shape := append(append([]int{}, x.Shape[:r-1]...), half)
	out := &Tensor{Shape: shape, Type: Complex64, F: make([]float64, 2*rows*half)}
	re, im := make([]float64, n), make([]float64, n)
	cos, sin := make([]float64, n/2), make([]float64, n/2)
	for k := range cos {
		cos[k], sin[k] = math.Cos(2*math.Pi*float64(k)/float64(n)), -math.Sin(2*math.Pi*float64(k)/float64(n))
	}
	bits := 0
	for 1<<bits < n {
		bits++
	}
	for row := range rows {
		clear(im)
		for i := range n {
			j := 0
			for b := range bits {
				j |= (i >> b & 1) << (bits - 1 - b)
			}
			re[j] = 0
			if i < w {
				re[j] = x.F[row*w+i]
			}
		}
		for size := 2; size <= n; size <<= 1 {
			step := n / size
			for s := 0; s < n; s += size {
				for k := range size / 2 {
					c, sn := cos[k*step], sin[k*step]
					a, b := s+k, s+k+size/2
					tr, ti := re[b]*c-im[b]*sn, re[b]*sn+im[b]*c
					re[b], im[b] = re[a]-tr, im[a]-ti
					re[a], im[a] = re[a]+tr, im[a]+ti
				}
			}
		}
		for k := range half {
			out.F[2*(row*half+k)], out.F[2*(row*half+k)+1] = re[k], im[k]
		}
	}
	return out, nil
}
