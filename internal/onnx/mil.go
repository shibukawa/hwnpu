package onnx

import (
	"math"
	"regexp"
	"slices"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// Program is the static part of a graph as a MIL program, for llm.WritePart: from the inputs,
// each of a fixed shape (the first called x in the program, the others by Names), to Outputs.
// One output is y; several are concatenated on Axis into y, in order (Widths says how much of
// the axis each takes). Every value the graph computes from shapes and constants is computed on
// the host while building (Shape of a program value is its static shape), so that the program
// holds only the computation on the inputs, in fp16.
type Program struct {
	G       *Graph
	Inputs  []ValueInfo
	Outputs []string
	Axis    int
	// Flat concatenates the outputs of different shapes (a cut of a graph, Graph.Cut) instead:
	// each, in its order, as (1, rows, size/rows) with rows the largest common divisor of their
	// sizes up to the Neural Engine's dimension (16384), on axis 2. OutShapes are their shapes;
	// Split takes y apart.
	Flat bool
	rows int
	// Names maps a graph input to the program's input name (the first is always x).
	Names map[string]string
	// widths and folded are filled in by Build.
	widths    []int
	folded    map[string]*Tensor
	shapes    map[string][]int
	outShapes [][]int
}

// Shapes are the static shapes of the graph's values in the program, after Build.
func (p *Program) Shapes() map[string][]int { return p.shapes }

// Folded are the values of the graph that Build computed on the host, from shapes and
// constants: what a host evaluation of the rest of the graph (Graph.Eval) is given besides the
// program's outputs.
func (p *Program) Folded() map[string]*Tensor { return p.folded }

// The llm.Model methods a program does not use.
func (p *Program) Hidden() int               { return 0 }
func (p *Program) Layers() int               { return 0 }
func (p *Program) Output() (int, bool)       { return 0, false }
func (p *Program) Embedding() string         { return "" }
func (p *Program) Tensors() map[string][]int { return nil }

// Widths is how much of y's axis each output takes, after Build.
func (p *Program) Widths() []int { return p.widths }

// OutShapes are the outputs' shapes, after Build.
func (p *Program) OutShapes() [][]int { return p.outShapes }

// Split takes y apart into the outputs as host values (Flat).
func (p *Program) Split(y []float32) map[string]*Tensor {
	out := map[string]*Tensor{}
	width := 0
	for _, w := range p.widths {
		width += w
	}
	at := 0
	for i, name := range p.Outputs {
		n, w := Len(p.outShapes[i]), p.widths[i]
		t := &Tensor{Shape: p.outShapes[i], F: make([]float64, n), Type: Float}
		for j := range n {
			t.F[j] = float64(y[j/w*width+at+j%w])
		}
		out[name] = t
		at += w
	}
	return out
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

type val struct {
	v *mlpackage.Var // a program value
	c *Tensor        // or a constant
}

func (x val) shape() []int {
	if x.v != nil {
		return x.v.Shape
	}
	return x.c.Shape
}

type lowering struct {
	*blocks.Graph
	vals map[string]val
}

var nameRe = regexp.MustCompile(`[^A-Za-z0-9_]`)

// Build writes the program.
func (p *Program) Build(b *mlpackage.Builder, w llm.Weights, opts llm.Options) (ins []mlpackage.Feature, out mlpackage.Feature, err error) {
	if opts.Format == "" {
		opts.Format = "fp16"
	}
	if opts.Group == 0 {
		opts.Group = 16
	}
	l := &lowering{Graph: &blocks.Graph{Name: "onnx", B: b, W: w, Opts: opts}, vals: map[string]val{}}
	var y *mlpackage.Var
	err = blocks.Run(func() {
		stop := map[string]bool{}
		for i, in := range p.Inputs {
			name := "x"
			if i > 0 {
				name = p.Names[in.Name]
				if name == "" {
					name = nameRe.ReplaceAllString(in.Name, "_")
				}
			}
			v := b.Input(name, mlpackage.Float16, in.Shape...)
			ins = append(ins, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: in.Shape})
			l.vals[in.Name] = val{v: v}
			stop[in.Name] = true
		}
		for k, t := range p.G.Init {
			if !stop[k] {
				l.vals[k] = val{c: t}
			}
		}
		for _, n := range p.G.Needed(p.Outputs, stop) {
			l.node(n)
		}
		p.folded, p.shapes = map[string]*Tensor{}, map[string][]int{}
		for k, x := range l.vals {
			if x.v != nil {
				p.shapes[k] = x.v.Shape
			}
			if x.c != nil {
				if _, init := p.G.Init[k]; !init {
					p.folded[k] = x.c
				}
			}
		}
		var outs []*mlpackage.Var
		p.widths, p.outShapes = nil, nil
		for _, o := range p.Outputs {
			x, ok := l.vals[o]
			if !ok {
				l.Fail("no value %s", o)
			}
			p.outShapes = append(p.outShapes, x.shape())
		}
		if p.Flat {
			p.rows = 0
			for _, s := range p.outShapes {
				p.rows = gcd(p.rows, Len(s))
			}
			for p.rows > llm.NeuralEngineMaxDim {
				d := 2
				for p.rows%d != 0 {
					d++
				}
				p.rows /= d
			}
		}
		for i, o := range p.Outputs {
			v := l.v(l.vals[o])
			if p.Flat {
				v = l.reshape(v, []int{1, p.rows, Len(p.outShapes[i]) / p.rows})
				p.widths = append(p.widths, v.Shape[2])
			} else {
				p.widths = append(p.widths, v.Shape[axis(p.Axis, len(v.Shape))])
			}
			outs = append(outs, v)
		}
		if len(outs) == 1 {
			y = b.NamedOp("identity", "y", map[string]any{"x": outs[0]}, mlpackage.Float16, outs[0].Shape...)
			return
		}
		a := axis(p.Axis, len(outs[0].Shape))
		if p.Flat {
			a = 2
		}
		shape := slices.Clone(outs[0].Shape)
		shape[a] = 0
		for _, v := range outs {
			shape[a] += v.Shape[a]
		}
		y = b.NamedOp("concat", "y", map[string]any{"values": outs, "axis": b.ScalarI32(int32(a)), "interleave": b.ScalarBool(false)}, mlpackage.Float16, shape...)
	})
	if err != nil {
		return nil, out, err
	}
	return ins, mlpackage.Feature{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}, nil
}

// v is x as a program value: a constant becomes an fp16 constant.
func (l *lowering) v(x val) *mlpackage.Var {
	if x.v != nil {
		return x.v
	}
	// fp16's largest finite value at most: a mask's -3.4e38 would be -inf, and 0 times it NaN
	vals := make([]float32, len(x.c.F))
	for i, f := range x.c.F {
		vals[i] = float32(math.Max(-65504, math.Min(65504, f)))
	}
	if len(x.c.Shape) == 0 {
		return l.B.ScalarF16(vals[0])
	}
	return l.F16s("k", x.c.Shape, vals)
}

func ints32(v []int) *[]int32 {
	out := make([]int32, len(v))
	for i, x := range v {
		out[i] = int32(x)
	}
	return &out
}

func (l *lowering) ints(v ...int) *mlpackage.Var { return l.B.Ints(*ints32(v)...) }

func (l *lowering) op(typ string, in map[string]any, shape ...int) *mlpackage.Var {
	return l.Op(typ, in, shape...)
}

// node lowers one node, or computes it on the host when its inputs are all constants.
func (l *lowering) node(n *Node) {
	in := make([]val, len(n.In))
	allConst := true
	for i, name := range n.In {
		if name == "" {
			continue
		}
		x, ok := l.vals[name]
		if !ok {
			l.Fail("%s %s: no value for %s", n.Op, n.Name, name)
		}
		in[i] = x
		if x.v != nil {
			allConst = false
		}
	}
	set := func(i int, x val) { l.vals[n.Out[i]] = x }
	if n.Op == "Shape" && in[0].v != nil { // a program value's shape is static
		t := &Tensor{Shape: []int{len(in[0].v.Shape)}, Type: Int64}
		for _, d := range in[0].v.Shape {
			t.F = append(t.F, float64(d))
		}
		r := len(in[0].v.Shape)
		start, end := clampAxis(int(n.Int("start", 0)), r), clampAxis(int(n.Int("end", int64(r))), r)
		t.F, t.Shape = t.F[start:end], []int{end - start}
		set(0, val{c: t})
		return
	}
	if allConst {
		ts := make([]*Tensor, len(in))
		for i, x := range in {
			ts[i] = x.c
		}
		outs, err := EvalNode(n, ts)
		if err != nil {
			l.Fail("%s %s: %v", n.Op, n.Name, err)
		}
		for i := range outs {
			if i < len(n.Out) {
				set(i, val{c: outs[i]})
			}
		}
		return
	}
	c := func(i int) *Tensor {
		if i >= len(in) || in[i].c == nil && in[i].v == nil {
			return nil
		}
		if in[i].c == nil {
			l.Fail("%s %s: input %d (%s) must be a constant", n.Op, n.Name, i, n.In[i])
		}
		return in[i].c
	}
	x := func(i int) *mlpackage.Var { return l.v(in[i]) }
	out := func(v *mlpackage.Var) { set(0, val{v: v}) }
	switch n.Op {
	case "Identity", "Dropout":
		set(0, in[0])
	case "Cast":
		to := int(n.Int("to", Float))
		if to != Float && to != Float16 && to != Double {
			l.Fail("Cast %s of a program value to type %d", n.Name, to)
		}
		set(0, in[0])
	case "Add", "Sub", "Mul", "Div", "Max", "Min":
		if n.Op == "Mul" && n.In[0] == n.In[1] {
			out(l.op("mul", map[string]any{"x": x(0), "y": x(0)}, x(0).Shape...))
			return
		}
		shape, err := Broadcast(in[0].shape(), in[1].shape())
		if err != nil {
			l.Fail("%s %s: %v", n.Op, n.Name, err)
		}
		typ := map[string]string{"Add": "add", "Sub": "sub", "Mul": "mul", "Div": "real_div", "Max": "maximum", "Min": "minimum"}[n.Op]
		out(l.op(typ, map[string]any{"x": x(0), "y": x(1)}, shape...))
	case "Pow":
		e := c(1)
		if e == nil || len(e.F) != 1 {
			l.Fail("Pow %s: the exponent must be a scalar constant", n.Name)
		}
		switch e.F[0] {
		case 1:
			set(0, in[0])
		case 2:
			out(l.op("mul", map[string]any{"x": x(0), "y": x(0)}, x(0).Shape...))
		case 0.5:
			out(l.op("sqrt", map[string]any{"x": x(0)}, x(0).Shape...))
		default:
			out(l.op("pow", map[string]any{"x": x(0), "y": l.B.ScalarF16(float32(e.F[0]))}, x(0).Shape...))
		}
	case "Sigmoid", "Relu", "Exp", "Sqrt", "Abs", "Erf", "Tanh", "Cos", "Sin", "Log":
		typ := map[string]string{"Sigmoid": "sigmoid", "Relu": "relu", "Exp": "exp", "Sqrt": "sqrt", "Abs": "abs", "Erf": "erf", "Tanh": "tanh", "Cos": "cos", "Sin": "sin", "Log": "log"}[n.Op]
		out(l.op(typ, map[string]any{"x": x(0)}, x(0).Shape...))
	case "Rsqrt":
		out(l.op("rsqrt", map[string]any{"x": x(0), "epsilon": l.B.ScalarF16(0)}, x(0).Shape...))
	case "Neg":
		out(l.op("mul", map[string]any{"x": x(0), "y": l.B.ScalarF16(-1)}, x(0).Shape...))
	case "Clip":
		lo, hi := math.Inf(-1), math.Inf(1)
		if a, ok := n.Attrs["min"]; ok {
			lo = a.F
		}
		if a, ok := n.Attrs["max"]; ok {
			hi = a.F
		}
		if t := c(1); t != nil {
			lo = t.F[0]
		}
		if t := c(2); t != nil {
			hi = t.F[0]
		}
		v := x(0)
		if !math.IsInf(lo, 0) {
			v = l.op("maximum", map[string]any{"x": v, "y": l.B.ScalarF16(float32(math.Max(lo, -65504)))}, v.Shape...)
		}
		if !math.IsInf(hi, 0) {
			v = l.op("minimum", map[string]any{"x": v, "y": l.B.ScalarF16(float32(math.Min(hi, 65504)))}, v.Shape...)
		}
		out(v)
	case "Softmax":
		a := axis(int(n.Int("axis", -1)), len(in[0].shape()))
		out(l.op("softmax", map[string]any{"x": x(0), "axis": l.B.ScalarI32(int32(a))}, x(0).Shape...))
	case "Conv":
		l.conv(n, in)
	case "MaxPool":
		l.maxPool(n, in)
	case "MatMul":
		l.matmul(n, in)
	case "Gemm":
		l.gemm(n, in)
	case "Reshape":
		shape, err := ReshapeTo(in[0].shape(), c(1).Ints(), n.Int("allowzero", 0) != 0)
		if err != nil {
			l.Fail("Reshape %s: %v", n.Name, err)
		}
		out(l.reshape(x(0), shape))
	case "Flatten":
		s := in[0].shape()
		a := axis(int(n.Int("axis", 1)), len(s))
		out(l.reshape(x(0), []int{Len(s[:a]), Len(s[a:])}))
	case "Unsqueeze", "Squeeze":
		axes := n.IntList("axes")
		if t := c(1); t != nil {
			axes = t.Ints()
		}
		ts, err := EvalNode(&Node{Op: n.Op, Attrs: map[string]Attr{"axes": {Ints: toInt64(axes)}}}, []*Tensor{{Shape: in[0].shape(), F: make([]float64, Len(in[0].shape()))}})
		if err != nil {
			l.Fail("%s %s: %v", n.Op, n.Name, err)
		}
		out(l.reshape(x(0), ts[0].Shape))
	case "Transpose":
		perm := n.IntList("perm")
		s := in[0].shape()
		if perm == nil {
			for i := len(s) - 1; i >= 0; i-- {
				perm = append(perm, i)
			}
		}
		shape := make([]int, len(s))
		for i, p := range perm {
			shape[i] = s[p]
		}
		out(l.op("transpose", map[string]any{"x": x(0), "perm": l.ints(perm...)}, shape...))
	case "Concat":
		var vs []*mlpackage.Var
		var shapes [][]int
		for i := range in {
			if in[i].v == nil && in[i].c == nil || Len(in[i].shape()) == 0 {
				continue
			}
			vs = append(vs, x(i))
			shapes = append(shapes, in[i].shape())
		}
		a := axis(int(n.Int("axis", 0)), len(shapes[0]))
		shape := slices.Clone(shapes[0])
		shape[a] = 0
		for _, s := range shapes {
			shape[a] += s[a]
		}
		if len(vs) == 1 {
			out(vs[0])
			return
		}
		out(l.op("concat", map[string]any{"values": vs, "axis": l.B.ScalarI32(int32(a)), "interleave": l.B.ScalarBool(false)}, shape...))
	case "Slice":
		var axes, steps []int
		if t := c(3); t != nil {
			axes = t.Ints()
		}
		if t := c(4); t != nil {
			steps = t.Ints()
		}
		begin, end, step := SliceParams(in[0].shape(), c(1).Ints(), c(2).Ints(), axes, steps)
		out(l.slice(x(0), begin, end, step))
	case "Split":
		s := in[0].shape()
		a := axis(int(n.Int("axis", 0)), len(s))
		sizes := n.IntList("split")
		if t := c(1); t != nil {
			sizes = t.Ints()
		}
		if sizes == nil {
			for range n.Out {
				sizes = append(sizes, s[a]/len(n.Out))
			}
		}
		at := 0
		for i, sz := range sizes {
			begin, end, step := make([]int, len(s)), slices.Clone(s), make([]int, len(s))
			for j := range step {
				step[j] = 1
			}
			begin[a], end[a] = at, at+sz
			set(i, val{v: l.slice(x(0), begin, end, step)})
			at += sz
		}
	case "Gather":
		idx := c(1)
		s := in[0].shape()
		a := axis(int(n.Int("axis", 0)), len(s))
		if idx == nil || len(idx.Shape) > 1 {
			l.Fail("Gather %s: a program value gathered by %v", n.Name, n.In[1])
		}
		// contiguous indices are a slice
		first := int(idx.F[0])
		if first < 0 {
			first += s[a]
		}
		for i, v := range idx.F {
			if int(v) != first+i && int(v)+s[a] != first+i {
				l.Fail("Gather %s: indices %v are not contiguous", n.Name, idx.F)
			}
		}
		begin, end, step := make([]int, len(s)), slices.Clone(s), make([]int, len(s))
		for j := range step {
			step[j] = 1
		}
		begin[a], end[a] = first, first+len(idx.F)
		v := l.slice(x(0), begin, end, step)
		if len(idx.Shape) == 0 {
			v = l.reshape(v, slices.Delete(slices.Clone(v.Shape), a, a+1))
		}
		out(v)
	case "ReduceMean", "ReduceSum", "ReduceMax":
		s := in[0].shape()
		axes := n.IntList("axes")
		if t := c(1); t != nil {
			axes = t.Ints()
		}
		if len(axes) == 0 {
			for i := range s {
				axes = append(axes, i)
			}
		}
		keep := n.Int("keepdims", 1) != 0
		shape := slices.Clone(s)
		var norm []int
		for _, a := range axes {
			a = axis(a, len(s))
			norm = append(norm, a)
			shape[a] = 1
		}
		typ := map[string]string{"ReduceMean": "reduce_mean", "ReduceSum": "reduce_sum", "ReduceMax": "reduce_max"}[n.Op]
		v := l.op(typ, map[string]any{"x": x(0), "axes": l.ints(norm...), "keep_dims": l.B.ScalarBool(true)}, shape...)
		if !keep {
			var sq []int
			for i, d := range shape {
				if !slices.Contains(norm, i) {
					sq = append(sq, d)
				}
			}
			v = l.reshape(v, sq)
		}
		out(v)
	case "Pad":
		s := in[0].shape()
		pads := n.IntList("pads")
		if t := c(1); t != nil {
			pads = t.Ints()
		}
		cval := n.Float("value", 0)
		if t := c(2); t != nil && len(t.F) > 0 {
			cval = t.F[0]
		}
		var axes []int
		if t := c(3); t != nil {
			axes = t.Ints()
		}
		full := FullPads(len(s), pads, axes)
		mode := n.Str("mode", "constant")
		shape := slices.Clone(s)
		first := -1 // MIL pads the last axes: from the first one padded
		for i := range s {
			if full[2*i] < 0 || full[2*i+1] < 0 {
				l.Fail("Pad %s: a negative pad", n.Name)
			}
			shape[i] += full[2*i] + full[2*i+1]
			if first < 0 && (full[2*i] != 0 || full[2*i+1] != 0) {
				first = i
			}
		}
		if first < 0 {
			set(0, in[0])
			return
		}
		if mode != "constant" && mode != "reflect" && mode != "edge" {
			l.Fail("Pad %s: mode %s", n.Name, mode)
		}
		if mode == "edge" {
			mode = "replicate"
		}
		out(l.op("pad", map[string]any{"x": x(0), "pad": l.ints(full[2*first:]...), "mode": l.Str(mode), "constant_val": l.B.ScalarF16(float32(cval))}, shape...))
	case "Gelu":
		mode := "EXACT"
		if n.Int("approximate", 0) != 0 || n.Str("approximate", "none") == "tanh" {
			mode = "TANH_APPROXIMATION"
		}
		out(l.op("gelu", map[string]any{"x": x(0), "mode": l.Str(mode)}, x(0).Shape...))
	case "Where":
		// the condition a program value of 0 and 1 (a program has no booleans; the caller gives
		// it as an input): cond a + (1 - cond) b, exact for either (b + cond (a - b) would round
		// a - b to fp16's coarse steps when b is a mask's large constant), constants clamped
		clamp := func(i int) *mlpackage.Var {
			if in[i].c == nil {
				return x(i)
			}
			t := &Tensor{Shape: in[i].c.Shape, Type: Float, F: make([]float64, len(in[i].c.F))}
			for j, v := range in[i].c.F {
				t.F[j] = math.Max(-3e4, math.Min(3e4, v))
			}
			return l.v(val{c: t})
		}
		ab, err := Broadcast(in[1].shape(), in[2].shape())
		shape := ab
		if err == nil {
			shape, err = Broadcast(in[0].shape(), ab)
		}
		if err != nil {
			l.Fail("Where %s: %v", n.Name, err)
		}
		a, b := clamp(1), clamp(2)
		cs := in[0].shape()
		notc := l.op("sub", map[string]any{"x": l.B.ScalarF16(1), "y": x(0)}, cs...)
		ca, err := Broadcast(cs, in[1].shape())
		if err != nil {
			l.Fail("Where %s: %v", n.Name, err)
		}
		cb, err := Broadcast(cs, in[2].shape())
		if err != nil {
			l.Fail("Where %s: %v", n.Name, err)
		}
		ta := l.op("mul", map[string]any{"x": x(0), "y": a}, ca...)
		tb := l.op("mul", map[string]any{"x": notc, "y": b}, cb...)
		out(l.op("add", map[string]any{"x": ta, "y": tb}, shape...))
	case "Resize", "Upsample":
		l.resize(n, in)
	case "Expand", "Tile":
		s := in[0].shape()
		var reps []int
		if n.Op == "Tile" {
			reps = c(1).Ints()
		} else {
			shape, err := Broadcast(s, c(1).Ints())
			if err != nil {
				l.Fail("Expand %s: %v", n.Name, err)
			}
			v := x(0)
			if len(shape) > len(s) {
				v = l.reshape(v, append(slices.Repeat([]int{1}, len(shape)-len(s)), s...))
				s = v.Shape
			}
			for i := range shape {
				reps = append(reps, shape[i]/s[i])
			}
			in[0] = val{v: v}
		}
		shape := slices.Clone(in[0].shape())
		for i := range shape {
			shape[i] *= reps[i]
		}
		out(l.op("tile", map[string]any{"x": x(0), "reps": l.ints(reps...)}, shape...))
	default:
		l.Fail("%s %s: not lowered to MIL", n.Op, n.Name)
	}
}

func toInt64(v []int) []int64 {
	out := make([]int64, len(v))
	for i, x := range v {
		out[i] = int64(x)
	}
	return out
}

func (l *lowering) reshape(v *mlpackage.Var, shape []int) *mlpackage.Var {
	if slices.Equal(v.Shape, shape) {
		return v
	}
	return l.op("reshape", map[string]any{"x": v, "shape": l.ints(shape...)}, shape...)
}

func (l *lowering) slice(v *mlpackage.Var, begin, end, step []int) *mlpackage.Var {
	shape := make([]int, len(begin))
	whole := true
	for i := range begin {
		if step[i] <= 0 {
			l.Fail("a slice of step %d", step[i])
		}
		shape[i] = max(0, (end[i]-begin[i]+step[i]-1)/step[i])
		whole = whole && begin[i] == 0 && end[i] == v.Shape[i] && step[i] == 1
	}
	if whole {
		return v
	}
	return l.op("slice_by_index", map[string]any{"x": v, "begin": l.ints(begin...), "end": l.ints(end...), "stride": l.ints(step...)}, shape...)
}

// weight is a constant input as an fp16 constant named after it.
func (l *lowering) weight(name string, t *Tensor, shape []int) *mlpackage.Var {
	vals := make([]float32, len(t.F))
	for i, f := range t.F {
		vals[i] = float32(f)
	}
	return l.F16s(nameRe.ReplaceAllString(name, "_"), shape, vals)
}

func (l *lowering) conv(n *Node, in []val) {
	if in[1].c == nil {
		l.Fail("Conv %s: the weight must be a constant", n.Name)
	}
	if ap := n.Str("auto_pad", "NOTSET"); ap != "NOTSET" && ap != "VALID" {
		l.Fail("Conv %s: auto_pad %s", n.Name, ap)
	}
	s, w := in[0].shape(), in[1].c
	k := len(w.Shape) - 2
	if k != 1 && k != 2 {
		l.Fail("Conv %s: %d spatial axes", n.Name, k)
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
	pad := make([]int, 2*k) // MIL: begin, end per axis
	shape := []int{s[0], w.Shape[0]}
	for i := range k {
		pad[2*i], pad[2*i+1] = pads[i], pads[k+i]
		shape = append(shape, (s[2+i]+pads[i]+pads[k+i]-dil[i]*(w.Shape[2+i]-1)-1)/strides[i]+1)
	}
	args := map[string]any{
		"x": l.v(in[0]), "weight": l.weight(n.In[1], w, w.Shape),
		"strides": l.ints(strides...), "pad_type": l.Str("custom"), "pad": l.ints(pad...),
		"dilations": l.ints(dil...), "groups": l.B.ScalarI32(int32(n.Int("group", 1))),
	}
	if len(n.In) > 2 && n.In[2] != "" {
		if in[2].c == nil {
			l.Fail("Conv %s: the bias must be a constant", n.Name)
		}
		args["bias"] = l.weight(n.In[2], in[2].c, in[2].c.Shape)
	}
	l.vals[n.Out[0]] = val{v: l.op("conv", args, shape...)}
}

func (l *lowering) maxPool(n *Node, in []val) {
	s := in[0].shape()
	k := n.IntList("kernel_shape")
	strides, pads := n.IntList("strides"), n.IntList("pads")
	if strides == nil {
		strides = slices.Repeat([]int{1}, len(k))
	}
	if pads == nil {
		pads = make([]int, 2*len(k))
	}
	if n.Int("ceil_mode", 0) != 0 || n.IntList("dilations") != nil && slices.Max(n.IntList("dilations")) > 1 {
		l.Fail("MaxPool %s: ceil_mode or dilations", n.Name)
	}
	// a pool of stride 1 that keeps the size, with a square window over 5: pools of 5 (and a
	// 3) one after another, the same maxima. Core ML puts windows of 9 and 13 on the CPU (M1).
	same := len(k) == 2 && k[0] == k[1] && k[0]%2 == 1 && strides[0] == 1 && strides[1] == 1 && slices.Equal(pads, slices.Repeat([]int{k[0] / 2}, 4))
	if same && k[0] > 5 {
		v := l.v(in[0])
		for left := k[0] - 1; left > 0; {
			step := min(4, left)
			if step == 3 {
				step = 2
			}
			w := step + 1
			v = l.op("max_pool", map[string]any{
				"x": v, "kernel_sizes": l.ints(w, w), "strides": l.ints(1, 1),
				"pad_type": l.Str("custom"), "pad": l.ints(w/2, w/2, w/2, w/2), "ceil_mode": l.B.ScalarBool(false),
			}, v.Shape...)
			left -= step
		}
		l.vals[n.Out[0]] = val{v: v}
		return
	}
	pad := make([]int, 2*len(k))
	shape := []int{s[0], s[1]}
	for i := range k {
		pad[2*i], pad[2*i+1] = pads[i], pads[len(k)+i]
		shape = append(shape, (s[2+i]+pads[i]+pads[len(k)+i]-k[i])/strides[i]+1)
	}
	l.vals[n.Out[0]] = val{v: l.op("max_pool", map[string]any{
		"x": l.v(in[0]), "kernel_sizes": l.ints(k...), "strides": l.ints(strides...),
		"pad_type": l.Str("custom"), "pad": l.ints(pad...), "ceil_mode": l.B.ScalarBool(false),
	}, shape...)}
}

func (l *lowering) matmul(n *Node, in []val) {
	a, b := in[0].shape(), in[1].shape()
	if len(a) < 2 || len(b) < 2 {
		l.Fail("MatMul %s of %v and %v", n.Name, a, b)
	}
	if in[1].c != nil && len(b) > 2 && Len(b[:len(b)-2]) == 1 && len(a) >= len(b) {
		// a constant of leading 1s ((1, k, n)): as (k, n)
		b = b[len(b)-2:]
		in[1] = val{c: &Tensor{Shape: b, F: in[1].c.F, Type: in[1].c.Type}}
	}
	if in[1].c != nil && len(b) == 2 {
		// x (..., k) times a constant (k, n): a linear layer with the weight (n, k)
		k, nn := b[0], b[1]
		w := transpose(in[1].c, []int{1, 0})
		shape := append(slices.Clone(a[:len(a)-1]), nn)
		if a[len(a)-1] != k {
			l.Fail("MatMul %s of %v and %v", n.Name, a, b)
		}
		l.vals[n.Out[0]] = val{v: l.op("linear", map[string]any{"x": l.v(in[0]), "weight": l.weight(n.In[1], w, w.Shape)}, shape...)}
		return
	}
	batch, err := Broadcast(a[:len(a)-2], b[:len(b)-2])
	if err != nil {
		l.Fail("MatMul %s: %v", n.Name, err)
	}
	shape := append(batch, a[len(a)-2], b[len(b)-1])
	l.vals[n.Out[0]] = val{v: l.op("matmul", map[string]any{"x": l.v(in[0]), "y": l.v(in[1]),
		"transpose_x": l.B.ScalarBool(false), "transpose_y": l.B.ScalarBool(false)}, shape...)}
}

func (l *lowering) gemm(n *Node, in []val) {
	if in[1].c == nil || n.Int("transA", 0) != 0 || n.Float("alpha", 1) != 1 || n.Float("beta", 1) != 1 {
		l.Fail("Gemm %s: only x times a constant, alpha and beta 1", n.Name)
	}
	w := in[1].c
	if n.Int("transB", 0) == 0 {
		w = transpose(w, []int{1, 0})
	}
	a := in[0].shape()
	args := map[string]any{"x": l.v(in[0]), "weight": l.weight(n.In[1], w, w.Shape)}
	if len(in) > 2 && in[2].c != nil {
		bias := in[2].c
		if len(bias.F) == 1 && w.Shape[0] > 1 {
			bias = &Tensor{Shape: []int{w.Shape[0]}, F: slices.Repeat(bias.F, w.Shape[0])}
		}
		args["bias"] = l.weight(n.In[2], bias, []int{w.Shape[0]})
	}
	l.vals[n.Out[0]] = val{v: l.op("linear", args, append(slices.Clone(a[:len(a)-1]), w.Shape[0])...)}
}

func (l *lowering) resize(n *Node, in []val) {
	if mode := n.Str("mode", "nearest"); mode != "nearest" {
		l.Fail("Resize %s: mode %s", n.Name, mode)
	}
	s := in[0].shape()
	var sh, sw int
	switch {
	case len(in) > 3 && in[3].c != nil && len(in[3].c.F) == 4:
		sizes := in[3].c.Ints()
		sh, sw = sizes[2]/s[2], sizes[3]/s[3]
		if sizes[2] != sh*s[2] || sizes[3] != sw*s[3] || sizes[0] != s[0] || sizes[1] != s[1] {
			l.Fail("Resize %s to %v: not a whole scale", n.Name, sizes)
		}
	case len(in) > 2 && in[2].c != nil && len(in[2].c.F) == 4:
		sc := in[2].c.F
		sh, sw = int(sc[2]), int(sc[3])
		if sc[0] != 1 || sc[1] != 1 || float64(sh) != sc[2] || float64(sw) != sc[3] {
			l.Fail("Resize %s by %v: not a whole scale", n.Name, sc)
		}
	case n.Op == "Upsample" && len(in) > 1 && in[1].c != nil:
		sc := in[1].c.F
		sh, sw = int(sc[2]), int(sc[3])
	default:
		l.Fail("Resize %s: no constant scales or sizes", n.Name)
	}
	l.vals[n.Out[0]] = val{v: l.op("upsample_nearest_neighbor", map[string]any{"x": l.v(in[0]),
		"scale_factor_height": l.B.ScalarI32(int32(sh)), "scale_factor_width": l.B.ScalarI32(int32(sw))}, s[0], s[1], s[2]*sh, s[3]*sw)}
}
