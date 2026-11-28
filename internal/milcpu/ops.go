package milcpu

import (
	"fmt"
	"math"
	"runtime"
	"sync"
	"unsafe"

	"github.com/shibukawa/hwnpu/internal/f32"
	"github.com/shibukawa/hwnpu/internal/milspec"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/sgemm"
	"github.com/shibukawa/hwnpu/internal/tensorops"
	"github.com/shibukawa/hwnpu/internal/vecops"
)

// arg returns the operation's input called name: a bound value or an inline constant. ok is false when
// the input is absent (optional inputs).
func (p *Program) arg(op *milspec.Operation, vals map[string]*value, name string) (v *value, ok bool, err error) {
	a, ok := op.GetInputs()[name]
	if !ok || len(a.GetArguments()) == 0 {
		return nil, false, nil
	}
	return p.binding(a.GetArguments()[0], vals)
}

func (p *Program) binding(b *milspec.Argument_Binding, vals map[string]*value) (*value, bool, error) {
	if n := b.GetName(); n != "" {
		v, ok := vals[n]
		if !ok {
			return nil, false, fmt.Errorf("value %s is not defined", n)
		}
		return v, true, nil
	}
	// an inline constant (a stride, a pad, an axis) decoded once: no operation writes into its
	// inputs, and Run holds p.mu
	bv := b.GetValue()
	if v, ok := p.inline[bv]; ok {
		return v, true, nil
	}
	v, err := p.decode(bv)
	if err == nil {
		if p.inline == nil {
			p.inline = map[*milspec.Value]*value{}
		}
		p.inline[bv] = v
	}
	return v, err == nil, err
}

// fixed reports whether the operation's input name is a constant of the program (an inline
// value, or a const operation's), whose buffer stays from run to run; an absent input is too.
func (p *Program) fixed(op *milspec.Operation, name string) bool {
	a, ok := op.GetInputs()[name]
	if !ok || len(a.GetArguments()) == 0 {
		return true
	}
	n := a.GetArguments()[0].GetName()
	return n == "" || p.constOut[n]
}

// reader reads the named inputs of an operation and keeps the first error.
type reader struct {
	p    *Program
	op   *milspec.Operation
	vals map[string]*value
	err  error
}

func (r *reader) get(name string) *value {
	if r.err != nil {
		return nil
	}
	v, ok, err := r.p.arg(r.op, r.vals, name)
	if err != nil {
		r.err = err
	} else if !ok {
		r.err = fmt.Errorf("input %s missing", name)
	}
	return v
}

func (r *reader) opt(name string) *value {
	if r.err != nil {
		return nil
	}
	v, _, err := r.p.arg(r.op, r.vals, name)
	if err != nil {
		r.err = err
	}
	return v
}

func (r *reader) floats(name string) *value {
	v := r.get(name)
	if r.err == nil && v.f == nil {
		r.err = fmt.Errorf("input %s is not a float tensor", name)
	}
	return v
}

func (r *reader) ints(name string, def ...int32) []int32 {
	v := r.opt(name)
	if r.err != nil {
		return nil
	}
	if v == nil {
		if def == nil {
			r.err = fmt.Errorf("input %s missing", name)
		}
		return def
	}
	if v.i == nil && v.q == nil {
		r.err = fmt.Errorf("input %s is not an int tensor", name)
	}
	return v.ints()
}

func (r *reader) int(name string, def int32) int32 {
	v := r.ints(name, def)
	if len(v) == 0 {
		return def
	}
	return v[0]
}

func (r *reader) bools(name string, n int) []bool {
	v := r.opt(name)
	if r.err != nil || v == nil {
		return make([]bool, n)
	}
	return v.b
}

func (r *reader) bool(name string) bool {
	b := r.bools(name, 1)
	return len(b) > 0 && b[0]
}

func (r *reader) float(name string, def float32) float32 {
	v := r.opt(name)
	if r.err != nil || v == nil || len(v.f) == 0 {
		return def
	}
	return v.f[0]
}

func (r *reader) str(name, def string) string {
	v := r.opt(name)
	if r.err != nil || v == nil {
		return def
	}
	return v.s
}

func (p *Program) eval(op *milspec.Operation, vals map[string]*value) (*value, error) {
	r := &reader{p: p, op: op, vals: vals}
	var out *value
	switch typ := op.GetType(); typ {
	case "const":
		return p.decodeAs(op.GetAttributes()["val"], p.f16Linear[op.GetOutputs()[0].GetName()])
	case "identity":
		x := r.floats("x")
		if r.err == nil {
			out = &value{shape: x.shape, f: append([]float32(nil), x.f...)}
		}
	case "clip":
		x := r.floats("x")
		lo, hi := r.float("alpha", 0), r.float("beta", 0)
		if r.err == nil {
			out = &value{shape: x.shape, f: make([]float32, len(x.f))}
			for j, v := range x.f {
				out.f[j] = min(max(v, lo), hi)
			}
		}
	case "cast": // between fp16 and fp32: the interpreter computes in fp32; to fp16 rounds
		x := r.floats("x")
		dt := r.str("dtype", "fp32")
		if r.err == nil {
			out = &value{shape: x.shape, f: append([]float32(nil), x.f...)}
			if dt == "fp16" {
				for j, v := range out.f {
					out.f[j] = mlpackage.Float16Value(mlpackage.Float16Bits(v))
				}
			}
		}
	case "add", "sub", "mul", "real_div", "maximum", "minimum":
		x, y := r.floats("x"), r.floats("y")
		if r.err == nil {
			out, r.err = elementBinary(&p.pool, x, y, typ)
		}
	case "silu", "sigmoid", "relu", "tanh", "exp", "sqrt", "softplus", "abs", "erf", "cos", "sin", "elu":
		x := r.floats("x")
		if r.err == nil {
			out = unary(&p.pool, x, typ)
		}
	case "log":
		x := r.floats("x")
		eps := r.float("epsilon", 1e-45)
		if r.err == nil {
			out = &value{shape: x.shape, f: make([]float32, len(x.f))}
			for i, v := range x.f {
				out.f[i] = float32(math.Log(float64(v) + float64(eps)))
			}
		}
	case "rsqrt":
		x := r.floats("x")
		eps := r.float("epsilon", 1e-12)
		if r.err == nil {
			out = &value{shape: x.shape, f: make([]float32, len(x.f))}
			for i, v := range x.f {
				out.f[i] = float32(1 / math.Sqrt(float64(v)+float64(eps)))
			}
		}
	case "gelu":
		x := r.floats("x")
		mode := r.str("mode", "EXACT")
		if r.err == nil {
			out, r.err = gelu(x, mode)
		}
	case "layer_norm":
		x := r.floats("x")
		axes := r.ints("axes")
		gamma, beta := r.opt("gamma"), r.opt("beta")
		eps := r.float("epsilon", 1e-5)
		if r.err == nil {
			out, r.err = layerNorm(x, axes, gamma, beta, eps)
		}
	case "softmax":
		x := r.floats("x")
		axis := r.int("axis", -1)
		if r.err == nil {
			out, r.err = softmax(x, int(axis))
		}
	case "reduce_mean", "reduce_sum", "reduce_max":
		x := r.floats("x")
		axes := r.ints("axes")
		keep := r.bool("keep_dims")
		if r.err == nil {
			out, r.err = reduce(x, axes, keep, typ)
		}
	case "linear":
		x, w := r.floats("x"), r.get("weight")
		b := r.opt("bias")
		if r.err == nil && w.f == nil && w.lut == nil && w.f16 == nil {
			r.err = fmt.Errorf("input weight is not a float tensor")
		}
		if r.err == nil {
			out, r.err = linear(x, w, b)
		}
	case "matmul":
		x, y := r.floats("x"), r.floats("y")
		tx, ty := r.bool("transpose_x"), r.bool("transpose_y")
		if r.err == nil {
			out, r.err = matmul(x, y, tx, ty)
		}
	case "reshape":
		x := r.floats("x")
		shape := r.ints("shape")
		if r.err == nil {
			out, r.err = reshape(x, shape)
		}
	case "transpose":
		x := r.floats("x")
		perm := r.ints("perm")
		if r.err == nil {
			out, r.err = transpose(&p.pool, x, perm)
		}
	case "tile":
		x := r.floats("x")
		reps := r.ints("reps")
		if r.err == nil {
			out, r.err = tile(x, reps)
		}
	case "pixel_shuffle":
		x := r.floats("x")
		f := r.int("upscale_factor", 1)
		if r.err == nil {
			out, r.err = pixelShuffle(x, int(f))
		}
	case "upsample_nearest_neighbor":
		x := r.floats("x")
		sh, sw := r.int("scale_factor_height", 1), r.int("scale_factor_width", 1)
		if r.err == nil {
			out, r.err = upsampleNearest(&p.pool, x, int(sh), int(sw))
		}
	case "slice_by_index":
		x := r.floats("x")
		n := len(x.shape)
		begin, end := r.ints("begin"), r.ints("end")
		stride := r.ints("stride", ones(n)...)
		bm, em, sm := r.bools("begin_mask", n), r.bools("end_mask", n), r.bools("squeeze_mask", n)
		if r.err == nil {
			out, r.err = sliceByIndex(x, begin, end, stride, bm, em, sm)
		}
	case "pad":
		x := r.floats("x")
		pad := r.ints("pad")
		mode := r.str("mode", "constant")
		c := r.float("constant_val", 0)
		if r.err == nil {
			switch mode {
			case "constant":
				out, r.err = padConst(x, pad, c)
			case "reflect", "replicate":
				out, r.err = padEdges(x, pad, mode)
			default:
				return nil, fmt.Errorf("pad mode %q is not supported", mode)
			}
		}
	case "concat":
		a := op.GetInputs()["values"]
		axis := r.int("axis", 0)
		if r.bool("interleave") {
			return nil, fmt.Errorf("interleaved concat is not supported")
		}
		var xs []*value
		for _, b := range a.GetArguments() {
			v, _, err := p.binding(b, vals)
			if err != nil {
				return nil, err
			}
			xs = append(xs, v)
		}
		if r.err == nil {
			out, r.err = concat(&p.pool, xs, int(axis))
		}
	case "pow":
		x, y := r.floats("x"), r.floats("y")
		if r.err == nil {
			out, r.err = broadcast(x, y, func(a, b float32) float32 { return float32(math.Pow(float64(a), float64(b))) })
		}
	case "conv_transpose":
		x, w := r.floats("x"), r.floats("weight")
		b := r.opt("bias")
		st := r.ints("strides", 1, 1)
		pad := r.ints("pad", 0, 0, 0, 0)
		var shape []int32
		if v := r.opt("output_shape"); v != nil {
			for _, d := range v.f {
				shape = append(shape, int32(d))
			}
			if v.i != nil {
				shape = shape[:0]
				for _, d := range v.i {
					shape = append(shape, int32(d))
				}
			}
		}
		if r.err == nil {
			out, r.err = convTranspose2d(x, w, b, st, pad, shape)
		}
	case "max_pool":
		x := r.floats("x")
		k, st := r.ints("kernel_sizes"), r.ints("strides", 1, 1)
		padType := r.str("pad_type", "valid")
		pad := r.ints("pad", 0, 0, 0, 0)
		if r.err == nil {
			if padType != "custom" && padType != "valid" {
				return nil, fmt.Errorf("pad_type %q is not supported", padType)
			}
			if padType == "valid" {
				pad = []int32{0, 0, 0, 0}
			}
			out, r.err = maxPool2d(&p.pool, x, k, st, pad)
		}
	case "conv":
		x, w := r.floats("x"), r.floats("weight")
		b := r.opt("bias")
		strides, dil := r.ints("strides", 1), r.ints("dilations", 1)
		padType := r.str("pad_type", "valid")
		pad := r.ints("pad", 0, 0)
		groups := r.int("groups", 1)
		if r.err == nil {
			if len(x.shape) == 4 {
				out, r.err = conv2d(&p.pool, x, w, b, strides, dil, padType, pad, int(groups), p.reluNext, p.fixed(op, "weight") && p.fixed(op, "bias"))
			} else {
				out, r.err = conv1d(&p.pool, p.fixed(op, "weight") && p.fixed(op, "bias"), x, w, b, strides, dil, padType, pad, int(groups))
			}
		}
	case "constexpr_lut_to_dense":
		idx, lut := r.get("indices"), r.floats("lut")
		if r.err == nil {
			if p.lutLinear[op.GetOutputs()[0].GetName()] {
				out, r.err = newLUTWeight(idx, lut)
			} else {
				out, r.err = lutToDense(idx, lut)
			}
		}
	case "constexpr_blockwise_shift_scale":
		data, scale := r.get("data"), r.floats("scale")
		offset := r.opt("offset")
		if r.err == nil {
			out, r.err = shiftScale(data, scale, offset)
		}
	default:
		return nil, fmt.Errorf("not supported")
	}
	if r.err != nil {
		return nil, r.err
	}
	return out, nil
}

func ones(n int) []int32 {
	v := make([]int32, n)
	for i := range v {
		v[i] = 1
	}
	return v
}

func strides(shape []int) []int { return tensorops.Strides(shape) }

func axis(a, rank int) (int, error) {
	if a < 0 {
		a += rank
	}
	if a < 0 || a >= rank {
		return 0, fmt.Errorf("axis %d out of range for rank %d", a, rank)
	}
	return a, nil
}

// parallel runs f over [0, n) in chunks on all CPUs.
func parallel(n int, f func(lo, hi int)) {
	workers := runtime.GOMAXPROCS(0)
	if Threads > 0 {
		workers = Threads
	}
	if workers > n {
		workers = n
	}
	if workers <= 1 {
		f(0, n)
		return
	}
	var wg sync.WaitGroup
	chunk := (n + workers - 1) / workers
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func() {
			defer wg.Done()
			f(lo, hi)
		}()
	}
	wg.Wait()
}

func binaryOp(typ string) func(a, b float32) float32 {
	switch typ {
	case "add":
		return func(a, b float32) float32 { return a + b }
	case "sub":
		return func(a, b float32) float32 { return a - b }
	case "mul":
		return func(a, b float32) float32 { return a * b }
	case "real_div":
		return func(a, b float32) float32 { return a / b }
	case "maximum":
		return func(a, b float32) float32 { return max(a, b) }
	default:
		return func(a, b float32) float32 { return min(a, b) }
	}
}

// broadcast applies f elementwise with numpy broadcasting.
func broadcast(x, y *value, f func(a, b float32) float32) (*value, error) {
	rank := max(len(x.shape), len(y.shape))
	shape := make([]int, rank)
	xs, ys := make([]int, rank), make([]int, rank)
	xst, yst := strides(x.shape), strides(y.shape)
	for i := 0; i < rank; i++ {
		xd, yd := 1, 1
		if j := i - (rank - len(x.shape)); j >= 0 {
			xd = x.shape[j]
			if xd != 1 {
				xs[i] = xst[j]
			}
		}
		if j := i - (rank - len(y.shape)); j >= 0 {
			yd = y.shape[j]
			if yd != 1 {
				ys[i] = yst[j]
			}
		}
		switch {
		case xd == yd || yd == 1:
			shape[i] = xd
		case xd == 1:
			shape[i] = yd
		default:
			return nil, fmt.Errorf("shapes %v and %v do not broadcast", x.shape, y.shape)
		}
	}
	n := numel(shape)
	out := &value{shape: shape, f: make([]float32, n)}
	if len(x.f) == n && len(y.f) == n {
		for i := range out.f {
			out.f[i] = f(x.f[i], y.f[i])
		}
		return out, nil
	}
	idx := make([]int, rank)
	xo, yo := 0, 0
	for i := 0; i < n; i++ {
		out.f[i] = f(x.f[xo], y.f[yo])
		for d := rank - 1; d >= 0; d-- {
			idx[d]++
			xo += xs[d]
			yo += ys[d]
			if idx[d] < shape[d] {
				break
			}
			xo -= xs[d] * idx[d]
			yo -= ys[d] * idx[d]
			idx[d] = 0
		}
	}
	if rank == 0 {
		out.shape = nil
	}
	return out, nil
}

func unary(pl *pool, x *value, typ string) *value {
	out := &value{shape: x.shape, f: pl.get(len(x.f))}
	var fn func(float64) float64
	switch typ {
	case "silu":
		elementwise(len(x.f), func(lo, hi int) { f32.SiLU(out.f[lo:hi], x.f[lo:hi]) })
		return out
	case "sigmoid":
		elementwise(len(x.f), func(lo, hi int) { f32.Sigmoid(out.f[lo:hi], x.f[lo:hi]) })
		return out
	case "exp":
		elementwise(len(x.f), func(lo, hi int) { f32.Exp(out.f[lo:hi], x.f[lo:hi]) })
		return out
	case "relu":
		elementwise(len(x.f), func(lo, hi int) { f32.ReLU(out.f[lo:hi], x.f[lo:hi]) })
		return out
	case "tanh":
		fn = math.Tanh
	case "sqrt":
		fn = math.Sqrt
	case "abs":
		fn = math.Abs
	case "softplus":
		// log(1 + exp(a)), without overflowing exp for large a
		fn = func(a float64) float64 { return math.Max(a, 0) + math.Log1p(math.Exp(-math.Abs(a))) }
	case "erf":
		elementwise(len(x.f), func(lo, hi int) { f32.Erf(out.f[lo:hi], x.f[lo:hi]) })
		return out
	case "elu": // alpha 1
		fn = func(a float64) float64 {
			if a < 0 {
				return math.Expm1(a)
			}
			return a
		}
	case "cos":
		fn = math.Cos
	case "sin":
		fn = math.Sin
	default:
		fn = func(a float64) float64 { return a }
	}
	elementwise(len(x.f), func(lo, hi int) {
		for i, v := range x.f[lo:hi] {
			out.f[lo+i] = float32(fn(float64(v)))
		}
	})
	return out
}

// elementwise runs f over n elements, in parts on Threads goroutines when n is large.
func elementwise(n int, f func(lo, hi int)) {
	if Threads == 1 || n < 1<<16 {
		f(0, n)
		return
	}
	parallel(n, f)
}

// elementBinary is an elementwise add, sub, mul, real_div, maximum or minimum of x and y broadcast
// (numpy's rules): the dimensions collapsed into runs where each operand moves by one or not
// at all, and a typed loop over each run.
func elementBinary(pl *pool, x, y *value, typ string) (*value, error) {
	rank := max(len(x.shape), len(y.shape))
	shape := make([]int, rank)
	xs, ys := make([]int, rank), make([]int, rank)
	xst, yst := strides(x.shape), strides(y.shape)
	for i := 0; i < rank; i++ {
		xd, yd := 1, 1
		if j := i - (rank - len(x.shape)); j >= 0 {
			xd = x.shape[j]
			if xd != 1 {
				xs[i] = xst[j]
			}
		}
		if j := i - (rank - len(y.shape)); j >= 0 {
			yd = y.shape[j]
			if yd != 1 {
				ys[i] = yst[j]
			}
		}
		switch {
		case xd == yd || yd == 1:
			shape[i] = xd
		case xd == 1:
			shape[i] = yd
		default:
			return nil, fmt.Errorf("shapes %v and %v do not broadcast", x.shape, y.shape)
		}
	}
	n := numel(shape)
	out := &value{shape: shape, f: pl.get(n)}
	if rank == 0 {
		out.shape = nil
	}
	// collapse: dims d-1 and d merge when each operand's stride at d-1 is its stride at d times
	// shape[d] (or both zero)
	var cs, cx, cy []int
	for d := 0; d < rank; d++ {
		if shape[d] == 1 {
			continue
		}
		if k := len(cs) - 1; k >= 0 && cx[k] == xs[d]*shape[d] && cy[k] == ys[d]*shape[d] {
			cs[k] *= shape[d]
			cx[k], cy[k] = xs[d], ys[d]
			continue
		}
		cs, cx, cy = append(cs, shape[d]), append(cx, xs[d]), append(cy, ys[d])
	}
	if len(cs) == 0 {
		cs, cx, cy = []int{1}, []int{1}, []int{1}
	}
	last := len(cs) - 1
	inner, ix, iy := cs[last], cx[last], cy[last]
	if (ix != 0 && ix != 1) || (iy != 0 && iy != 1) {
		inner, ix, iy = 1, 0, 0 // a run of one: the general walk
		cs, cx, cy = append(cs, 1), append(cx, 0), append(cy, 0)
		last++
	}
	kernel := binaryRun(typ, ix, iy)
	outer := n / inner
	elementwise(outer*max(1, inner/64), func(lo, hi int) {
		scale := max(1, inner/64)
		lo, hi = lo/scale, (hi+scale-1)/scale
		if lo >= outer {
			return
		}
		hi = min(hi, outer)
		// the operands' offsets of run lo
		idx := make([]int, last)
		r := lo
		xo, yo := 0, 0
		for d := last - 1; d >= 0; d-- {
			idx[d] = r % cs[d]
			r /= cs[d]
			xo += idx[d] * cx[d]
			yo += idx[d] * cy[d]
		}
		for run := lo; run < hi; run++ {
			kernel(out.f[run*inner:(run+1)*inner], x.f[xo:], y.f[yo:])
			for d := last - 1; d >= 0; d-- {
				idx[d]++
				xo += cx[d]
				yo += cy[d]
				if idx[d] < cs[d] {
					break
				}
				xo -= cx[d] * idx[d]
				yo -= cy[d] * idx[d]
				idx[d] = 0
			}
		}
	})
	return out, nil
}

// binaryRun is a loop of op over a run: o[i] = op(x[i ix], y[i iy]), ix and iy 0 or 1.
func binaryRun(typ string, ix, iy int) func(o, x, y []float32) {
	f := binaryOp(typ)
	switch {
	case ix == 1 && iy == 1:
		switch typ {
		case "add":
			return func(o, x, y []float32) {
				x, y = x[:len(o)], y[:len(o)]
				for i := range o {
					o[i] = x[i] + y[i]
				}
			}
		case "mul":
			return func(o, x, y []float32) {
				x, y = x[:len(o)], y[:len(o)]
				for i := range o {
					o[i] = x[i] * y[i]
				}
			}
		case "real_div":
			return func(o, x, y []float32) {
				x, y = x[:len(o)], y[:len(o)]
				for i := range o {
					o[i] = x[i] / y[i]
				}
			}
		}
		return func(o, x, y []float32) {
			x, y = x[:len(o)], y[:len(o)]
			for i := range o {
				o[i] = f(x[i], y[i])
			}
		}
	case ix == 1:
		switch typ {
		case "add":
			return func(o, x, y []float32) {
				b := y[0]
				for i, v := range x[:len(o)] {
					o[i] = v + b
				}
			}
		case "mul":
			return func(o, x, y []float32) {
				b := y[0]
				for i, v := range x[:len(o)] {
					o[i] = v * b
				}
			}
		}
		return func(o, x, y []float32) {
			b := y[0]
			for i, v := range x[:len(o)] {
				o[i] = f(v, b)
			}
		}
	case iy == 1:
		switch typ {
		case "add":
			return func(o, x, y []float32) {
				a := x[0]
				for i, v := range y[:len(o)] {
					o[i] = a + v
				}
			}
		case "mul":
			return func(o, x, y []float32) {
				a := x[0]
				for i, v := range y[:len(o)] {
					o[i] = a * v
				}
			}
		case "real_div":
			return func(o, x, y []float32) {
				a := x[0]
				for i, v := range y[:len(o)] {
					o[i] = a / v
				}
			}
		}
		return func(o, x, y []float32) {
			a := x[0]
			for i, v := range y[:len(o)] {
				o[i] = f(a, v)
			}
		}
	}
	return func(o, x, y []float32) {
		v := f(x[0], y[0])
		for i := range o {
			o[i] = v
		}
	}
}

func gelu(x *value, mode string) (*value, error) {
	out := &value{shape: x.shape, f: make([]float32, len(x.f))}
	for i, v := range x.f {
		a := float64(v)
		switch mode {
		case "EXACT":
			a = 0.5 * a * (1 + math.Erf(a/math.Sqrt2))
		case "TANH_APPROXIMATION":
			a = 0.5 * a * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(a+0.044715*a*a*a)))
		case "SIGMOID_APPROXIMATION":
			a = a / (1 + math.Exp(-1.702*a))
		default:
			return nil, fmt.Errorf("gelu mode %q is not supported", mode)
		}
		out.f[i] = float32(a)
	}
	return out, nil
}

// layerNorm normalizes over the trailing axes and applies gamma and beta (over those axes).
func layerNorm(x *value, axes []int32, gamma, beta *value, eps float32) (*value, error) {
	rank := len(x.shape)
	first := rank
	for _, a := range axes {
		ax, err := axis(int(a), rank)
		if err != nil {
			return nil, err
		}
		first = min(first, ax)
	}
	if len(axes) != rank-first {
		return nil, fmt.Errorf("layer_norm over axes %v: only trailing axes are supported", axes)
	}
	n := numel(x.shape[first:])
	out := &value{shape: x.shape, f: make([]float32, len(x.f))}
	for o := 0; o < len(x.f); o += n {
		row := x.f[o : o+n]
		var mean, sq float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(n)
		for _, v := range row {
			d := float64(v) - mean
			sq += d * d
		}
		inv := 1 / math.Sqrt(sq/float64(n)+float64(eps))
		for j, v := range row {
			y := (float64(v) - mean) * inv
			if gamma != nil {
				y *= float64(gamma.f[j])
			}
			if beta != nil {
				y += float64(beta.f[j])
			}
			out.f[o+j] = float32(y)
		}
	}
	return out, nil
}

// outer, n, inner split a shape around one axis.
func around(shape []int, ax int) (outer, n, inner int) {
	outer, inner = 1, 1
	for i, d := range shape {
		switch {
		case i < ax:
			outer *= d
		case i > ax:
			inner *= d
		}
	}
	return outer, shape[ax], inner
}

func softmax(x *value, a int) (*value, error) {
	ax, err := axis(a, len(x.shape))
	if err != nil {
		return nil, err
	}
	outer, n, inner := around(x.shape, ax)
	out := &value{shape: x.shape, f: make([]float32, len(x.f))}
	for o := 0; o < outer; o++ {
		for in := 0; in < inner; in++ {
			base := o*n*inner + in
			m := math.Inf(-1)
			for k := 0; k < n; k++ {
				m = math.Max(m, float64(x.f[base+k*inner]))
			}
			var z float64
			for k := 0; k < n; k++ {
				z += math.Exp(float64(x.f[base+k*inner]) - m)
			}
			for k := 0; k < n; k++ {
				out.f[base+k*inner] = float32(math.Exp(float64(x.f[base+k*inner])-m) / z)
			}
		}
	}
	return out, nil
}

// reduce takes the mean, the sum or the largest (reduce_mean, reduce_sum, reduce_max) over axes.
func reduce(x *value, axes []int32, keep bool, typ string) (*value, error) {
	rank := len(x.shape)
	reduced := make([]bool, rank)
	for _, a := range axes {
		ax, err := axis(int(a), rank)
		if err != nil {
			return nil, err
		}
		reduced[ax] = true
	}
	var shape, kept []int
	count := 1
	for i, d := range x.shape {
		if reduced[i] {
			count *= d
			kept = append(kept, 1)
			if keep {
				shape = append(shape, 1)
			}
		} else {
			kept = append(kept, d)
			shape = append(shape, d)
		}
	}
	sums := make([]float64, numel(kept))
	if typ == "reduce_max" {
		for i := range sums {
			sums[i] = math.Inf(-1)
		}
	}
	kst := strides(kept)
	idx := make([]int, rank)
	for i := range x.f {
		o := 0
		for d := 0; d < rank; d++ {
			if !reduced[d] {
				o += idx[d] * kst[d]
			}
		}
		if typ == "reduce_max" {
			sums[o] = math.Max(sums[o], float64(x.f[i]))
		} else {
			sums[o] += float64(x.f[i])
		}
		for d := rank - 1; d >= 0; d-- {
			if idx[d]++; idx[d] < x.shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	out := &value{shape: shape, f: make([]float32, len(sums))}
	for i, s := range sums {
		if typ == "reduce_mean" {
			s /= float64(count)
		}
		out.f[i] = float32(s)
	}
	return out, nil
}

func dot(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a)
	b = b[:n]
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

// linear is x (..., k) times weight (n, k) transposed, plus bias (n).
func linear(x, w, bias *value) (*value, error) {
	if len(w.shape) != 2 || len(x.shape) == 0 || x.shape[len(x.shape)-1] != w.shape[1] {
		return nil, fmt.Errorf("x %v and weight %v do not match", x.shape, w.shape)
	}
	n, k := w.shape[0], w.shape[1]
	m := len(x.f) / k
	shape := append(append([]int{}, x.shape[:len(x.shape)-1]...), n)
	out := &value{shape: shape, f: make([]float32, m*n)}
	if bias != nil {
		for i := 0; i < m; i++ {
			copy(out.f[i*n:(i+1)*n], bias.f[:n])
		}
	}
	if w.lut != nil && m <= lutRowsMax && k >= 4 && vecops.HaveLUTRows {
		w.lut.mulAdd(m, x.f, out.f)
		return out, nil
	}
	var b sgemm.Source = sgemm.Transposed{Data: w.f, Stride: k, N: n}
	if w.lut != nil {
		b = w.lut
	} else if w.f16 != nil {
		b = w.f16
	}
	sgemm.MulAdd(m, n, k, sgemm.A{Data: x.f, Stride: k}, b, out.f, n, Threads)
	return out, nil
}

// f16Weight is a linear's weight (n, k) in fp16, converted as sgemm packs it: half the float32
// weight's memory.
type f16Weight struct {
	n, k int
	h    []byte
}

// Pack writes a block of the weight's panels, as sgemm.Transposed packs a float32 one.
func (w *f16Weight) Pack(dst []float32, k0, kc, j0, nc int) {
	var buf [sgemm.KC]float32
	row := buf[:0]
	if kc <= len(buf) {
		row = buf[:kc]
	} else {
		row = make([]float32, kc)
	}
	for p := 0; p*sgemm.NR < nc; p++ {
		d := dst[p*kc*sgemm.NR:][:kc*sgemm.NR]
		c0 := j0 + p*sgemm.NR
		cols := min(sgemm.NR, w.n-c0)
		for c := range cols {
			off := (c0+c)*w.k + k0
			vecops.F16ToF32(row, w.h[2*off:2*(off+kc)])
			for kk, v := range row {
				d[kk*sgemm.NR+c] = v
			}
		}
		if cols < sgemm.NR {
			for kk := range kc {
				clear(d[kk*sgemm.NR+cols : kk*sgemm.NR+sgemm.NR])
			}
		}
	}
}

// lutWeight is a linear's weight (n, k) as its 8 bit table (constexpr_lut_to_dense): the index of
// each element, a byte, and the table of each block of the grid (gn, gk), read as sgemm packs
// it. A quarter of the float32 weight's memory, which it is not made into.
type lutWeight struct {
	n, k, size int
	bn, bk, gk int // a block's rows and columns, the grid's columns
	idx        []byte
	table      []float32 // (gn, gk, size)
}

// newLUTWeight is the lutWeight of a constexpr_lut_to_dense of indices idx (n, k) and lut lut
// (gn, gk, size, 1), as a value (no f).
func newLUTWeight(idx, lut *value) (*value, error) {
	if len(idx.shape) != 2 || idx.q == nil || idx.qSigned || len(lut.shape) != 4 || lut.shape[3] != 1 {
		return nil, fmt.Errorf("indices %v and lut %v are not a linear's weight", idx.shape, lut.shape)
	}
	n, k, gn, gk, size := idx.shape[0], idx.shape[1], lut.shape[0], lut.shape[1], lut.shape[2]
	if gn == 0 || gk == 0 || n%gn != 0 || k%gk != 0 || len(idx.q) != n*k || len(lut.f) != gn*gk*size {
		return nil, fmt.Errorf("lut grid %v does not divide %v", lut.shape[:2], idx.shape)
	}
	for _, q := range idx.q {
		if int(q) >= size {
			return nil, fmt.Errorf("index %d of a table of %d", q, size)
		}
	}
	w := &lutWeight{n: n, k: k, size: size, bn: n / gn, bk: k / gk, gk: gk, idx: idx.q, table: lut.f}
	return &value{shape: idx.shape, lut: w}, nil
}

// Pack writes a block of the weight's panels, as sgemm.Transposed packs a float32 one.
func (w *lutWeight) Pack(dst []float32, k0, kc, j0, nc int) {
	for p := 0; p*sgemm.NR < nc; p++ {
		d := dst[p*kc*sgemm.NR:][:kc*sgemm.NR]
		c0 := j0 + p*sgemm.NR
		cols := min(sgemm.NR, w.n-c0)
		for c := range cols {
			r := c0 + c
			row := w.idx[r*w.k+k0:][:kc]
			tables := w.table[(r/w.bn)*w.gk*w.size:]
			if w.gk == 1 { // a table for the whole row
				t := tables[:w.size]
				for kk, q := range row {
					d[kk*sgemm.NR+c] = t[q]
				}
				continue
			}
			for kk, q := range row {
				d[kk*sgemm.NR+c] = tables[((k0+kk)/w.bk)*w.size+int(q)]
			}
		}
		if cols < sgemm.NR {
			for kk := range kc {
				clear(d[kk*sgemm.NR+cols : kk*sgemm.NR+sgemm.NR])
			}
		}
	}
}

// lutRowsMax is the most rows of x a linear with a lutWeight multiplies by looking the weights
// up as it goes (lutWeight.mulAdd); sgemm packs them for more, whose packing is then shared.
const lutRowsMax = 8

// mulAdd adds x (m, k) times the weight's transpose to out (m, n), with the sums sgemm makes:
// for each output a sum over a KC block in order of k, added to out block after block. The
// weights are looked up 16 rows at a time (vecops.LUTRows16), the rows split over Threads
// goroutines: the products are of one token, which sgemm would run on one (sgemm.MinWork),
// spending most of it packing the weights.
func (w *lutWeight) mulAdd(m int, x, out []float32) {
	blocks := (w.n + 15) / 16
	threads := Threads
	if threads <= 0 {
		threads = runtime.GOMAXPROCS(0)
	}
	// at least 64 Ki weights a goroutine
	per := max((blocks+threads-1)/threads, (64<<10)/(16*w.k), 1)
	if per >= blocks {
		w.mulAddRows(m, x, out, 0, blocks)
		return
	}
	var wg sync.WaitGroup
	for b := 0; b < blocks; b += per {
		wg.Add(1)
		go func(b0, b1 int) {
			defer wg.Done()
			w.mulAddRows(m, x, out, b0, b1)
		}(b, min(blocks, b+per))
	}
	wg.Wait()
}

// mulAddRows is mulAdd for the blocks of 16 rows of the weight b0 to b1.
func (w *lutWeight) mulAddRows(m int, x, out []float32, b0, b1 int) {
	var rows, off [16]int32
	var acc [16]float32
	for b := b0; b < b1; b++ {
		r0 := b * 16
		lanes := min(16, w.n-r0)
		for l := range rows {
			rows[l] = int32(min(l, lanes-1) * w.k) // past the last row: the last row again
		}
		idx := w.idx[r0*w.k:]
		for i := range m {
			xi := x[i*w.k : (i+1)*w.k]
			oi := out[i*w.n+r0 : i*w.n+r0+lanes]
			for pc := 0; pc < w.k; pc += sgemm.KC {
				end := min(pc+sgemm.KC, w.k)
				acc = [16]float32{}
				for s := pc; s < end; {
					kb := s / w.bk
					e := min(end, (kb+1)*w.bk)
					for l := range off {
						r := r0 + min(l, lanes-1)
						off[l] = int32(((r/w.bn)*w.gk + kb) * w.size)
					}
					vecops.LUTRows16(&acc, xi[s:e], idx, &rows, s, &off, w.table)
					s = e
				}
				for l := range oi {
					oi[l] += acc[l]
				}
			}
		}
	}
}

// matmul is the batched product of the last two axes, broadcasting the batch axes.
func matmul(x, y *value, tx, ty bool) (*value, error) {
	if len(x.shape) < 2 || len(y.shape) < 2 {
		return nil, fmt.Errorf("matmul needs rank 2 or more, got %v and %v", x.shape, y.shape)
	}
	xr, yr := len(x.shape), len(y.shape)
	m, k := x.shape[xr-2], x.shape[xr-1]
	if tx {
		m, k = k, m
	}
	k2, n := y.shape[yr-2], y.shape[yr-1]
	if ty {
		k2, n = n, k2
	}
	if k != k2 {
		return nil, fmt.Errorf("inner sizes of %v and %v differ", x.shape, y.shape)
	}
	// batch indices through broadcast on the leading axes
	xb := &value{shape: x.shape[:xr-2], f: make([]float32, numel(x.shape[:xr-2]))}
	yb := &value{shape: y.shape[:yr-2], f: make([]float32, numel(y.shape[:yr-2]))}
	for i := range xb.f {
		xb.f[i] = float32(i)
	}
	for i := range yb.f {
		yb.f[i] = float32(i)
	}
	xi, err := broadcast(xb, yb, func(a, _ float32) float32 { return a })
	if err != nil {
		return nil, err
	}
	yi, _ := broadcast(xb, yb, func(_, b float32) float32 { return b })
	batch := len(xi.f)
	shape := append(append([]int{}, xi.shape...), m, n)
	out := &value{shape: shape, f: make([]float32, batch*m*n)}
	for bi := 0; bi < batch; bi++ {
		xb, yb := int(xi.f[bi]), int(yi.f[bi])
		a := sgemm.A{Data: x.f[xb*m*k : (xb+1)*m*k], Stride: k}
		if tx { // x is (k, m)
			a = sgemm.A{Data: a.Data, Stride: m, Trans: true}
		}
		yd := y.f[yb*k*n : (yb+1)*k*n]
		var b sgemm.Source = sgemm.Matrix{Data: yd, Stride: n, N: n}
		if ty { // y is (n, k)
			b = sgemm.Transposed{Data: yd, Stride: k, N: n}
		}
		sgemm.MulAdd(m, n, k, a, b, out.f[bi*m*n:(bi+1)*m*n], n, Threads)
	}
	return out, nil
}

// transpose2 transposes a (rows, cols) matrix.
func transpose2(a []float32, rows, cols int) []float32 {
	t := make([]float32, len(a))
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			t[j*rows+i] = a[i*cols+j]
		}
	}
	return t
}

func reshape(x *value, s []int32) (*value, error) {
	shape := make([]int, len(s))
	infer, known := -1, 1
	for i, d := range s {
		switch {
		case d == -1 && infer < 0:
			infer = i
		case d == 0:
			shape[i] = x.shape[i]
			known *= shape[i]
		case d > 0:
			shape[i] = int(d)
			known *= shape[i]
		default:
			return nil, fmt.Errorf("bad shape %v", s)
		}
	}
	if infer >= 0 {
		shape[infer] = len(x.f) / known
	}
	if numel(shape) != len(x.f) {
		return nil, fmt.Errorf("cannot reshape %v to %v", x.shape, s)
	}
	return &value{shape: shape, f: x.f}, nil
}

// permute copies x into a new layout: walking the output in order, the x offset starts at start
// and moves by src[d] along output axis d.
func permute(pl *pool, x *value, start int, shape []int, src []int) *value {
	out := &value{shape: shape, f: pl.get(numel(shape))}
	tensorops.Permute(out.f, x.f, start, shape, src)
	return out
}

func transpose(pl *pool, x *value, perm []int32) (*value, error) {
	if len(perm) != len(x.shape) {
		return nil, fmt.Errorf("perm %v for shape %v", perm, x.shape)
	}
	shape, src := tensorops.TransposePlan(x.shape, perm)
	return permute(pl, x, 0, shape, src), nil
}

func tile(x *value, reps []int32) (*value, error) {
	if len(reps) != len(x.shape) {
		return nil, fmt.Errorf("reps %v for shape %v", reps, x.shape)
	}
	shape := make([]int, len(reps))
	for i, r := range reps {
		shape[i] = x.shape[i] * int(r)
	}
	out := &value{shape: shape, f: make([]float32, numel(shape))}
	tensorops.Tile(out.f, x.f, x.shape, shape)
	return out, nil
}

// pixelShuffle is PyTorch's PixelShuffle of x (N, C f f, H, W): (N, C, H f, W f), output
// channel c at (y f + i, x f + j) from input channel c f f + i f + j at (y, x).
func pixelShuffle(x *value, f int) (*value, error) {
	if len(x.shape) != 4 || f < 1 || x.shape[1]%(f*f) != 0 {
		return nil, fmt.Errorf("pixel_shuffle of shape %v by %d", x.shape, f)
	}
	N, CF, H, W := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	C := CF / (f * f)
	out := &value{shape: []int{N, C, H * f, W * f}, f: make([]float32, len(x.f))}
	for n := 0; n < N; n++ {
		for c := 0; c < C; c++ {
			for i := 0; i < f; i++ {
				for j := 0; j < f; j++ {
					src := x.f[((n*CF+c*f*f+i*f+j)*H)*W:][:H*W]
					for y := 0; y < H; y++ {
						dst := out.f[((n*C+c)*H*f+y*f+i)*W*f:]
						for xx := 0; xx < W; xx++ {
							dst[xx*f+j] = src[y*W+xx]
						}
					}
				}
			}
		}
	}
	return out, nil
}

// upsampleNearest repeats each pixel of x (..., H, W) as an sh x sw block.
func upsampleNearest(pl *pool, x *value, sh, sw int) (*value, error) {
	n := len(x.shape)
	if n < 2 || sh < 1 || sw < 1 {
		return nil, fmt.Errorf("upsample of shape %v by %d x %d", x.shape, sh, sw)
	}
	H, W := x.shape[n-2], x.shape[n-1]
	shape := append([]int(nil), x.shape...)
	shape[n-2], shape[n-1] = H*sh, W*sw
	out := &value{shape: shape, f: pl.get(numel(shape))}
	planes := numel(x.shape) / (H * W)
	for p := 0; p < planes; p++ {
		src, dst := x.f[p*H*W:][:H*W], out.f[p*H*sh*W*sw:]
		if sh == 2 && sw == 2 {
			// each row's columns twice, then the row twice
			for y := range H {
				s, d := src[y*W:][:W], dst[2*y*2*W:][:4*W]
				for i, v := range s {
					d[2*i], d[2*i+1] = v, v
				}
				copy(d[2*W:], d[:2*W])
			}
			continue
		}
		for y := 0; y < H*sh; y++ {
			row := dst[y*W*sw:][:W*sw]
			for xx := range row {
				row[xx] = src[(y/sh)*W+xx/sw]
			}
		}
	}
	return out, nil
}

func sliceByIndex(x *value, begin, end, stride []int32, bm, em, sm []bool) (*value, error) {
	rank := len(x.shape)
	if len(begin) != rank || len(end) != rank || len(stride) != rank {
		return nil, fmt.Errorf("begin %v end %v stride %v for shape %v", begin, end, stride, x.shape)
	}
	st := strides(x.shape)
	var shape, outShape []int
	src := []int{}
	base := 0
	for i, d := range x.shape {
		b, e, s := int(begin[i]), int(end[i]), int(stride[i])
		if s == 0 {
			return nil, fmt.Errorf("zero stride")
		}
		if b < 0 {
			b += d
		}
		if e < 0 {
			e += d
		}
		if bm[i] {
			b = 0
			if s < 0 {
				b = d - 1
			}
		}
		if em[i] {
			e = d
			if s < 0 {
				e = -1
			}
		}
		if sm[i] {
			e = b + 1
		}
		n := 0
		if s > 0 && e > b {
			n = (e - b + s - 1) / s
		} else if s < 0 && b > e {
			n = (b - e - s - 1) / -s
		}
		base += b * st[i]
		shape = append(shape, n)
		src = append(src, s*st[i])
		if !sm[i] {
			outShape = append(outShape, n)
		}
	}
	out := permute(nil, x, base, shape, src)
	out.shape = outShape
	return out, nil
}

// padConst pads the last len(pad)/2 axes with c: pad holds (before, after) pairs in axis order.
func padConst(x *value, pad []int32, c float32) (*value, error) {
	return padValue(x, pad, "constant", c)
}

// padEdges pads the last len(pad)/2 axes by reflection (the edge not repeated) or by repeating
// the edge.
func padEdges(x *value, pad []int32, mode string) (*value, error) {
	return padValue(x, pad, mode, 0)
}

func padValue(x *value, pad []int32, mode string, c float32) (*value, error) {
	rank := len(x.shape)
	k := len(pad) / 2
	if len(pad)%2 != 0 || k > rank {
		return nil, fmt.Errorf("pad %v for shape %v", pad, x.shape)
	}
	pads := make([]int, 2*rank)
	for j := range 2 * k {
		pads[2*(rank-k)+j] = int(pad[j])
	}
	shape := tensorops.PadShape(x.shape, pads)
	out := &value{shape: shape, f: make([]float32, numel(shape))}
	tensorops.Pad(out.f, x.f, x.shape, pads, mode, c)
	return out, nil
}

func concat(pl *pool, xs []*value, a int) (*value, error) {
	if len(xs) == 0 {
		return nil, fmt.Errorf("nothing to concat")
	}
	ax, err := axis(a, len(xs[0].shape))
	if err != nil {
		return nil, err
	}
	shape := append([]int{}, xs[0].shape...)
	shape[ax] = 0
	srcs := make([][]float32, len(xs))
	sizes := make([]int, len(xs))
	for i, x := range xs {
		if len(x.shape) != len(shape) || x.f == nil {
			return nil, fmt.Errorf("cannot concat %v", x.shape)
		}
		for d := range shape {
			if d != ax && x.shape[d] != shape[d] {
				return nil, fmt.Errorf("cannot concat %v and %v", xs[0].shape, x.shape)
			}
		}
		shape[ax] += x.shape[ax]
		srcs[i] = x.f
		sizes[i] = x.shape[ax]
	}
	outer, _, inner := around(shape, ax)
	dst := pl.get(numel(shape))
	return &value{shape: shape, f: tensorops.Concat(dst, srcs, sizes, outer, inner)}, nil
}

// conv1d is a grouped 1 D convolution of x (n, c_in, w) with weight (c_out, c_in/groups, k):
// conv2d's over an image of one row (its products on sgemm, its depthwise convolutions in
// parallel). A direct loop of one thread took 1.5 s of Supertonic 3's text encoder (128
// characters), 91 % of it here.
func conv1d(pl *pool, fixed bool, x, w, bias *value, stride, dil []int32, padType string, pad []int32, groups int) (*value, error) {
	if len(x.shape) != 3 || len(w.shape) != 3 {
		return nil, fmt.Errorf("only 1 D convolutions are supported, got x %v weight %v", x.shape, w.shape)
	}
	var p []int32
	switch padType {
	case "valid":
	case "custom":
		if len(pad) < 2 {
			return nil, fmt.Errorf("a 1 D convolution's custom pad %v is not of 2 values", pad)
		}
		p = []int32{0, 0, pad[0], pad[1]}
	default:
		return nil, fmt.Errorf("pad_type %q is not supported", padType)
	}
	x2 := &value{shape: []int{x.shape[0], x.shape[1], 1, x.shape[2]}, f: x.f}
	w2 := &value{shape: []int{w.shape[0], w.shape[1], 1, w.shape[2]}, f: w.f}
	y, err := conv2d(pl, x2, w2, bias, []int32{1, stride[0]}, []int32{1, dil[0]}, padType, p, groups, false, fixed)
	if err != nil {
		return nil, err
	}
	y.shape = []int{y.shape[0], y.shape[1], y.shape[3]}
	return y, nil
}

// conv2d is a grouped 2 D convolution of x (n, c_in, h, w) with weight (c_out, c_in/groups, kh,
// kw), pad (top, bottom, left, right) for pad_type custom. The output channels run in parallel.
func conv2d(bp *pool, x, w, bias *value, stride, dil []int32, padType string, pad []int32, groups int, relu, fixed bool) (*value, error) {
	if len(w.shape) != 4 {
		return nil, fmt.Errorf("a 2 D convolution needs a weight of rank 4, got %v", w.shape)
	}
	n, cin, ih, iw := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	cout, cg, kh, kw := w.shape[0], w.shape[1], w.shape[2], w.shape[3]
	if groups <= 0 || cin != cg*groups || cout%groups != 0 {
		return nil, fmt.Errorf("x %v, weight %v and groups %d do not match", x.shape, w.shape, groups)
	}
	at := func(v []int32, i int) int {
		if len(v) == 1 {
			return int(v[0])
		}
		return int(v[i])
	}
	sh, sw, dh, dw := at(stride, 0), at(stride, 1), at(dil, 0), at(dil, 1)
	var pt, pb, pl, pr int
	switch padType {
	case "valid":
	case "custom":
		if len(pad) != 4 {
			return nil, fmt.Errorf("%d pads for a 2 D convolution", len(pad))
		}
		pt, pb, pl, pr = int(pad[0]), int(pad[1]), int(pad[2]), int(pad[3])
	default:
		return nil, fmt.Errorf("pad_type %q is not supported", padType)
	}
	oh := (ih+pt+pb-dh*(kh-1)-1)/sh + 1
	ow := (iw+pl+pr-dw*(kw-1)-1)/sw + 1
	out := &value{shape: []int{n, cout, oh, ow}, f: bp.get(n * cout * oh * ow)}
	perGroup := cout / groups
	if cg == 1 && perGroup == 1 && kh == 3 && kw == 3 && sh == 1 && sw == 1 && dh == 1 && dw == 1 {
		depthwise3(bp, x, w, bias, out, pt, pl, relu)
		return out, nil
	}
	if kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0 && oh == ih && ow == iw && cg <= pointwiseK {
		pointwise(bp, x, w, bias, out, groups, relu, bp.packed(w, bias, groups, fixed))
		return out, nil
	}
	if K := cg * kh * kw; kh*kw > 1 && K <= directK {
		directConv(bp, x, w, bias, out, groups, sh, sw, dh, dw, pt, pl, relu, bp.packed(w, bias, groups, fixed))
		return out, nil
	}
	if relu {
		defer f32.ReLU(out.f, out.f)
	}
	if cg == 1 && perGroup == 1 {
		// depthwise: a product of one row a channel, the direct loop
		parallel(cout, func(lo, hi int) {
			for oc := lo; oc < hi; oc++ {
				conv2dChannel(x, w, bias, out, oc, oc/perGroup, n, cin, cg, ih, iw, kh, kw, oh, ow, sh, sw, dh, dw, pt, pl)
			}
		})
		return out, nil
	}
	// each group's output (perGroup x oh ow) = weight (perGroup x cg kh kw) times the im2col of
	// its input channels, packed straight from x
	K, N := cg*kh*kw, oh*ow
	for b := 0; b < n; b++ {
		for g := 0; g < groups; g++ {
			o := out.f[(b*cout+g*perGroup)*N:][:perGroup*N]
			if bias != nil {
				for oc := range perGroup {
					v := bias.f[g*perGroup+oc]
					row := o[oc*N : (oc+1)*N]
					for i := range row {
						row[i] = v
					}
				}
			} else {
				clear(o)
			}
			xg := x.f[(b*cin+g*cg)*ih*iw:][:cg*ih*iw]
			var src sgemm.Source
			if kh == 1 && kw == 1 && sh == 1 && sw == 1 && pt == 0 && pl == 0 && oh == ih && ow == iw {
				src = sgemm.Matrix{Data: xg, Stride: N, N: N}
			} else {
				src = &im2col{x: xg, ih: ih, iw: iw, kh: kh, kw: kw, oh: oh, ow: ow, sh: sh, sw: sw, dh: dh, dw: dw, pt: pt, pl: pl}
			}
			wg := sgemm.A{Data: w.f[g*perGroup*K:][:perGroup*K], Stride: K}
			sgemm.MulAdd(perGroup, N, K, wg, src, o, N, Threads)
		}
	}
	return out, nil
}

// Threads is how many goroutines a product (a convolution's, a linear's, a matmul's) takes: 0
// for GOMAXPROCS.
var Threads int

// im2col is a convolution's im2col matrix (cin kh kw x oh ow) of one image's channels x, read
// by sgemm as it packs: row (c, ky, kx), column (y, x) is x[c][y sh + ky dh - pt][x sw + kx dw
// - pl], zero outside.
type im2col struct {
	x                                              []float32
	ih, iw, kh, kw, oh, ow, sh, sw, dh, dw, pt, pl int
}

func (m *im2col) Pack(dst []float32, k0, kc, j0, nc int) {
	n := m.oh * m.ow
	for p := 0; p*sgemm.NR < nc; p++ {
		d := dst[p*kc*sgemm.NR:][:kc*sgemm.NR]
		c0 := j0 + p*sgemm.NR
		w := min(sgemm.NR, n-c0)
		for kk := range kc {
			k := k0 + kk
			c, ky, kx := k/(m.kh*m.kw), k/m.kw%m.kh, k%m.kw
			plane := m.x[c*m.ih*m.iw:][:m.ih*m.iw]
			o := d[kk*sgemm.NR : kk*sgemm.NR+sgemm.NR]
			// the panel's columns in runs of one output row
			for j := 0; j < w; {
				col := c0 + j
				y, xx := col/m.ow, col%m.ow
				run := min(w-j, m.ow-xx)
				seg := o[j : j+run]
				sy := y*m.sh + ky*m.dh - m.pt
				if sy < 0 || sy >= m.ih {
					clear(seg)
					j += run
					continue
				}
				row := plane[sy*m.iw : (sy+1)*m.iw]
				sx := xx*m.sw + kx*m.dw - m.pl
				if m.sw == 1 && sx >= 0 && sx+run <= m.iw {
					copy(seg, row[sx:sx+run])
				} else {
					for i := range seg {
						if s := sx + i*m.sw; s >= 0 && s < m.iw {
							seg[i] = row[s]
						} else {
							seg[i] = 0
						}
					}
				}
				j += run
			}
			clear(o[w:])
		}
	}
}

// conv2dChannel computes output channel oc of conv2d, of group g.
func conv2dChannel(x, w, bias, out *value, oc, g, n, cin, cg, ih, iw, kh, kw, oh, ow, sh, sw, dh, dw, pt, pl int) {
	cout := out.shape[1]
	{
		for b := 0; b < n; b++ {
			o := out.f[(b*cout+oc)*oh*ow:][:oh*ow]
			if bias != nil {
				for i := range o {
					o[i] = bias.f[oc]
				}
			} else {
				clear(o)
			}
			for ic := 0; ic < cg; ic++ {
				xp := x.f[(b*cin+g*cg+ic)*ih*iw:][:ih*iw]
				for ky := 0; ky < kh; ky++ {
					for kx := 0; kx < kw; kx++ {
						wv := w.f[((oc*cg+ic)*kh+ky)*kw+kx]
						if wv == 0 {
							continue
						}
						for y := 0; y < oh; y++ {
							sy := y*sh + ky*dh - pt
							if sy < 0 || sy >= ih {
								continue
							}
							row, orow := xp[sy*iw:][:iw], o[y*ow:][:ow]
							if sw == 1 {
								// the columns whose source is inside the row, one axpy
								off := kx*dw - pl
								x0, x1 := max(0, -off), min(ow, iw-off)
								if x0 < x1 {
									f32.Axpy(orow[x0:x1], wv, row[x0+off:x1+off])
								}
								continue
							}
							for xx := 0; xx < ow; xx++ {
								if sx := xx*sw + kx*dw - pl; sx >= 0 && sx < iw {
									orow[xx] += wv * row[sx]
								}
							}
						}
					}
				}
			}
		}
	}
}

// depthwise3 is a depthwise 3 x 3 convolution of stride 1 into out (n, c, oh, ow): each
// channel's plane copied into one padded by pt rows and pl columns (oh+2 x ow+2, zeros around),
// then each output row from three of its rows (f32.Depthwise3: AVX2), with the ReLU after it
// when relu. Before, nine passes of an axpy a row over the output, about 3 G multiply-adds a
// second.
func depthwise3(bp *pool, x, w, bias, out *value, pt, pl int, relu bool) {
	n, c, ih, iw := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	oh, ow := out.shape[2], out.shape[3]
	P := ow + 2
	parallel(n*c, func(lo, hi int) {
		pad := bp.work((oh + 2) * P)
		defer bp.done(pad)
		pp := *pad
		clear(pp)
		for plane := lo; plane < hi; plane++ {
			ch := plane % c
			src := x.f[plane*ih*iw:][:ih*iw]
			for y := range ih {
				if r := y + pt; r >= 0 && r < oh+2 {
					row := pp[r*P:][:P]
					a, b := max(0, -pl), min(iw, P-pl)
					if a < b {
						copy(row[pl+a:pl+b], src[y*iw+a:y*iw+b])
					}
				}
			}
			var w9 [9]float32
			copy(w9[:], w.f[ch*9:ch*9+9])
			var b float32
			if bias != nil {
				b = bias.f[ch]
			}
			o := out.f[plane*oh*ow:][:oh*ow]
			for y := range oh {
				f32.Depthwise3(o[y*ow:(y+1)*ow], pp[y*P:], pp[(y+1)*P:], pp[(y+2)*P:], &w9, b, relu)
			}
		}
	})
}

// directK is the most taps (input channels of a group times the kernel's) a convolution is
// computed directly with (directConv) rather than as a product on sgemm: YuNet's first, 3
// channels by 3 x 3 of stride 2, took 1.2 ms of the 4.6 its run took on im2col and sgemm.
const directK = 64

// directConv is conv2d into out (n, cout, oh, ow) as weighted sums of input rows: each input
// channel's plane padded and cut into sw phases (phase q of a padded row: its columns sw i + q),
// so that a tap (ky, kx) of output row y is a run of one phase's row y sh + ky dh from column
// kx dw / sw, and an output row the sum of the cg kh kw runs weighted by the taps
// (f32.WeightedRows: AVX2), with the ReLU after it when relu.
func directConv(bp *pool, x, w, bias, out *value, groups, sh, sw, dh, dw, pt, pl int, relu bool, pk *packedConv) {
	n, cin, ih, iw := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	cout, cg, kh, kw := w.shape[0], w.shape[1], w.shape[2], w.shape[3]
	oh, ow := out.shape[2], out.shape[3]
	perGroup, K := cout/groups, cg*kh*kw
	pw := ow + (kw-1)*dw/sw + 1     // a phase's row: every column a tap reads
	pr := (oh-1)*sh + (kh-1)*dh + 1 // the padded rows the taps read
	plane := sw * pr * pw           // a channel's phases
	buf := bp.work(cin * plane)
	defer bp.done(buf)
	ph := *buf
	for b := range n {
		parallel(cin, func(lo, hi int) {
			buf := bp.work(sw * pw)
			defer bp.done(buf)
			t := *buf // a padded row: t[j] = the input's column j - pl
			for c := lo; c < hi; c++ {
				src := x.f[(b*cin+c)*ih*iw:][:ih*iw]
				for r := range pr {
					phase := func(q int) []float32 { return ph[c*plane+(q*pr+r)*pw:][:pw] }
					y := r - pt
					if y < 0 || y >= ih {
						for q := range sw {
							clear(phase(q))
						}
						continue
					}
					s := src[y*iw:][:iw]
					a, e := min(pl, len(t)), min(len(t), iw+pl)
					clear(t[:a])
					if a < e {
						copy(t[a:e], s[a-pl:e-pl])
					}
					clear(t[max(a, e):])
					switch sw {
					case 1:
						copy(phase(0), t)
					case 2:
						f32.Deinterleave2(phase(0), phase(1), t)
					default:
						for q := range sw {
							row := phase(q)
							for i := range row {
								row[i] = t[sw*i+q]
							}
						}
					}
				}
			}
		})
		parallel(len(pk.blocks), func(lo, hi int) {
			rows := make([]*float32, K)
			for bi := lo; bi < hi; bi++ {
				ob, wb, bb := pk.blocks[bi], pk.w[bi], pk.bias[bi]
				g := ob.oc / perGroup
				o := out.f[(b*cout+ob.oc)*oh*ow:][:ob.n*oh*ow]
				for y := range oh {
					j := 0
					for ic := range cg {
						base := (g*cg + ic) * plane
						for ky := range kh {
							r := y*sh + ky*dh
							for kx := range kw {
								q, off := kx*dw%sw, kx*dw/sw
								rows[j] = &ph[base+(q*pr+r)*pw+off]
								j++
							}
						}
					}
					rowsInto(o, oh*ow, y*ow, ow, ob.n, rows, wb, &bb, relu)
				}
			}
		})
	}
}

// packedConv is a convolution's weights as directConv and pointwise take them: its channels in
// blocks, each block's weights interleaved and its bias.
type packedConv struct {
	blocks []block
	w      [][]float32
	bias   [][8]float32
}

// packed is w and bias packed (packedConv), kept for the next runs when fixed (both are
// constants of the program, whose buffers stay): repacking YuNet's every run made 180 KB.
func (pl *pool) packed(w, bias *value, groups int, fixed bool) *packedConv {
	key := unsafe.SliceData(w.f)
	if fixed && pl != nil {
		if pk, ok := pl.convs[key]; ok {
			return pk
		}
	}
	cout := w.shape[0]
	K := len(w.f) / cout
	pk := &packedConv{blocks: blocksOf(cout, cout/groups)}
	for _, ob := range pk.blocks {
		pk.w = append(pk.w, interleave(w.f[ob.oc*K:][:ob.n*K], ob.n, K))
		pk.bias = append(pk.bias, biasOf(bias, ob.oc, ob.n))
	}
	if fixed && pl != nil {
		if pl.convs == nil {
			pl.convs = map[*float32]*packedConv{}
		}
		pl.convs[key] = pk
	}
	return pk
}

// block is the output channels of one group from oc that a pass computes from the same input
// rows: n of 8 (f32.WeightedRows8), 4 (f32.WeightedRows4) or 1 (f32.WeightedRows).
type block struct{ oc, n int }

// blocksOf cuts each group of perGroup of cout channels into blocks of 8, then one of 4, then
// single channels.
func blocksOf(cout, perGroup int) []block {
	var bs []block
	for g0 := 0; g0 < cout; g0 += perGroup {
		oc, end := g0, g0+perGroup
		for ; end-oc >= 8; oc += 8 {
			bs = append(bs, block{oc, 8})
		}
		if end-oc >= 4 {
			bs = append(bs, block{oc, 4})
			oc += 4
		}
		for ; oc < end; oc++ {
			bs = append(bs, block{oc, 1})
		}
	}
	return bs
}

// interleave is the weights of n output channels, K each, as the block's WeightedRows takes
// them: tap j's n together (wb[n j+o]).
func interleave(w []float32, n, K int) []float32 {
	wb := make([]float32, n*K)
	for o := range n {
		for j := range K {
			wb[n*j+o] = w[o*K+j]
		}
	}
	return wb
}

// biasOf is the bias of n channels from oc (zeros past them, and without a bias).
func biasOf(bias *value, oc, n int) [8]float32 {
	var b [8]float32
	if bias != nil {
		copy(b[:n], bias.f[oc:oc+n])
	}
	return b
}

// rowsInto writes columns [at, at+m) of a block's n output planes (each of size plane, one after
// the other in o) from rows.
func rowsInto(o []float32, plane, at, m, n int, rows []*float32, wb []float32, b *[8]float32, relu bool) {
	switch n {
	case 8:
		var dst [8]*float32
		for c := range dst {
			dst[c] = &o[c*plane+at]
		}
		f32.WeightedRows8(&dst, m, rows, wb, b, relu)
	case 4:
		dst := [4]*float32{&o[at], &o[plane+at], &o[2*plane+at], &o[3*plane+at]}
		f32.WeightedRows4(&dst, m, rows, wb, (*[4]float32)(b[:4]), relu)
	default:
		f32.WeightedRows(o[at:][:m], rows, wb, b[0], relu)
	}
}

// pointwiseK is the most input channels a 1 x 1 convolution of stride 1 is computed with as
// weighted sums of its input planes (pointwise) rather than as a product on sgemm: YuNet's
// 32 to 64 channels at 64 x 80 took 0.41 ms on sgemm (packing what is read once), 0.33 here;
// beyond a few hundred channels sgemm's blocking over them wins.
const pointwiseK = 256

// pointwise is a 1 x 1 convolution of stride 1 without pads into out: each output plane the
// weighted sum of its group's input planes (f32.WeightedRows8, eight output channels a pass),
// in blocks of columns whose input stays in the core's first cache.
func pointwise(bp *pool, x, w, bias, out *value, groups int, relu bool, pk *packedConv) {
	n, cin := x.shape[0], x.shape[1]
	cout, cg := w.shape[0], w.shape[1]
	N := x.shape[2] * x.shape[3]
	perGroup := cout / groups
	cb := max(16, (8192/cg)&^15) // cg rows of cb floats: 32 KB, the core's first cache's 48 less the outputs
	for b := range n {
		blocks := (N + cb - 1) / cb
		parallel(blocks, func(lo, hi int) {
			rows := make([]*float32, cg)
			// the block's input rows packed, cb+16 apart: planes of 4 KB multiples (YuNet's
			// 64 x 80) put one column of every row in the same set of the first cache, whose
			// 12 ways the rows then evicted each other from (slower than sgemm)
			st := cb + 16
			buf := bp.work(groups * cg * st)
			defer bp.done(buf)
			in := *buf
			for blk := lo; blk < hi; blk++ {
				c0 := blk * cb
				m := min(cb, N-c0)
				for ic := range groups * cg {
					copy(in[ic*st:][:m], x.f[(b*cin+ic)*N+c0:][:m])
				}
				for i, ob := range pk.blocks {
					g := ob.oc / perGroup
					for ic := range cg {
						rows[ic] = &in[(g*cg+ic)*st]
					}
					rowsInto(out.f[(b*cout+ob.oc)*N:], N, c0, m, ob.n, rows, pk.w[i], &pk.bias[i], relu)
				}
			}
		})
	}
}

// blockIndex maps each element of shape to its block in a block grid of shape grid.
func blockIndex(shape, grid []int) []int {
	gst := strides(grid)
	block := make([]int, len(shape))
	for i := range shape {
		block[i] = shape[i] / grid[i]
	}
	out := make([]int, numel(shape))
	idx := make([]int, len(shape))
	for i := range out {
		o := 0
		for d := range shape {
			o += idx[d] / block[d] * gst[d]
		}
		out[i] = o
		for d := len(shape) - 1; d >= 0; d-- {
			if idx[d]++; idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}

// lutToDense expands palettized indices with one table per block: lut has shape grid + [size, 1].
func lutToDense(idx, lut *value) (*value, error) {
	rank := len(idx.shape)
	if idx.i == nil && idx.q == nil || len(lut.shape) != rank+2 || lut.shape[rank+1] != 1 {
		return nil, fmt.Errorf("indices %v and lut %v do not match", idx.shape, lut.shape)
	}
	grid, size := lut.shape[:rank], lut.shape[rank]
	for d := range grid {
		if grid[d] == 0 || idx.shape[d]%grid[d] != 0 {
			return nil, fmt.Errorf("lut grid %v does not divide %v", grid, idx.shape)
		}
	}
	blocks := blockIndex(idx.shape, grid)
	out := &value{shape: idx.shape, f: make([]float32, idx.qLen())}
	for i := range out.f {
		out.f[i] = lut.f[blocks[i]*size+int(idx.qAt(i))]
	}
	return out, nil
}

// shiftScale is (data - offset) * scale with one scale (and offset) per block.
func shiftScale(data, scale, offset *value) (*value, error) {
	rank := len(data.shape)
	if data.i == nil && data.q == nil || len(scale.shape) != rank {
		return nil, fmt.Errorf("data %v and scale %v do not match", data.shape, scale.shape)
	}
	blocks := blockIndex(data.shape, scale.shape)
	out := &value{shape: data.shape, f: make([]float32, data.qLen())}
	for i := range out.f {
		v := float32(data.qAt(i))
		if offset != nil {
			if offset.f != nil {
				v -= offset.f[blocks[i]]
			} else {
				v -= float32(offset.qAt(blocks[i]))
			}
		}
		out.f[i] = v * scale.f[blocks[i]]
	}
	return out, nil
}

// maxPool2d is a 2 D max pool of x (n, c, h, w) with kernel k, strides st and pads (top, bottom,
// left, right), the padding never chosen.
func maxPool2d(bp *pool, x *value, k, st, pad []int32) (*value, error) {
	if len(x.shape) != 4 || len(k) != 2 || len(st) != 2 || len(pad) != 4 {
		return nil, fmt.Errorf("max_pool of %v with kernel %v, strides %v, pad %v", x.shape, k, st, pad)
	}
	n, c, h, w := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	kh, kw, sh, sw := int(k[0]), int(k[1]), int(st[0]), int(st[1])
	pt, pb, pl, pr := int(pad[0]), int(pad[1]), int(pad[2]), int(pad[3])
	oh, ow := (h+pt+pb-kh)/sh+1, (w+pl+pr-kw)/sw+1
	out := &value{shape: []int{n, c, oh, ow}, f: bp.get(n * c * oh * ow)}
	if kh == 2 && kw == 2 && sh == 2 && sw == 2 && pt == 0 && pl == 0 && 2*oh <= h && 2*ow <= w {
		// YuNet's: two rows to one, f32.MaxPool2 (AVX2)
		parallel(n*c, func(lo, hi int) {
			for p := lo; p < hi; p++ {
				src, dst := x.f[p*h*w:(p+1)*h*w], out.f[p*oh*ow:(p+1)*oh*ow]
				for i := range oh {
					f32.MaxPool2(dst[i*ow:(i+1)*ow], src[2*i*w:], src[(2*i+1)*w:])
				}
			}
		})
		return out, nil
	}
	parallel(n*c, func(lo, hi int) {
		for p := lo; p < hi; p++ {
			src, dst := x.f[p*h*w:(p+1)*h*w], out.f[p*oh*ow:(p+1)*oh*ow]
			for i := range oh {
				for j := range ow {
					m := float32(math.Inf(-1))
					for a := range kh {
						y := i*sh - pt + a
						if y < 0 || y >= h {
							continue
						}
						for b := range kw {
							xx := j*sw - pl + b
							if xx >= 0 && xx < w && src[y*w+xx] > m {
								m = src[y*w+xx]
							}
						}
					}
					dst[i*ow+j] = m
				}
			}
		}
	})
	return out, nil
}

// convTranspose2d is the transpose of a 2 D convolution (its gradient with respect to its input):
// x (n, cin, h, w), weight (cin, cout, kh, kw); the output (n, cout, (h-1)*sh+kh, (w-1)*sw+kw)
// less pad (top, bottom, left, right), or output_shape's height and width taken from the top
// left.
func convTranspose2d(x, w, bias *value, st, pad, outShape []int32) (*value, error) {
	if len(x.shape) != 4 || len(w.shape) != 4 || w.shape[0] != x.shape[1] || len(st) != 2 || len(pad) != 4 {
		return nil, fmt.Errorf("conv_transpose of %v by %v", x.shape, w.shape)
	}
	n, cin, h, wd := x.shape[0], x.shape[1], x.shape[2], x.shape[3]
	cout, kh, kw := w.shape[1], w.shape[2], w.shape[3]
	sh, sw := int(st[0]), int(st[1])
	rh, rw := (h-1)*sh+kh, (wd-1)*sw+kw
	pt, pl := int(pad[0]), int(pad[2])
	oh, ow := rh-int(pad[0])-int(pad[1]), rw-int(pad[2])-int(pad[3])
	if len(outShape) == 4 {
		oh, ow = int(outShape[2]), int(outShape[3])
	}
	out := &value{shape: []int{n, cout, oh, ow}, f: make([]float32, n*cout*oh*ow)}
	parallel(cout, func(lo, hi int) {
		for o := lo; o < hi; o++ {
			for b := range n {
				dst := out.f[(b*cout+o)*oh*ow:][:oh*ow]
				if bias != nil {
					for i := range dst {
						dst[i] = bias.f[o]
					}
				}
				for c := range cin {
					src := x.f[(b*cin+c)*h*wd:][:h*wd]
					for a := range kh {
						for e := range kw {
							wv := w.f[((c*cout+o)*kh+a)*kw+e]
							if wv == 0 {
								continue
							}
							for i := range h {
								y := i*sh + a - pt
								if y < 0 || y >= oh {
									continue
								}
								for j := range wd {
									xx := j*sw + e - pl
									if xx >= 0 && xx < ow {
										dst[y*ow+xx] += wv * src[i*wd+j]
									}
								}
							}
						}
					}
				}
			}
		}
	})
	return out, nil
}
