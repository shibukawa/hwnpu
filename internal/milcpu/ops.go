package milcpu

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/shibukawa/hwnpu/internal/milspec"
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
	v, err := p.decode(b.GetValue())
	return v, err == nil, err
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
	if v.i == nil {
		r.err = fmt.Errorf("input %s is not an int tensor", name)
	}
	return v.i
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
		return p.decode(op.GetAttributes()["val"])
	case "identity":
		x := r.floats("x")
		if r.err == nil {
			out = &value{shape: x.shape, f: append([]float32(nil), x.f...)}
		}
	case "add", "sub", "mul", "real_div", "maximum", "minimum":
		x, y := r.floats("x"), r.floats("y")
		if r.err == nil {
			out, r.err = broadcast(x, y, binaryOp(typ))
		}
	case "silu", "sigmoid", "relu", "tanh", "exp", "sqrt":
		x := r.floats("x")
		if r.err == nil {
			out = unary(x, typ)
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
		x, w := r.floats("x"), r.floats("weight")
		b := r.opt("bias")
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
			out, r.err = transpose(x, perm)
		}
	case "tile":
		x := r.floats("x")
		reps := r.ints("reps")
		if r.err == nil {
			out, r.err = tile(x, reps)
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
			if mode != "constant" {
				return nil, fmt.Errorf("pad mode %q is not supported", mode)
			}
			out, r.err = padConst(x, pad, c)
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
			out, r.err = concat(xs, int(axis))
		}
	case "conv":
		x, w := r.floats("x"), r.floats("weight")
		b := r.opt("bias")
		strides, dil := r.ints("strides", 1), r.ints("dilations", 1)
		padType := r.str("pad_type", "valid")
		pad := r.ints("pad", 0, 0)
		groups := r.int("groups", 1)
		if r.err == nil {
			out, r.err = conv1d(x, w, b, strides, dil, padType, pad, int(groups))
		}
	case "constexpr_lut_to_dense":
		idx, lut := r.get("indices"), r.floats("lut")
		if r.err == nil {
			out, r.err = lutToDense(idx, lut)
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

func strides(shape []int) []int {
	s := make([]int, len(shape))
	n := 1
	for i := len(shape) - 1; i >= 0; i-- {
		s[i] = n
		n *= shape[i]
	}
	return s
}

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

func unary(x *value, typ string) *value {
	out := &value{shape: x.shape, f: make([]float32, len(x.f))}
	for i, v := range x.f {
		a := float64(v)
		switch typ {
		case "silu":
			a = a / (1 + math.Exp(-a))
		case "sigmoid":
			a = 1 / (1 + math.Exp(-a))
		case "relu":
			a = math.Max(a, 0)
		case "tanh":
			a = math.Tanh(a)
		case "exp":
			a = math.Exp(a)
		case "sqrt":
			a = math.Sqrt(a)
		}
		out.f[i] = float32(a)
	}
	return out
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
	parallel(n, func(lo, hi int) {
		for j := lo; j < hi; j++ {
			wr := w.f[j*k : (j+1)*k]
			var b float32
			if bias != nil {
				b = bias.f[j]
			}
			for i := 0; i < m; i++ {
				out.f[i*n+j] = dot(x.f[i*k:(i+1)*k], wr) + b
			}
		}
	})
	return out, nil
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
	// row major copies so that every product is a dot product of contiguous rows
	xm := func(bi int) []float32 {
		a := x.f[bi*m*k : (bi+1)*m*k]
		if !tx {
			return a
		}
		return transpose2(a, k, m)
	}
	ym := func(bi int) []float32 {
		a := y.f[bi*k*n : (bi+1)*k*n]
		if ty {
			return a
		}
		return transpose2(a, k, n)
	}
	shape := append(append([]int{}, xi.shape...), m, n)
	out := &value{shape: shape, f: make([]float32, batch*m*n)}
	parallel(batch*m, func(lo, hi int) {
		lastB := -1
		var a, bt []float32
		for r := lo; r < hi; r++ {
			bi, i := r/m, r%m
			if bi != lastB {
				a, bt, lastB = xm(int(xi.f[bi])), ym(int(yi.f[bi])), bi
			}
			for j := 0; j < n; j++ {
				out.f[(bi*m+i)*n+j] = dot(a[i*k:(i+1)*k], bt[j*k:(j+1)*k])
			}
		}
	})
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
func permute(x *value, start int, shape []int, src []int) *value {
	out := &value{shape: shape, f: make([]float32, numel(shape))}
	idx := make([]int, len(shape))
	off := start
	for i := range out.f {
		out.f[i] = x.f[off]
		for d := len(shape) - 1; d >= 0; d-- {
			idx[d]++
			off += src[d]
			if idx[d] < shape[d] {
				break
			}
			off -= src[d] * idx[d]
			idx[d] = 0
		}
	}
	return out
}

func transpose(x *value, perm []int32) (*value, error) {
	if len(perm) != len(x.shape) {
		return nil, fmt.Errorf("perm %v for shape %v", perm, x.shape)
	}
	st := strides(x.shape)
	shape, src := make([]int, len(perm)), make([]int, len(perm))
	for i, p := range perm {
		shape[i], src[i] = x.shape[p], st[p]
	}
	return permute(x, 0, shape, src), nil
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
	st := strides(x.shape)
	idx := make([]int, len(shape))
	for i := range out.f {
		off := 0
		for d := range shape {
			off += (idx[d] % x.shape[d]) * st[d]
		}
		out.f[i] = x.f[off]
		for d := len(shape) - 1; d >= 0; d-- {
			if idx[d]++; idx[d] < shape[d] {
				break
			}
			idx[d] = 0
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
	out := permute(x, base, shape, src)
	out.shape = outShape
	return out, nil
}

// padConst pads the last len(pad)/2 axes with c: pad holds (before, after) pairs in axis order.
func padConst(x *value, pad []int32, c float32) (*value, error) {
	rank := len(x.shape)
	k := len(pad) / 2
	if len(pad)%2 != 0 || k > rank {
		return nil, fmt.Errorf("pad %v for shape %v", pad, x.shape)
	}
	shape := append([]int{}, x.shape...)
	before := make([]int, rank)
	for j := 0; j < k; j++ {
		ax := rank - k + j
		before[ax] = int(pad[2*j])
		shape[ax] += int(pad[2*j]) + int(pad[2*j+1])
	}
	out := &value{shape: shape, f: make([]float32, numel(shape))}
	for i := range out.f {
		out.f[i] = c
	}
	ost := strides(shape)
	idx := make([]int, rank)
	for _, v := range x.f {
		o := 0
		for d := 0; d < rank; d++ {
			o += (idx[d] + before[d]) * ost[d]
		}
		out.f[o] = v
		for d := rank - 1; d >= 0; d-- {
			if idx[d]++; idx[d] < x.shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out, nil
}

func concat(xs []*value, a int) (*value, error) {
	if len(xs) == 0 {
		return nil, fmt.Errorf("nothing to concat")
	}
	ax, err := axis(a, len(xs[0].shape))
	if err != nil {
		return nil, err
	}
	shape := append([]int{}, xs[0].shape...)
	shape[ax] = 0
	for _, x := range xs {
		if len(x.shape) != len(shape) || x.f == nil {
			return nil, fmt.Errorf("cannot concat %v", x.shape)
		}
		for d := range shape {
			if d != ax && x.shape[d] != shape[d] {
				return nil, fmt.Errorf("cannot concat %v and %v", xs[0].shape, x.shape)
			}
		}
		shape[ax] += x.shape[ax]
	}
	outer, _, inner := around(shape, ax)
	out := &value{shape: shape, f: make([]float32, 0, numel(shape))}
	for o := 0; o < outer; o++ {
		for _, x := range xs {
			w := x.shape[ax] * inner
			out.f = append(out.f, x.f[o*w:(o+1)*w]...)
		}
	}
	return out, nil
}

// conv1d is a grouped 1 D convolution of x (n, c_in, w) with weight (c_out, c_in/groups, k).
func conv1d(x, w, bias *value, stride, dil []int32, padType string, pad []int32, groups int) (*value, error) {
	if len(x.shape) != 3 || len(w.shape) != 3 {
		return nil, fmt.Errorf("only 1 D convolutions are supported, got x %v weight %v", x.shape, w.shape)
	}
	n, cin, width := x.shape[0], x.shape[1], x.shape[2]
	cout, cg, k := w.shape[0], w.shape[1], w.shape[2]
	if groups <= 0 || cin != cg*groups || cout%groups != 0 {
		return nil, fmt.Errorf("x %v, weight %v and groups %d do not match", x.shape, w.shape, groups)
	}
	s, d := int(stride[0]), int(dil[0])
	pl, pr := 0, 0
	switch padType {
	case "valid":
	case "custom":
		pl, pr = int(pad[0]), int(pad[1])
	default:
		return nil, fmt.Errorf("pad_type %q is not supported", padType)
	}
	ow := (width+pl+pr-d*(k-1)-1)/s + 1
	out := &value{shape: []int{n, cout, ow}, f: make([]float32, n*cout*ow)}
	perGroup := cout / groups
	for b := 0; b < n; b++ {
		for oc := 0; oc < cout; oc++ {
			g := oc / perGroup
			for t := 0; t < ow; t++ {
				var acc float32
				if bias != nil {
					acc = bias.f[oc]
				}
				for ic := 0; ic < cg; ic++ {
					xr := x.f[(b*cin+g*cg+ic)*width:]
					for j := 0; j < k; j++ {
						p := t*s + j*d - pl
						if p >= 0 && p < width {
							acc += w.f[(oc*cg+ic)*k+j] * xr[p]
						}
					}
				}
				out.f[(b*cout+oc)*ow+t] = acc
			}
		}
	}
	return out, nil
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
	if idx.i == nil || len(lut.shape) != rank+2 || lut.shape[rank+1] != 1 {
		return nil, fmt.Errorf("indices %v and lut %v do not match", idx.shape, lut.shape)
	}
	grid, size := lut.shape[:rank], lut.shape[rank]
	for d := range grid {
		if grid[d] == 0 || idx.shape[d]%grid[d] != 0 {
			return nil, fmt.Errorf("lut grid %v does not divide %v", grid, idx.shape)
		}
	}
	blocks := blockIndex(idx.shape, grid)
	out := &value{shape: idx.shape, f: make([]float32, len(idx.i))}
	for i, q := range idx.i {
		out.f[i] = lut.f[blocks[i]*size+int(q)]
	}
	return out, nil
}

// shiftScale is (data - offset) * scale with one scale (and offset) per block.
func shiftScale(data, scale, offset *value) (*value, error) {
	rank := len(data.shape)
	if data.i == nil || len(scale.shape) != rank {
		return nil, fmt.Errorf("data %v and scale %v do not match", data.shape, scale.shape)
	}
	blocks := blockIndex(data.shape, scale.shape)
	out := &value{shape: data.shape, f: make([]float32, len(data.i))}
	for i, q := range data.i {
		v := float32(q)
		if offset != nil {
			if offset.f != nil {
				v -= offset.f[blocks[i]]
			} else {
				v -= float32(offset.i[blocks[i]])
			}
		}
		out.f[i] = v * scale.f[blocks[i]]
	}
	return out, nil
}
