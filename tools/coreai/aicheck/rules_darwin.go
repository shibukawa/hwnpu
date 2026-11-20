package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// rules checks the Neural Engine's numerical rules that arXiv 2606.22283 describes, on small
// programs with chosen inputs against the values float32 gives (design.md 16.274):
//
//   - slice: a slice of the width (last) axis starting at 0 or not, at values 1 to 5000 (the
//     paper: a fixed x16 gain on a slice starting past 0, Inf above 4094)
//   - conv: a linear whose sum or partial sums pass 2^15 (the paper: keep conv outputs under
//     2^15)
//   - act: the activations over -20..20 and at large inputs (the paper: 33 point piecewise
//     linear tables, softplus(+inf) = 0)
//   - sub: products and inputs below fp16's normal range
//   - inner1: a matmul of inner dimension 1 (memory: ane-matmul-inner-dim-1)
//   - div: a division by a constant
//
// HWNPU_COREAI_UNITS=cpu runs the same programs on Core AI's CPU, to tell the engine's errors
// from the lowering's.
//
//	aicheck rules [-only slice,conv,...]
func rules(args []string) error {
	fs := flag.NewFlagSet("rules", flag.ExitOnError)
	only := fs.String("only", "", "the checks to run, comma separated (default all)")
	fs.Parse(args)
	ctx := context.Background()
	if !coreai.OnCPU() {
		if err := llm.LockNPU(ctx); err != nil {
			return err
		}
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "aicheck-rules-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	r := &ruleRunner{ctx: ctx, dev: dev, dir: dir}
	checks := []struct {
		name string
		f    func() error
	}{
		{"slice", r.slice}, {"conv", r.conv}, {"act", r.act}, {"sub", r.sub}, {"inner1", r.inner1}, {"div", r.div}, {"log", r.log},
	}
	for _, c := range checks {
		if *only != "" && !strings.Contains(","+*only+",", ","+c.name+",") {
			continue
		}
		fmt.Printf("== %s\n", c.name)
		if err := c.f(); err != nil {
			fmt.Printf("  FAIL %v\n", err)
		}
	}
	return nil
}

type ruleRunner struct {
	ctx context.Context
	dev backend.Device
	dir string
	n   int
}

// call writes the program body makes, loads it, runs it once on the fp16 inputs given by name
// and returns its output and where it ran.
func (r *ruleRunner) call(body func(g *milpart.Graph), in map[string][]float32) ([]float32, string, error) {
	r.n++
	name := fmt.Sprintf("r%d", r.n)
	path := filepath.Join(r.dir, name+".mlpackage")
	if _, err := llm.WritePart(path, &milpart.Part{Name: name, Model: "rules", Body: body}, nil, llm.Options{Backend: llm.BackendCoreML, Format: "fp16"}, nil); err != nil {
		return nil, "", err
	}
	g, err := r.dev.Compile(r.ctx, types.GraphSource{Path: path})
	if err != nil {
		return nil, "", err
	}
	defer g.Close()
	where := "ane"
	if p := g.Placement(); !p.OnNPU() {
		where = fmt.Sprintf("NOT-ANE(%v %s)", p.Ops, p.Reason)
	}
	if coreai.OnCPU() {
		where = "cpu"
	}
	req, err := g.NewRequest()
	if err != nil {
		return nil, "", err
	}
	var out backend.Buffer
	var outN int
	for _, a := range g.Args() {
		b, err := r.dev.AllocBuffer(a.Bytes, types.BufferKind(0))
		if err != nil {
			return nil, "", err
		}
		req.Bind(a.Index, b)
		if a.Kind == types.ArgInput {
			v, ok := in[a.Name]
			if !ok {
				return nil, "", fmt.Errorf("no values for input %s", a.Name)
			}
			hwnpu.EncodeF16(b.Bytes(), v)
		} else {
			out, outN = b, a.Bytes/2
		}
	}
	f, err := req.Submit()
	if err == nil {
		err = f.Wait(r.ctx)
	}
	if err != nil {
		return nil, "", err
	}
	got := make([]float32, outN)
	hwnpu.DecodeF16(got, out.Bytes())
	return got, where, nil
}

func fill(n int, v float32) []float32 {
	f := make([]float32, n)
	for i := range f {
		f[i] = v
	}
	return f
}

// f16 rounds to fp16.
func f16(v float32) float32 {
	b := make([]byte, 2)
	hwnpu.EncodeF16(b, []float32{v})
	o := make([]float32, 1)
	hwnpu.DecodeF16(o, b)
	return o[0]
}

// stats is the share of got equal to want (after fp16 rounding) and the ratio got/want at the
// first element that differs.
func stats(got, want []float32) string {
	bad, first := 0, -1
	for i := range want {
		w := f16(want[i])
		if got[i] != w && !(math.IsNaN(float64(got[i])) && math.IsNaN(float64(w))) {
			if first < 0 {
				first = i
			}
			bad++
		}
	}
	if bad == 0 {
		return "exact"
	}
	return fmt.Sprintf("%d/%d differ, first [%d] got %g want %g (x%.4g)", bad, len(want), first, got[first], want[first], got[first]/want[first])
}

// slice: x (1, 16, 8, 8192) of one value; y = slice of axis 3 from lo, 2048 wide, plus a
// zero input (so the slice is not the whole program). Then the same on axes 1 and 2.
func (r *ruleRunner) slice() error {
	const C, H, W, n = 16, 8, 8192, 2048
	axes := []int{3, 2, 1}
	if os.Getenv("RULES_LOS") != "" {
		axes = []int{3}
	}
	for _, axis := range axes {
		los := []int{0, 16, 1000, 4096}
		if e := os.Getenv("RULES_LOS"); e != "" {
			los = nil
			for _, f := range strings.Split(e, ",") {
				var v int
				fmt.Sscan(f, &v)
				los = append(los, v)
			}
		}
		width := n
		if e := os.Getenv("RULES_WIDTH"); e != "" {
			fmt.Sscan(e, &width)
		}
		switch axis {
		case 2:
			los, width = []int{0, 1, 4}, 4
		case 1:
			los, width = []int{0, 1, 8}, 8
		}
		for _, lo := range los {
			vals := []float32{1, 1000, 4000, 4094, 4096, 5000, 30000}
			if os.Getenv("RULES_LOS") != "" {
				vals = []float32{4000, 4100}
			}
			for _, v := range vals {
				shape := []int{1, C, H, W}
				shape[axis] = width
				body := func(g *milpart.Graph) {
					x := g.Input("x", 1, C, H, W)
					z := g.Input("z", shape...)
					g.Output(g.Add(g.Slice(x, axis, lo, lo+width), z))
				}
				got, where, err := r.call(body, map[string][]float32{"x": fill(C*H*W, v), "z": fill(shape[1]*shape[2]*shape[3], 0)})
				if err != nil {
					return err
				}
				fmt.Printf("  axis %d lo %4d v %6g: %-6s %s\n", axis, lo, v, where, stats(got, fill(len(got), v)))
			}
		}
	}
	return nil
}

// conv: a linear of in inputs of value a to 64 outputs. Weights all 1 (the sum is in*a), or +1
// for the first half and -1 for the second (the sum is 0, the partial sums reach in/2*a).
func (r *ruleRunner) conv() error {
	const rows, out = 64, 64
	for _, in := range []int{512, 1024} {
		for _, cancel := range []bool{false, true} {
			w := make([]float32, out*in)
			for o := 0; o < out; o++ {
				for i := 0; i < in; i++ {
					w[o*in+i] = 1
					if cancel && i >= in/2 {
						w[o*in+i] = -1
					}
				}
			}
			for _, a := range []float32{8, 16, 32, 48, 63, 64, 65, 100, 127} {
				body := func(g *milpart.Graph) {
					x := g.Input("x", 1, rows, in)
					g.Output(g.LinearConst(x, "w", w, out, nil))
				}
				got, where, err := r.call(body, map[string][]float32{"x": fill(rows*in, a)})
				if err != nil {
					return err
				}
				want, peak := float32(in)*a, float32(in)*a
				if cancel {
					want, peak = 0, float32(in/2)*a
				}
				if want > 65504 {
					continue
				}
				fmt.Printf("  in %4d cancel %-5v a %3g: sum %6g peak %6g: %-6s %s\n", in, cancel, a, want, peak, where, stats(got, fill(len(got), want)))
			}
		}
	}
	// a 1x1 convolution of (1, in, 8, 8): the same as conv rather than linear
	for _, a := range []float32{32, 64, 100} {
		const in, cout = 512, 64
		body := func(g *milpart.Graph) {
			x := g.Input("x", 1, in, 8, 8)
			g.Output(g.Conv2D(x, "w", fill(cout*in, 1), nil, cout, 1, 1))
		}
		got, where, err := r.call(body, map[string][]float32{"x": fill(in*64, a)})
		if err != nil {
			return err
		}
		fmt.Printf("  conv1x1 in %d a %3g: sum %6g: %-6s %s\n", in, a, float32(in)*a, where, stats(got, fill(len(got), float32(in)*a)))
	}
	// reduce_sum over the last axis, the same values
	for _, a := range []float32{32, 64, 100} {
		const in = 512
		body := func(g *milpart.Graph) {
			x := g.Input("x", 1, 64, in)
			g.Output(g.Reduce("sum", x, -1))
		}
		got, where, err := r.call(body, map[string][]float32{"x": fill(64*in, a)})
		if err != nil {
			return err
		}
		fmt.Printf("  reduce_sum in %d a %3g: sum %6g: %-6s %s\n", in, a, float32(in)*a, where, stats(got, fill(len(got), float32(in)*a)))
	}
	return nil
}

// act: each activation over 65536 points of -20..20, and the points 30, 100, 1000, 60000.
func (r *ruleRunner) act() error {
	const n = 65536
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = f16(-20 + 40*float32(i)/(n-1))
	}
	big := []float32{-60000, -1000, -100, -30, 30, 100, 1000, 60000}
	copy(xs[n-len(big):], big)
	sig := func(x float64) float64 { return 1 / (1 + math.Exp(-x)) }
	acts := []struct {
		name string
		op   func(g *milpart.Graph, x *mlpackage.Var) *mlpackage.Var
		ref  func(x float64) float64
	}{
		{"sigmoid", func(g *milpart.Graph, x *mlpackage.Var) *mlpackage.Var { return g.Unary("sigmoid", x) }, sig},
		{"silu", func(g *milpart.Graph, x *mlpackage.Var) *mlpackage.Var { return g.Unary("silu", x) }, func(x float64) float64 { return x * sig(x) }},
		{"silu(exp)", func(g *milpart.Graph, x *mlpackage.Var) *mlpackage.Var { return g.SiLU(x) }, func(x float64) float64 { return x * sig(x) }},
		{"softplus", func(g *milpart.Graph, x *mlpackage.Var) *mlpackage.Var { return g.Unary("softplus", x) }, func(x float64) float64 {
			if x > 30 {
				return x
			}
			return math.Log1p(math.Exp(x))
		}},
		{"tanh", func(g *milpart.Graph, x *mlpackage.Var) *mlpackage.Var { return g.Unary("tanh", x) }, math.Tanh},
		{"gelu", func(g *milpart.Graph, x *mlpackage.Var) *mlpackage.Var { return g.Activation(x, "gelu") }, func(x float64) float64 { return 0.5 * x * (1 + math.Erf(x/math.Sqrt2)) }},
		{"exp", func(g *milpart.Graph, x *mlpackage.Var) *mlpackage.Var { return g.Unary("exp", x) }, math.Exp},
	}
	for _, a := range acts {
		body := func(g *milpart.Graph) {
			x := g.Input("x", 1, 64, n/64)
			g.Output(a.op(g, x))
		}
		got, where, err := r.call(body, map[string][]float32{"x": xs})
		if err != nil {
			fmt.Printf("  %-9s FAIL %v\n", a.name, err)
			continue
		}
		// the largest error relative to max(|ref|, 1) inside -20..20, and the kinks: where the
		// second difference of the output passes 8 times the reference's
		worst, at := 0.0, 0.0
		kinks := 0
		m := n - len(big)
		for i := 0; i < m; i++ {
			ref := a.ref(float64(xs[i]))
			e := math.Abs(float64(got[i])-ref) / math.Max(math.Abs(ref), 1)
			if e > worst && !math.IsInf(ref, 0) && math.Abs(ref) < 65504 {
				worst, at = e, float64(xs[i])
			}
		}
		// kinks at a coarser step (every 64th point, 0.04) to stay above fp16's rounding
		const step = 64
		for i := step; i+step < m; i += step {
			d2 := float64(got[i+step]) - 2*float64(got[i]) + float64(got[i-step])
			r2 := a.ref(float64(xs[i+step])) - 2*a.ref(float64(xs[i])) + a.ref(float64(xs[i-step]))
			if math.Abs(d2-r2) > 0.02*math.Max(math.Abs(a.ref(float64(xs[i]))), 1)*0.04 {
				kinks++
			}
		}
		var tail []string
		for i, x := range big {
			tail = append(tail, fmt.Sprintf("%g:%g", x, got[m+i]))
		}
		fmt.Printf("  %-9s %-6s worst rel %.3g at %.3g, second-difference misses %d of %d; %s\n", a.name, where, worst, at, kinks, m/step-2, strings.Join(tail, " "))
	}
	return nil
}

// sub: x times y where the product falls below fp16's smallest normal (6.1e-5), and a
// subnormal input added to 0.
func (r *ruleRunner) sub() error {
	const n = 64 * 1024
	for _, p := range [][2]float32{{1e-2, 1e-2}, {1e-2, 5e-3}, {1e-2, 1e-3}, {1e-3, 1e-3}, {1e-2, 1e-4}} {
		body := func(g *milpart.Graph) {
			g.Output(g.Mul(g.Input("x", 1, 64, n/64), g.Input("y", 1, 64, n/64)))
		}
		got, where, err := r.call(body, map[string][]float32{"x": fill(n, p[0]), "y": fill(n, p[1])})
		if err != nil {
			return err
		}
		fmt.Printf("  %g x %g = %g: %-6s %s\n", p[0], p[1], f16(p[0]*p[1]), where, stats(got, fill(n, p[0]*p[1])))
	}
	for _, v := range []float32{1e-4, 3e-5, 1e-6} {
		body := func(g *milpart.Graph) {
			g.Output(g.Add(g.Input("x", 1, 64, n/64), g.Input("z", 1, 64, n/64)))
		}
		got, where, err := r.call(body, map[string][]float32{"x": fill(n, v), "z": fill(n, 0)})
		if err != nil {
			return err
		}
		fmt.Printf("  %g + 0: %-6s %s\n", v, where, stats(got, fill(n, v)))
	}
	// a linear whose products are subnormal but whose sum is normal: 512 x 1e-3 x 1e-3 = 5.1e-4
	body := func(g *milpart.Graph) {
		g.Output(g.LinearConst(g.Input("x", 1, 64, 512), "w", fill(64*512, 1e-3), 64, nil))
	}
	got, where, err := r.call(body, map[string][]float32{"x": fill(64*512, 1e-3)})
	if err != nil {
		return err
	}
	fmt.Printf("  linear 512 x 1e-3 x 1e-3 = %g: %-6s %s\n", 512e-6, where, stats(got, fill(len(got), 512e-6)))
	return nil
}

// inner1: (1, 8, M, 1) x (1, 8, 1, 128) for M 64 and 1, and the same as a broadcast multiply.
func (r *ruleRunner) inner1() error {
	for _, M := range []int{64, 1} {
		if err := r.inner1M(M); err != nil {
			return err
		}
	}
	return nil
}

func (r *ruleRunner) inner1M(M int) error {
	a, b := make([]float32, 8*M), make([]float32, 8*128)
	for i := range a {
		a[i] = f16(float32(i%64)/32 - 1 + 0.01)
	}
	for i := range b {
		b[i] = f16(float32(i%128)/64 - 1)
	}
	want := make([]float32, 8*M*128)
	for h := 0; h < 8; h++ {
		for i := 0; i < M; i++ {
			for j := 0; j < 128; j++ {
				want[(h*M+i)*128+j] = a[h*M+i] * b[h*128+j]
			}
		}
	}
	for _, mul := range []bool{false, true} {
		body := func(g *milpart.Graph) {
			x, y := g.Input("a", 1, 8, M, 1), g.Input("b", 1, 8, 1, 128)
			if mul {
				g.Output(g.Mul(x, y))
			} else {
				g.Output(g.MatMul(x, y, false, false))
			}
		}
		got, where, err := r.call(body, map[string][]float32{"a": a, "b": b})
		if err != nil {
			return err
		}
		worst := 0.0
		for i := range want {
			worst = math.Max(worst, math.Abs(float64(got[i]-want[i]))/math.Max(math.Abs(float64(want[i])), 1e-2))
		}
		fmt.Printf("  M %2d matmul=%-5v: %-6s worst rel %.3g\n", M, !mul, where, worst)
	}
	return nil
}

// div: x / 3 and x / 7 (ailower turns it into a multiply unless HWNPU_AILOWER_DIVIDE=1).
func (r *ruleRunner) div() error {
	const n = 64 * 1024
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = f16(float32(i%2048) - 1024)
	}
	for _, d := range []float32{3, 7, 0.1} {
		body := func(g *milpart.Graph) {
			g.Output(g.Op("real_div", map[string]any{"x": g.Input("x", 1, 64, n/64), "y": g.B.ScalarF16(d)}, 1, 64, n/64))
		}
		got, where, err := r.call(body, map[string][]float32{"x": xs})
		if err != nil {
			return err
		}
		worst := 0.0
		for i := range xs {
			w := float64(xs[i]) / float64(f16(d))
			if e := math.Abs(float64(got[i])-w) / math.Max(math.Abs(w), 1); e > worst {
				worst = e
			}
		}
		fmt.Printf("  x / %g: %-6s worst rel %.3g (fp16 half ulp 4.9e-4)\n", d, where, worst)
	}
	return nil
}

// log: log, sqrt and rsqrt at 1 .. 65504 (softplus, lowered as log(1 + exp(x)), went to 0 where
// exp(x) passes 2^15).
func (r *ruleRunner) log() error {
	const n = 64 * 1024
	xs := make([]float32, n)
	for i := range xs {
		xs[i] = f16(2 + 65000*float32(i)/(n-1))
	}
	ops := []struct {
		name string
		ref  func(float64) float64
	}{{"log", math.Log}, {"sqrt", math.Sqrt}, {"rsqrt", func(x float64) float64 { return 1 / math.Sqrt(x) }}}
	for _, o := range ops {
		body := func(g *milpart.Graph) { g.Output(g.Unary(o.name, g.Input("x", 1, 64, n/64))) }
		got, where, err := r.call(body, map[string][]float32{"x": xs})
		if err != nil {
			return err
		}
		firstBad := float32(-1)
		worst := 0.0
		for i := range xs {
			w := o.ref(float64(xs[i]))
			e := math.Abs(float64(got[i])-w) / math.Abs(w)
			if e > 0.01 && firstBad < 0 {
				firstBad = xs[i]
			}
			worst = math.Max(worst, e)
		}
		fmt.Printf("  %-5s %-6s worst rel %.3g, first above 1%% at x=%g (got %g)\n", o.name, where, worst, firstBad, got[max(0, indexOf(xs, firstBad))])
	}
	return nil
}

func indexOf(xs []float32, v float32) int {
	for i, x := range xs {
		if x == v {
			return i
		}
	}
	return -1
}
