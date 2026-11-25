package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/backend"
	"github.com/shibukawa/bonsaiai/internal/backend/coreai"
	"github.com/shibukawa/bonsaiai/internal/imagegen/milpart"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
	"github.com/shibukawa/bonsaiai/internal/mxfp4"
	"github.com/shibukawa/bonsaiai/internal/types"
)

// mx times experts in MXFP4 on the Neural Engine, given as Core AI's MX arrays: e2m1 values two
// a byte (float4E2M1FN, the checkpoint's bytes as they are) and an exponent byte for each 32
// in-features (float8E8M0FN), dequantized by blockwise_shift_scale. An expert is gpt-oss's: gate
// and up interleaved in one (2 inter, hidden) array, down (hidden, inter); the program's
// activation is a plain SwiGLU of the even and odd rows (the timing does not depend on it).
// Against the host's float32 of the same codes, then timed as moe4.
//
//	aicheck mx [-model gptoss] [-k 4] [-layers 1] [-calls 100]
func mx(argv []string) error {
	fs := flag.NewFlagSet("mx", flag.ExitOnError)
	model := fs.String("model", "gptoss", "the layer's sizes (as moe)")
	layers := fs.Int("layers", 1, "layers' experts held")
	calls := fs.Int("calls", 100, "calls timed")
	topk := fs.Int("k", 0, "experts a call (0: the model's)")
	constant := fs.Bool("const", false, "the experts as constants of the program (MX constants), timed alone")
	only := fs.Bool("only", false, "with -const: the program only x times gate|up (to see what leaves the Neural Engine)")
	write := fs.String("write", "", "write the program here (an .aimodel) and stop, for coreai-build on other chips")
	form := fs.String("form", "", "with -const: mx (MX constants) or lut4 (the codes with one table of e2m1 values a 16 rows, the scales left out); at run time: lut (default) or int4")
	fs.Parse(argv)
	sh, ok := moeShapes[*model]
	if !ok {
		return fmt.Errorf("mx: no model %s", *model)
	}
	H, I, E, K := sh.hidden, sh.inter, sh.experts, sh.topK
	if *topk > 0 {
		K = *topk
	}
	if H%32 != 0 || I%32 != 0 {
		return fmt.Errorf("mx: %d and %d are not blocks of 32", H, I)
	}
	ctx := context.Background()
	if err := llm.LockNPU(ctx); err != nil {
		return err
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "aicheck-mx-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	expertBytes := 2*I*H/2 + 2*I*H/32 + H*I/2 + H*I/32
	fmt.Printf("%s in MXFP4: an expert %.1f MB; %d layers held %.2f GB\n", *model, float64(expertBytes)/1e6, *layers, float64(*layers*E*expertBytes)/1e9)

	// at run time: the codes a uint4 input (the checkpoint's bytes as they are), looked up in a
	// constant table of e2m1's 16 values (lut_to_dense), times the blocks' scales (fp16, an input
	// of (rows, cols / 32)) as the weight's elements; or (-form int4) int4 codes dequantized by
	// blockwise_shift_scale with the same scales (not MXFP4's values: for the timing)
	body := func(g *milpart.Graph) {
		lin := func(x, w *mlpackage.Var, out int) *mlpackage.Var {
			return g.Op("linear", map[string]any{"x": x, "weight": w}, 1, 1, out)
		}
		table := make([]float32, 16)
		copy(table, mxfp4.Values[:])
		lut, err := g.B.F16s("e2m1", []int{1, 1, 16, 1}, table)
		if err != nil {
			panic(err)
		}
		weight := func(name string, rows, cols int) *mlpackage.Var {
			s := g.Input("s"+name, rows, cols/32)
			if *form == "int4" {
				return g.B.Op("constexpr_blockwise_shift_scale", map[string]any{"data": g.InputI4(name, rows, cols), "scale": s}, mlpackage.Float16, rows, cols)
			}
			v := g.B.Op("constexpr_lut_to_dense", map[string]any{"indices": g.InputU4(name, rows, cols), "lut": lut}, mlpackage.Float16, rows, cols)
			if *form == "lutonly" { // the table alone, the scales left out (timing)
				return g.Add(v, g.Mul(g.Reshape(g.Slice(s, 1, 0, 1), rows, 1), g.B.ScalarF16(0)))
			}
			return g.Reshape(g.Mul(g.Reshape(v, rows, cols/32, 32), g.Reshape(s, rows, cols/32, 1)), rows, cols)
		}
		x := g.Input("x", 1, 1, H)
		var acc *mlpackage.Var
		for k := 0; k < K; k++ {
			gu := weight(fmt.Sprintf("gu%d", k), 2*I, H)
			dn := weight(fmt.Sprintf("dn%d", k), H, I)
			h := g.Reshape(lin(x, gu, 2*I), 1, I, 2)
			gate, up := g.Reshape(g.Slice(h, 2, 0, 1), 1, 1, I), g.Reshape(g.Slice(h, 2, 1, 2), 1, 1, I)
			a := g.Mul(g.SiLU(gate), up)
			y := g.Mul(lin(a, dn, H), g.Input(fmt.Sprintf("r%d", k), 1, 1, 1))
			if acc == nil {
				acc = y
			} else {
				acc = g.Add(acc, y)
			}
		}
		g.Output(acc)
	}
	rng := rand.New(rand.NewSource(1))
	// codes: random nibbles; exponents around 2^-6 (gpt-oss's experts' scales lie about there)
	fill := func(w, e []byte) {
		for i := range w {
			w[i] = byte(rng.Intn(256))
		}
		for i := range e {
			e[i] = byte(127 - 8 + rng.Intn(4))
		}
	}
	type consts struct{ gu, sgu, dn, sdn []byte }
	var cw []consts
	if *constant {
		cw = make([]consts, K)
		for k := range cw {
			c := consts{make([]byte, 2*I*H/2), make([]byte, 2*I*H/32), make([]byte, H*I/2), make([]byte, H*I/32)}
			fill(c.gu, c.sgu)
			fill(c.dn, c.sdn)
			cw[k] = c
		}
		body = func(g *milpart.Graph) {
			must := func(v *mlpackage.Var, err error) *mlpackage.Var {
				if err != nil {
					panic(err)
				}
				return v
			}
			x := g.Input("x", 1, 1, H)
			var acc *mlpackage.Var
			for k, c := range cw {
				w := func(name string, rows, cols int, codes, exps []byte) *mlpackage.Var {
					if *form == "lut4" { // else mx
						lut := make([]uint16, rows/16*16)
						for i := range lut {
							lut[i] = mlpackage.Float16Bits(mxfp4.Values[i%16] / 64)
						}
						return must(g.B.LUT4(name, []int{rows, cols}, []int{16, cols}, codes, lut))
					}
					return must(g.B.MXFP4(name, rows, cols, codes, exps))
				}
				lin := func(x, w *mlpackage.Var, out int) *mlpackage.Var {
					return g.Op("linear", map[string]any{"x": x, "weight": w}, 1, 1, out)
				}
				gu := w(fmt.Sprintf("gu%d", k), 2*I, H, c.gu, c.sgu)
				if *only {
					g.Output(lin(x, gu, 2*I))
					return
				}
				dn := w(fmt.Sprintf("dn%d", k), H, I, c.dn, c.sdn)
				mm := func(x, w *mlpackage.Var, _ ...int) *mlpackage.Var { return lin(x, w, w.Shape[0]) }
				h := g.Reshape(lin(x, gu, 2*I), 1, I, 2)
				gate, up := g.Reshape(g.Slice(h, 2, 0, 1), 1, 1, I), g.Reshape(g.Slice(h, 2, 1, 2), 1, 1, I)
				y := g.Mul(mm(g.Mul(g.SiLU(gate), up), dn, 1, 1, H), g.B.ScalarF16(1/float32(K)))
				if acc == nil {
					acc = y
				} else {
					acc = g.Add(acc, y)
				}
			}
			g.Output(acc)
		}
	}
	path := filepath.Join(dir, "mx.aimodel")
	if _, err := llm.WritePart(path, &milpart.Part{Name: "mx", Model: "moe", Body: body}, nil, llm.Options{Backend: llm.BackendCoreAI, Format: "fp16"}, nil); err != nil {
		return err
	}
	if *write != "" {
		return os.Rename(path, *write)
	}
	t := time.Now()
	gp, err := dev.Compile(ctx, types.GraphSource{Path: path})
	if err != nil {
		return err
	}
	defer gp.Close()
	fmt.Printf("  compiled in %.1fs, ane=%v %v\n", time.Since(t).Seconds(), gp.Placement().OnNPU(), gp.Placement().Ops)
	args := map[string]types.ArgInfo{}
	for _, a := range gp.Args() {
		args[a.Name] = a
	}
	alloc := func(name string) (backend.Buffer, error) {
		a := args[name]
		if ia, ok := gp.(backend.InputAllocator); ok && a.Kind == types.ArgInput {
			if b, err := ia.AllocInput(a.Index); err != nil || b != nil {
				return b, err
			}
		}
		return dev.AllocBuffer(a.Bytes, types.BufferKind(0))
	}
	if *constant {
		req, _ := gp.NewRequest()
		xb, _ := alloc("x")
		xv := make([]float32, H)
		for i := range xv {
			xv[i] = float32(rng.NormFloat64())
		}
		bonsaiai.EncodeF16(xb.Bytes(), xv)
		bonsaiai.DecodeF16(xv, xb.Bytes())
		req.Bind(args["x"].Index, xb)
		yb, _ := alloc("y")
		req.Bind(args["y"].Index, yb)
		run := func() error {
			f, err := req.Submit()
			if err != nil {
				return err
			}
			return f.Wait(ctx)
		}
		if err := run(); err != nil {
			return err
		}
		got := make([]float32, H)
		bonsaiai.DecodeF16(got, yb.Bytes())
		want := make([]float32, H)
		for _, c := range cw {
			mxExpert(want, c.gu, c.sgu, c.dn, c.sdn, xv, I, 1/float32(K))
		}
		fmt.Printf("  against the host's float32: cos %.6f rel %.4f\n", cosine(got, want), relErr(got, want))
		for i := 0; i < 5; i++ {
			run()
		}
		t := time.Now()
		for i := 0; i < *calls; i++ {
			if err := run(); err != nil {
				return err
			}
		}
		fmt.Printf("  a call %.3f ms\n", float64(time.Since(t).Microseconds())/1e3/float64(*calls))
		return nil
	}
	type expert struct {
		gu, dn, sgu, sdn backend.Buffer
		egu, edn         []byte // the exponent bytes the scales hold
	}
	// scales: fp16 2^(e-127) from the exponent bytes
	scales := func(dst []byte, e []byte) {
		f := make([]float32, len(e))
		for i, b := range e {
			f[i] = float32(math.Ldexp(1, int(b)-127))
		}
		bonsaiai.EncodeF16(dst, f)
	}
	held := make([][]expert, *layers)
	t = time.Now()
	for l := range held {
		held[l] = make([]expert, E)
		for e := range held[l] {
			var x expert
			for _, b := range []struct {
				p    *backend.Buffer
				name string
			}{{&x.gu, "gu0"}, {&x.dn, "dn0"}, {&x.sgu, "sgu0"}, {&x.sdn, "sdn0"}} {
				if *b.p, err = alloc(b.name); err != nil {
					return err
				}
			}
			x.egu, x.edn = make([]byte, 2*I*H/32), make([]byte, H*I/32)
			fill(x.gu.Bytes(), x.egu)
			fill(x.dn.Bytes(), x.edn)
			scales(x.sgu.Bytes(), x.egu)
			scales(x.sdn.Bytes(), x.edn)
			held[l][e] = x
		}
	}
	fmt.Printf("  arrays made and filled in %.1fs\n", time.Since(t).Seconds())
	req, _ := gp.NewRequest()
	xb, _ := alloc("x")
	xv := make([]float32, H)
	for i := range xv {
		xv[i] = float32(rng.NormFloat64())
	}
	bonsaiai.EncodeF16(xb.Bytes(), xv)
	bonsaiai.DecodeF16(xv, xb.Bytes())
	req.Bind(args["x"].Index, xb)
	for k := 0; k < K; k++ {
		r, _ := alloc(fmt.Sprintf("r%d", k))
		bonsaiai.EncodeF16(r.Bytes(), []float32{1 / float32(K)})
		req.Bind(args[fmt.Sprintf("r%d", k)].Index, r)
	}
	yb, _ := alloc("y")
	req.Bind(args["y"].Index, yb)
	bind := func(l int, chosen []int) {
		for k, e := range chosen {
			x := held[l][e]
			req.Bind(args[fmt.Sprintf("gu%d", k)].Index, x.gu)
			req.Bind(args[fmt.Sprintf("dn%d", k)].Index, x.dn)
			req.Bind(args[fmt.Sprintf("sgu%d", k)].Index, x.sgu)
			req.Bind(args[fmt.Sprintf("sdn%d", k)].Index, x.sdn)
		}
	}
	run := func() error {
		f, err := req.Submit()
		if err != nil {
			return err
		}
		return f.Wait(ctx)
	}
	chosen := make([]int, K)
	for k := range chosen {
		chosen[k] = K - 1 - k
	}
	bind(0, chosen)
	if err := run(); err != nil {
		return err
	}
	got := make([]float32, H)
	bonsaiai.DecodeF16(got, yb.Bytes())
	want := make([]float32, H)
	for _, e := range chosen {
		x := held[0][e]
		mxExpert(want, x.gu.Bytes(), x.egu, x.dn.Bytes(), x.edn, xv, I, 1/float32(K))
	}
	fmt.Printf("  against the host's float32: cos %.6f rel %.4f\n", cosine(got, want), relErr(got, want))
	timeIt := func(step func() error) (time.Duration, error) {
		for i := 0; i < 5; i++ {
			if err := step(); err != nil {
				return 0, err
			}
		}
		t := time.Now()
		for i := 0; i < *calls; i++ {
			if err := step(); err != nil {
				return 0, err
			}
		}
		return time.Since(t) / time.Duration(*calls), nil
	}
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1e3 }
	same, err := timeIt(run)
	if err != nil {
		return err
	}
	one, err := timeIt(func() error { bind(0, rng.Perm(E)[:K]); return run() })
	if err != nil {
		return err
	}
	all, err := timeIt(func() error { bind(rng.Intn(*layers), rng.Perm(E)[:K]); return run() })
	if err != nil {
		return err
	}
	fmt.Printf("  a call: the same experts %.3f ms, chosen in one layer %.3f ms, chosen in any of %d layers %.3f ms\n", ms(same), ms(one), *layers, ms(all))
	return nil
}

// mxRef is the host's float32 of an MXFP4 matrix (rows of len(x) in-features) times x.
func mxRef(out []float32, w, e []byte, x []float32) {
	n := len(x)
	row := make([]float32, n)
	for r := range out {
		mxfp4.Dequantize(row, w[r*n/2:(r+1)*n/2], e[r*n/32:(r+1)*n/32])
		var s float64
		for i, v := range row {
			s += float64(v) * float64(x[i])
		}
		out[r] = float32(s)
	}
}

// mxExpert adds r times the expert's output on x to out, as the program computes it (its
// activation fp16).
func mxExpert(out []float32, gu, sgu, dn, sdn []byte, x []float32, I int, r float32) {
	h := make([]float32, 2*I)
	mxRef(h, gu, sgu, x)
	a := make([]float32, I)
	for j := range a {
		g, u := h[2*j], h[2*j+1]
		a[j] = g / (1 + float32(math.Exp(float64(-g)))) * u
	}
	ab := make([]byte, 2*I)
	bonsaiai.EncodeF16(ab, a)
	bonsaiai.DecodeF16(a, ab)
	d := make([]float32, len(out))
	mxRef(d, dn, sdn, a)
	for o := range out {
		out[o] += d[o] * r
	}
}
