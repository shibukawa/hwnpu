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
	"github.com/shibukawa/bonsaiai/internal/types"
)

// gatherCheck checks the experts picked inside the program: a constant table of E experts'
// weights (E, out, in), the K expert numbers an input (cast to int16, the Neural Engine's index
// type), gather_nd taking their matrices and a product with x. Against the host's float32 of
// the same table for several choices, and timed against the same product with K constant
// matrices (what a dense model of the active size costs).
//
//	aicheck gather [-e 8] [-out 256] [-in 256] [-k 2] [-int8] [-calls 200]
func gatherCheck(args []string) error {
	fs := flag.NewFlagSet("gather", flag.ExitOnError)
	E := fs.Int("e", 8, "experts in the table")
	O := fs.Int("out", 256, "an expert's output width")
	I := fs.Int("in", 256, "an expert's input width")
	K := fs.Int("k", 2, "experts picked a call")
	int8w := fs.Bool("int8", false, "the table in int8 with an fp16 scale a row, else fp16")
	raw8 := fs.Bool("raw8", true, "with -int8: the table an int8 constant gathered, cast and scaled after (else a constexpr dequantized before)")
	calls := fs.Int("calls", 200, "calls timed")
	checks := fs.Int("checks", 4, "choices checked against the host")
	each := fs.Bool("each", false, "a gather_nd and a product for each picked expert (one index each), concatenated")
	fs.Parse(args)
	ctx := context.Background()
	if err := llm.LockNPU(ctx); err != nil {
		return err
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "aicheck-gather-*")
	if err != nil {
		return err
	}
	if os.Getenv("KEEP") != "" {
		fmt.Println("packages in", dir)
	} else {
		defer os.RemoveAll(dir)
	}
	e, o, in, k := *E, *O, *I, *K
	fmt.Printf("table %d x %d x %d (%s, %.1f MB), %d picked\n", e, o, in, map[bool]string{true: "int8", false: "fp16"}[*int8w],
		float64(e*o*in)*map[bool]float64{true: 1, false: 2}[*int8w]/1e6, k)

	// the table's values as the device sees them
	rng := rand.New(rand.NewSource(1))
	tbl := make([]float32, e*o*in)
	var data []byte
	var scales []uint16
	if *int8w {
		data = make([]byte, len(tbl))
		scales = make([]uint16, e*o)
		for r := 0; r < e*o; r++ {
			s := float32(1/(73*math.Sqrt(float64(in)))) * (0.5 + rng.Float32())
			scales[r] = mlpackage.Float16Bits(s)
			sv := make([]float32, 1)
			bonsaiai.DecodeF16(sv, []byte{byte(scales[r]), byte(scales[r] >> 8)})
			for c := 0; c < in; c++ {
				q := int8(rng.Intn(255) - 127)
				data[r*in+c] = byte(q)
				tbl[r*in+c] = float32(q) * sv[0]
			}
		}
	} else {
		for i := range tbl {
			tbl[i] = float32(rng.NormFloat64() / math.Sqrt(float64(in)))
		}
		raw := make([]byte, 2*len(tbl))
		bonsaiai.EncodeF16(raw, tbl)
		bonsaiai.DecodeF16(tbl, raw)
	}
	table := func(g *milpart.Graph, name string, n int, off int) *mlpackage.Var {
		if *int8w {
			w, err := g.B.Int8Scaled(name, []int{n, o, in}, []int{1, 1, in}, data[off*o*in:(off+n)*o*in], scales[off*o:(off+n)*o])
			if err != nil {
				panic(err)
			}
			return w
		}
		w, err := g.B.F16s(name, []int{n, o, in}, tbl[off*o*in:(off+n)*o*in])
		if err != nil {
			panic(err)
		}
		return w
	}
	mm := func(g *milpart.Graph, x, w *mlpackage.Var) *mlpackage.Var {
		return g.Op("matmul", map[string]any{"x": x, "y": w, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(true)}, k, 1, o)
	}
	gatherBody := func(g *milpart.Graph) {
		x := g.Input("x", k, 1, in)
		i := idx16(g, "i", k, 1)
		if *int8w && *raw8 {
			t, err := g.B.I8s("w", []int{e, o, in}, data)
			if err != nil {
				panic(err)
			}
			sc, err := g.B.F16("s", []int{e, o, 1}, scales)
			if err != nil {
				panic(err)
			}
			pick := func(i *mlpackage.Var, n int) *mlpackage.Var {
				w := g.B.Op("gather_nd", map[string]any{"x": t, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, mlpackage.Int8, n, o, in)
				s := g.Op("gather_nd", map[string]any{"x": sc, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, n, o, 1)
				wf := g.B.Op("cast", map[string]any{"x": w, "dtype": g.B.Str("c", "fp16")}, mlpackage.Float16, n, o, in)
				return g.Mul(wf, s)
			}
			if !*each {
				g.Output(mm(g, x, pick(i, k)))
				return
			}
			var ys []*mlpackage.Var
			for kk := 0; kk < k; kk++ {
				ys = append(ys, g.Op("matmul", map[string]any{"x": g.Slice(x, 0, kk, kk+1), "y": pick(g.Slice(i, 0, kk, kk+1), 1), "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(true)}, 1, 1, o))
			}
			if k == 1 {
				g.Output(ys[0])
			} else {
				g.Output(g.Concat(0, ys...))
			}
			return
		}
		t := table(g, "w", e, 0)
		if !*each {
			w := g.Op("gather_nd", map[string]any{"x": t, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, k, o, in)
			g.Output(mm(g, x, w))
			return
		}
		var ys []*mlpackage.Var
		for kk := 0; kk < k; kk++ {
			w := g.Op("gather_nd", map[string]any{"x": t, "indices": g.Slice(i, 0, kk, kk+1), "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, 1, o, in)
			ys = append(ys, g.Op("matmul", map[string]any{"x": g.Slice(x, 0, kk, kk+1), "y": w, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(true)}, 1, 1, o))
		}
		if k == 1 {
			g.Output(ys[0])
		} else {
			g.Output(g.Concat(0, ys...))
		}
	}
	denseBody := func(g *milpart.Graph) { // the first K experts, constant
		x := g.Input("x", k, 1, in)
		if !*int8w {
			g.Output(mm(g, x, table(g, "w", k, 0)))
			return
		}
		var ys []*mlpackage.Var // linear with a 2-D int8 weight, as the models have it
		for kk := 0; kk < k; kk++ {
			w, err := g.B.Int8Scaled(fmt.Sprintf("w%d", kk), []int{o, in}, []int{1, in}, data[kk*o*in:(kk+1)*o*in], scales[kk*o:(kk+1)*o])
			if err != nil {
				panic(err)
			}
			ys = append(ys, g.Op("linear", map[string]any{"x": g.Slice(x, 0, kk, kk+1), "weight": w}, 1, 1, o))
		}
		if k == 1 {
			g.Output(ys[0])
		} else {
			g.Output(g.Concat(0, ys...))
		}
	}
	compile := func(name string, body func(g *milpart.Graph)) (backend.Graph, error) {
		path := filepath.Join(dir, name+".mlpackage")
		if _, err := llm.WritePart(path, &milpart.Part{Name: name, Model: "gather", Body: body}, nil, llm.Options{Backend: llm.BackendCoreML, Format: "fp16"}, nil); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		t := time.Now()
		g, err := dev.Compile(ctx, types.GraphSource{Path: path})
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		p := g.Placement()
		fmt.Printf("  %-8s compiled in %.1fs, ane=%v %v\n", name, time.Since(t).Seconds(), p.OnNPU(), p.Ops)
		return g, nil
	}

	type prog struct {
		g      backend.Graph
		req    backend.Request
		x, i   backend.Buffer
		y      backend.Buffer
		hasIdx bool
	}
	open := func(g backend.Graph) (*prog, error) {
		p := &prog{g: g}
		p.req, _ = g.NewRequest()
		for _, a := range g.Args() {
			var b backend.Buffer
			var err error
			if ia, ok := g.(backend.InputAllocator); ok && a.Kind == types.ArgInput {
				b, err = ia.AllocInput(a.Index)
			}
			if b == nil && err == nil {
				b, err = dev.AllocBuffer(a.Bytes, types.BufferKind(0))
			}
			if err != nil {
				return nil, err
			}
			p.req.Bind(a.Index, b)
			switch a.Name {
			case "x":
				p.x = b
			case "i":
				p.i, p.hasIdx = b, true
			default:
				p.y = b
			}
		}
		return p, nil
	}
	call := func(p *prog) error {
		f, err := p.req.Submit()
		if err != nil {
			return err
		}
		return f.Wait(ctx)
	}

	gG, err := compile("gather", gatherBody)
	if err != nil {
		return err
	}
	defer gG.Close()
	gD, err := compile("dense", denseBody)
	if err != nil {
		return err
	}
	defer gD.Close()
	pg, err := open(gG)
	if err != nil {
		return err
	}
	pd, err := open(gD)
	if err != nil {
		return err
	}

	xv := make([]float32, k*in)
	for i := range xv {
		xv[i] = float32(rng.NormFloat64())
	}
	bonsaiai.EncodeF16(pg.x.Bytes(), xv)
	bonsaiai.DecodeF16(xv, pg.x.Bytes())
	copy(pd.x.Bytes(), pg.x.Bytes())
	setIdx := func(ix []int) {
		f := make([]float32, len(ix))
		for j, v := range ix {
			f[j] = float32(v)
		}
		bonsaiai.EncodeF16(pg.i.Bytes(), f)
	}
	ref := func(ix []int) []float32 {
		y := make([]float32, k*o)
		for kk, ex := range ix {
			for r := 0; r < o; r++ {
				var s float32
				row := tbl[(ex*o+r)*in : (ex*o+r+1)*in]
				for c, w := range row {
					s += w * xv[kk*in+c]
				}
				y[kk*o+r] = s
			}
		}
		return y
	}
	got := make([]float32, k*o)
	choices := [][]int{}
	first := make([]int, k)
	for j := range first {
		first[j] = j
	}
	choices = append(choices, first)
	last := make([]int, k)
	for j := range last {
		last[j] = e - 1 - j
	}
	choices = append(choices, last)
	for len(choices) < *checks {
		choices = append(choices, rng.Perm(e)[:k])
	}
	for _, ix := range choices {
		setIdx(ix)
		if err := call(pg); err != nil {
			return fmt.Errorf("gather: %w", err)
		}
		bonsaiai.DecodeF16(got, pg.y.Bytes())
		want := ref(ix)
		// which expert each output matches best, to tell a wrong pick from rounding
		picked := make([]int, k)
		for kk := range ix {
			best, bc := -1, -2.0
			for ex := 0; ex < e; ex++ {
				alt := ref(append(append([]int{}, ix[:kk]...), append([]int{ex}, ix[kk+1:]...)...))
				c := cosine(got[kk*o:(kk+1)*o], alt[kk*o:(kk+1)*o])
				if c > bc {
					best, bc = ex, c
				}
			}
			picked[kk] = best
		}
		fmt.Printf("  experts %v: cos %.6f rel %.4f, outputs match experts %v\n", ix, cosine(got, want), relErr(got, want), picked)
	}
	if err := call(pd); err != nil {
		return fmt.Errorf("dense: %w", err)
	}
	bonsaiai.DecodeF16(got, pd.y.Bytes())
	fmt.Printf("  dense against the host: cos %.6f\n", cosine(got, ref(first)))

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
	td, err := timeIt(func() error { return call(pd) })
	if err != nil {
		return err
	}
	setIdx(first)
	tsame, err := timeIt(func() error { return call(pg) })
	if err != nil {
		return err
	}
	trot, err := timeIt(func() error { setIdx(rng.Perm(e)[:k]); return call(pg) })
	if err != nil {
		return err
	}
	fmt.Printf("  a call: dense %.3f ms, gather same experts %.3f ms, gather random experts %.3f ms\n", ms(td), ms(tsame), ms(trot))
	return nil
}
