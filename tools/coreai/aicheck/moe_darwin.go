package main

import (
	"context"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/backend"
	"github.com/shibukawa/bonsaiai/internal/backend/coreai"
	"github.com/shibukawa/bonsaiai/internal/imagegen/milpart"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/mlpackage"
	"github.com/shibukawa/bonsaiai/internal/types"
)

// moeShape is a mixture of experts layer's sizes: hidden, an expert's intermediate width, the
// experts and how many a token takes.
type moeShape struct{ hidden, inter, experts, topK int }

var moeShapes = map[string]moeShape{
	"gptoss":    {2880, 2880, 32, 4}, // gpt-oss-20b
	"qwen3moe":  {2048, 768, 128, 8}, // Qwen3-30B-A3B
	"small":     {1024, 512, 16, 4},  // for a quick try
	"lfm2moe":   {2048, 1792, 32, 4}, // LFM2-8B-A1B (22 of its 24 layers)
	"olmoe":     {2048, 1024, 64, 8}, // OLMoE-1B-7B (16 layers)
	"granite3b": {1536, 512, 40, 8},  // Granite 3.1 3B-A800M (32 layers)
	"granite1b": {1024, 512, 32, 8},  // Granite 3.1 1B-A400M (24 layers)
}

// moe measures a mixture of experts layer on the Neural Engine with the experts' weights passed
// at run time: each expert's weights in arrays of their own (Core AI's input arrays, no copy at a
// call), the chosen ones bound to one program per layer shape. Against the same product with
// constant weights (what a dense model of the active size costs), the router's own call and
// the host's top k.
//
//	aicheck moe [-model gptoss|qwen3moe|small] [-rows 1] [-layers 2] [-calls 200] [-fp16]
func moe(args []string) error {
	fs := flag.NewFlagSet("moe", flag.ExitOnError)
	model := fs.String("model", "gptoss", "the layer's sizes: gptoss, qwen3moe, small")
	rows := fs.Int("rows", 1, "tokens a call (1: decoding)")
	layers := fs.Int("layers", 2, "layers' experts held, the calls going round them (more than the caches)")
	calls := fs.Int("calls", 200, "calls timed")
	fp16 := fs.Bool("fp16", false, "also the experts' weights at run time in fp16 (twice the memory)")
	copyMode := fs.Bool("copy", false, "also: K fixed arrays bound once, the chosen experts copied into them before each call")
	split := fs.Int("split", 0, "experts a program takes (0: all of a token's); a layer then takes several calls")
	topk := fs.Int("topk", 0, "experts a token takes (0: the model's)")
	scaleOut := fs.Bool("scale-out", false, "scale the products' columns instead of the weights (int8: a cast alone on the weights)")
	fs.Parse(args)
	sh, ok := moeShapes[*model]
	if !ok {
		return fmt.Errorf("moe: no model %s", *model)
	}
	ctx := context.Background()
	if err := llm.LockNPU(ctx); err != nil {
		return err
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "aicheck-moe-*")
	if err != nil {
		return err
	}
	if os.Getenv("KEEP") != "" {
		fmt.Println("packages in", dir)
	} else {
		defer os.RemoveAll(dir)
	}
	H, I, E, K, T := sh.hidden, sh.inter, sh.experts, sh.topK, *rows
	if *topk > 0 {
		K = *topk
	}
	Ktot := K // the experts a token takes; K is a program's
	if *split > 0 && *split < K {
		K = *split
	}
	nsub := (Ktot + K - 1) / K
	perExpert := H*2*I + I*H
	fmt.Printf("%s: hidden %d, expert %d, %d experts, top %d, %d row(s); an expert %.1f MB in int8, a token's experts %.1f MB, %d layers held %.2f GB\n",
		*model, H, I, E, Ktot, T, float64(perExpert)/1e6, float64(Ktot*perExpert)/1e6, *layers, float64(*layers*E*perExpert)/1e9)

	compile := func(name string, body func(g *milpart.Graph)) (backend.Graph, error) {
		path := filepath.Join(dir, name+".mlpackage")
		if _, err := llm.WritePart(path, &milpart.Part{Name: name, Model: "moe", Body: body}, nil, llm.Options{Backend: llm.BackendCoreML, Format: "fp16"}, nil); err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		t := time.Now()
		g, err := dev.Compile(ctx, types.GraphSource{Path: path})
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		fmt.Printf("  %-12s compiled in %.1fs, ane=%v\n", name, time.Since(t).Seconds(), g.Placement().OnNPU())
		return g, nil
	}
	f16 := func(g *milpart.Graph, x *mlpackage.Var) *mlpackage.Var {
		return g.B.Op("cast", map[string]any{"x": x, "dtype": g.B.Str("c", "fp16")}, mlpackage.Float16, x.Shape...)
	}
	mm := func(g *milpart.Graph, x, w *mlpackage.Var, shape ...int) *mlpackage.Var {
		return g.Op("matmul", map[string]any{"x": x, "y": w, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, shape...)
	}
	dynScaled := false // building the int8 program with its scales as inputs
	// the experts' SwiGLU: gate and up in one product, the down projection, weighted and summed
	expertsBody := func(weights func(g *milpart.Graph, k int) (gu, dn *mlpackage.Var)) func(g *milpart.Graph) {
		return func(g *milpart.Graph) {
			x := g.Input("x", 1, T, H)
			var acc *mlpackage.Var
			for k := 0; k < K; k++ {
				gu, dn := weights(g, k)
				h := mm(g, x, gu, 1, T, 2*I)
				if *scaleOut && dynScaled {
					h = g.Mul(h, g.Input(fmt.Sprintf("sgu%d", k), 1, 1, 2*I))
				}
				a := g.Mul(g.SiLU(g.Slice(h, 2, 0, I)), g.Slice(h, 2, I, 2*I))
				d := mm(g, a, dn, 1, T, H)
				if *scaleOut && dynScaled {
					d = g.Mul(d, g.Input(fmt.Sprintf("sdn%d", k), 1, 1, H))
				}
				y := g.Mul(d, g.Input(fmt.Sprintf("r%d", k), 1, T, 1))
				if acc == nil {
					acc = y
				} else {
					acc = g.Add(acc, y)
				}
			}
			g.Output(acc)
		}
	}
	dynI8body := expertsBody(func(g *milpart.Graph, k int) (*mlpackage.Var, *mlpackage.Var) {
		if *scaleOut {
			return f16(g, g.InputI8(fmt.Sprintf("gu%d", k), 1, H, 2*I)), f16(g, g.InputI8(fmt.Sprintf("dn%d", k), 1, I, H))
		}
		gu := g.Mul(f16(g, g.InputI8(fmt.Sprintf("gu%d", k), 1, H, 2*I)), g.Input(fmt.Sprintf("sgu%d", k), 1, 1, 2*I))
		dn := g.Mul(f16(g, g.InputI8(fmt.Sprintf("dn%d", k), 1, I, H)), g.Input(fmt.Sprintf("sdn%d", k), 1, 1, H))
		return gu, dn
	})
	dynI8 := func(g *milpart.Graph) { dynScaled = true; dynI8body(g); dynScaled = false }
	dynF16 := expertsBody(func(g *milpart.Graph, k int) (*mlpackage.Var, *mlpackage.Var) {
		return g.Input(fmt.Sprintf("gu%d", k), 1, H, 2*I), g.Input(fmt.Sprintf("dn%d", k), 1, I, H)
	})

	// the constant reference is written with linear (the weight (out, in) as the models have it)
	constBody := func(g *milpart.Graph) {
		x := g.Input("x", 1, T, H)
		lin := func(x *mlpackage.Var, name string, out, in int) *mlpackage.Var {
			data := make([]byte, out*in)
			for i := range data {
				data[i] = byte(rand.Intn(255) - 127)
			}
			scales := make([]uint16, out)
			for i := range scales {
				scales[i] = mlpackage.Float16Bits(float32(1 / (73 * math.Sqrt(float64(in)))))
			}
			w, err := g.B.Int8Scaled(name, []int{out, in}, []int{1, in}, data, scales)
			if err != nil {
				panic(err)
			}
			return g.Op("linear", map[string]any{"x": x, "weight": w}, 1, T, out)
		}
		var acc *mlpackage.Var
		for k := 0; k < Ktot; k++ {
			h := lin(x, fmt.Sprintf("gu%d", k), 2*I, H)
			a := g.Mul(g.SiLU(g.Slice(h, 2, 0, I)), g.Slice(h, 2, I, 2*I))
			y := g.Mul(lin(a, fmt.Sprintf("dn%d", k), H, I), g.Input(fmt.Sprintf("r%d", k), 1, T, 1))
			if acc == nil {
				acc = y
			} else {
				acc = g.Add(acc, y)
			}
		}
		g.Output(acc)
	}
	routerBody := func(g *milpart.Graph) {
		x := g.Input("x", 1, T, H)
		g.Output(g.LinearConst(x, "wr", randW(E*H, H), E, nil))
	}
	tinyBody := func(g *milpart.Graph) {
		x := g.Input("x", 1, T, 64)
		g.Output(g.AddScalar(x, 1))
	}

	gDyn, err := compile("experts_i8", dynI8)
	if err != nil {
		return err
	}
	gConst, err := compile("experts_const", constBody)
	if err != nil {
		return err
	}
	gRouter, err := compile("router", routerBody)
	if err != nil {
		return err
	}
	gTiny, err := compile("tiny", tinyBody)
	if err != nil {
		return err
	}

	// arguments by name
	argIndex := func(g backend.Graph) map[string]types.ArgInfo {
		m := map[string]types.ArgInfo{}
		for _, a := range g.Args() {
			m[a.Name] = a
		}
		return m
	}
	alloc := func(g backend.Graph, name string) (backend.Buffer, error) {
		a, ok := argIndex(g)[name]
		if !ok {
			return nil, fmt.Errorf("no argument %s", name)
		}
		if a.Kind == types.ArgInput {
			if ia, ok := g.(backend.InputAllocator); ok {
				b, err := ia.AllocInput(a.Index)
				if err != nil {
					return nil, err
				}
				if b != nil {
					return b, nil
				}
			}
		}
		return dev.AllocBuffer(a.Bytes, types.BufferKind(0))
	}
	rng := rand.New(rand.NewSource(1))
	fillF16 := func(b backend.Buffer, n int, gen func() float32) []float32 {
		f := make([]float32, n)
		for i := range f {
			f[i] = gen()
		}
		bonsaiai.EncodeF16(b.Bytes(), f)
		bonsaiai.DecodeF16(f, b.Bytes()) // as the device sees them
		return f
	}

	// the experts' arrays: int8 weights and fp16 scales a column, for every layer held
	type expert struct {
		gu, dn, sgu, sdn backend.Buffer
		guv, dnv         []int8
		sguv, sdnv       []float32
	}
	held := make([][]expert, *layers)
	t := time.Now()
	for l := range held {
		held[l] = make([]expert, E)
		for e := range held[l] {
			var x expert
			if x.gu, err = alloc(gDyn, "gu0"); err != nil {
				return err
			}
			if x.dn, err = alloc(gDyn, "dn0"); err != nil {
				return err
			}
			if x.sgu, err = alloc(gDyn, "sgu0"); err != nil {
				return err
			}
			if x.sdn, err = alloc(gDyn, "sdn0"); err != nil {
				return err
			}
			gb, db := x.gu.Bytes(), x.dn.Bytes()
			for i := range gb {
				gb[i] = byte(rng.Intn(255) - 127)
			}
			for i := range db {
				db[i] = byte(rng.Intn(255) - 127)
			}
			if l == 0 && e < 2*K { // kept for the reference check
				x.guv = make([]int8, len(gb))
				for i, v := range gb {
					x.guv[i] = int8(v)
				}
				x.dnv = make([]int8, len(db))
				for i, v := range db {
					x.dnv[i] = int8(v)
				}
			}
			sg, sd := float32(1/(73*math.Sqrt(float64(H)))), float32(1/(73*math.Sqrt(float64(I))))
			x.sguv = fillF16(x.sgu, 2*I, func() float32 { return sg * (0.5 + rng.Float32()) })
			x.sdnv = fillF16(x.sdn, H, func() float32 { return sd * (0.5 + rng.Float32()) })
			held[l][e] = x
		}
	}
	fmt.Printf("  experts' arrays made and filled in %.1fs\n", time.Since(t).Seconds())

	// the program's own inputs and output
	dynArgs := argIndex(gDyn)
	reqDyn, _ := gDyn.NewRequest()
	xb, err := alloc(gDyn, "x")
	if err != nil {
		return err
	}
	xv := fillF16(xb, T*H, func() float32 { return float32(rng.NormFloat64()) })
	reqDyn.Bind(dynArgs["x"].Index, xb)
	rbs := make([]backend.Buffer, K)
	for k := range rbs {
		if rbs[k], err = alloc(gDyn, fmt.Sprintf("r%d", k)); err != nil {
			return err
		}
		reqDyn.Bind(dynArgs[fmt.Sprintf("r%d", k)].Index, rbs[k])
	}
	yb, err := alloc(gDyn, "y")
	if err != nil {
		return err
	}
	reqDyn.Bind(dynArgs["y"].Index, yb)
	bindExperts := func(l int, chosen []int, weights []float32) {
		for k, e := range chosen {
			x := held[l][e]
			reqDyn.Bind(dynArgs[fmt.Sprintf("gu%d", k)].Index, x.gu)
			reqDyn.Bind(dynArgs[fmt.Sprintf("dn%d", k)].Index, x.dn)
			reqDyn.Bind(dynArgs[fmt.Sprintf("sgu%d", k)].Index, x.sgu)
			reqDyn.Bind(dynArgs[fmt.Sprintf("sdn%d", k)].Index, x.sdn)
			bonsaiai.EncodeF16(rbs[k].Bytes(), repeat(weights[k], T))
		}
	}
	runReq := func(r backend.Request) error {
		f, err := r.Submit()
		if err != nil {
			return err
		}
		return f.Wait(ctx)
	}
	choose := func() ([]int, []float32) {
		p := rng.Perm(E)[:K]
		w := make([]float32, K)
		for i := range w {
			w[i] = 1 / float32(K)
		}
		return p, w
	}

	// 1. correctness: one call against the host's float32 of the same int8 weights
	chosen := []int{0, 1, 2, 3, 4, 5, 6, 7}[:K]
	weights := make([]float32, K)
	for i := range weights {
		weights[i] = 1 / float32(K)
	}
	bindExperts(0, chosen, weights)
	if err := runReq(reqDyn); err != nil {
		return fmt.Errorf("experts_i8: %w", err)
	}
	got := make([]float32, T*H)
	bonsaiai.DecodeF16(got, yb.Bytes())
	want := make([]float32, T*H)
	for k, e := range chosen {
		x := held[0][e]
		for r := 0; r < T; r++ {
			h := make([]float32, 2*I)
			for i := 0; i < H; i++ {
				xi := xv[r*H+i]
				row := x.guv[i*2*I : (i+1)*2*I]
				for j := range h {
					h[j] += xi * float32(row[j])
				}
			}
			for j := range h {
				h[j] *= x.sguv[j]
			}
			a := make([]float32, I)
			for j := range a {
				g := h[j]
				a[j] = g / (1 + float32(math.Exp(float64(-g)))) * h[I+j]
			}
			y := make([]float32, H)
			for j := 0; j < I; j++ {
				row := x.dnv[j*H : (j+1)*H]
				for o := range y {
					y[o] += a[j] * float32(row[o])
				}
			}
			for o := range y {
				want[r*H+o] += weights[k] * y[o] * x.sdnv[o]
			}
		}
	}
	fmt.Printf("  experts_i8 against the host's float32: cos %.6f rel %.4f\n", cosine(got, want), relErr(got, want))

	// 2. times
	timeIt := func(n int, step func() error) (time.Duration, error) {
		for i := 0; i < 5; i++ {
			if err := step(); err != nil {
				return 0, err
			}
		}
		t := time.Now()
		for i := 0; i < n; i++ {
			if err := step(); err != nil {
				return 0, err
			}
		}
		return time.Since(t) / time.Duration(n), nil
	}
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1e3 }

	fixed, err := timeIt(*calls, func() error {
		for i := 0; i < nsub; i++ {
			if err := runReq(reqDyn); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	var bindTime time.Duration
	rot, err := timeIt(*calls, func() error {
		l := rng.Intn(*layers)
		for i := 0; i < nsub; i++ {
			p, w := choose()
			t := time.Now()
			bindExperts(l, p, w)
			bindTime += time.Since(t)
			if err := runReq(reqDyn); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		return err
	}
	bindTime /= time.Duration(*calls + 5)
	// how many distinct experts the calls may go round before binding costs: K of the first n
	// experts of layer 0, in any slot
	type poolTime struct {
		n int
		t time.Duration
	}
	var pools []poolTime
	for n := Ktot; ; n *= 2 {
		if n > E {
			n = E
		}
		d, err := timeIt(*calls, func() error {
			for i := 0; i < nsub; i++ {
				p := rng.Perm(n)[:K]
				w := make([]float32, K)
				for i := range w {
					w[i] = 1 / float32(K)
				}
				bindExperts(0, p, w)
				if err := runReq(reqDyn); err != nil {
					return err
				}
			}
			return nil
		})
		if err != nil {
			return err
		}
		pools = append(pools, poolTime{n, d})
		if n == E {
			break
		}
	}

	var copyRun, copyTime time.Duration
	copyCos := math.NaN()
	if *copyMode {
		slots := make([]expert, K)
		for k := range slots {
			var err error
			for _, b := range []struct {
				dst  *backend.Buffer
				name string
			}{{&slots[k].gu, "gu"}, {&slots[k].dn, "dn"}, {&slots[k].sgu, "sgu"}, {&slots[k].sdn, "sdn"}} {
				if *b.dst, err = alloc(gDyn, b.name+"0"); err != nil {
					return err
				}
				reqDyn.Bind(dynArgs[fmt.Sprintf("%s%d", b.name, k)].Index, *b.dst)
			}
		}
		par := runtime.NumCPU()
		pcopy := func(dst, src []byte) {
			var wg sync.WaitGroup
			n := (len(src) + par - 1) / par
			for lo := 0; lo < len(src); lo += n {
				hi := min(lo+n, len(src))
				wg.Add(1)
				go func() { defer wg.Done(); copy(dst[lo:hi], src[lo:hi]) }()
			}
			wg.Wait()
		}
		fill := func(l int, chosen []int, weights []float32) {
			for k, e := range chosen {
				x := held[l][e]
				pcopy(slots[k].gu.Bytes(), x.gu.Bytes())
				pcopy(slots[k].dn.Bytes(), x.dn.Bytes())
				copy(slots[k].sgu.Bytes(), x.sgu.Bytes())
				copy(slots[k].sdn.Bytes(), x.sdn.Bytes())
				bonsaiai.EncodeF16(rbs[k].Bytes(), repeat(weights[k], T))
			}
		}
		// the copied experts give the reference's output (the arrays are read at each call)
		fill(0, chosen, weights)
		if err := runReq(reqDyn); err != nil {
			return err
		}
		got := make([]float32, T*H)
		bonsaiai.DecodeF16(got, yb.Bytes())
		copyCos = cosine(got, want)
		copyRun, err = timeIt(*calls, func() error {
			l := rng.Intn(*layers)
			for i := 0; i < nsub; i++ {
				p, w := choose()
				t := time.Now()
				fill(l, p, w)
				copyTime += time.Since(t)
				if err := runReq(reqDyn); err != nil {
					return err
				}
			}
			return nil
		})
		if err != nil {
			return err
		}
		copyTime /= time.Duration(*calls + 5)
	}
	simple := func(g backend.Graph) (backend.Request, error) {
		r, _ := g.NewRequest()
		for _, a := range g.Args() {
			b, err := alloc(g, a.Name)
			if err != nil {
				return nil, err
			}
			if a.Kind == types.ArgInput {
				n := a.Bytes / 2
				fillF16(b, n, func() float32 { return float32(rng.NormFloat64() * 0.5) })
			}
			r.Bind(a.Index, b)
		}
		return r, nil
	}
	reqConst, err := simple(gConst)
	if err != nil {
		return err
	}
	cst, err := timeIt(*calls, func() error { return runReq(reqConst) })
	if err != nil {
		return err
	}
	reqRouter, err := simple(gRouter)
	if err != nil {
		return err
	}
	var outBuf backend.Buffer
	for _, a := range gRouter.Args() {
		if a.Kind == types.ArgOutput {
			outBuf, _ = dev.AllocBuffer(a.Bytes, types.BufferKind(0))
			reqRouter.Bind(a.Index, outBuf)
		}
	}
	logits := make([]float32, T*E)
	var topkTime time.Duration
	router, err := timeIt(*calls, func() error {
		if err := runReq(reqRouter); err != nil {
			return err
		}
		t := time.Now()
		bonsaiai.DecodeF16(logits, outBuf.Bytes())
		topK(logits[:E], Ktot)
		topkTime += time.Since(t)
		return nil
	})
	if err != nil {
		return err
	}
	topkTime /= time.Duration(*calls + 5)
	reqTiny, err := simple(gTiny)
	if err != nil {
		return err
	}
	tiny, err := timeIt(*calls, func() error { return runReq(reqTiny) })
	if err != nil {
		return err
	}

	fmt.Printf("\n  a call of a tiny program            %7.3f ms\n", ms(tiny))
	fmt.Printf("  experts, constant int8 (reference)  %7.3f ms\n", ms(cst))
	if nsub > 1 {
		fmt.Printf("  (%d experts a program: %d calls a layer)\n", K, nsub)
	}
	fmt.Printf("  experts, int8 at run time, the same %7.3f ms\n", ms(fixed))
	fmt.Printf("  experts, int8 at run time, chosen   %7.3f ms (binding %.3f ms of it)\n", ms(rot), ms(bindTime))
	for _, p := range pools {
		fmt.Printf("    going round %4d experts (%7.1f MB)  %7.3f ms\n", p.n, float64(p.n*perExpert)/1e6, ms(p.t))
	}
	fmt.Printf("  router (constant)                   %7.3f ms, top %d on the host %.3f ms\n", ms(router), Ktot, ms(topkTime))
	if *copyMode {
		fmt.Printf("  experts copied into fixed arrays    %7.3f ms (the copy %.3f ms of it, %d threads; cos against the reference %.6f)\n", ms(copyRun), ms(copyTime), runtime.NumCPU(), copyCos)
		fmt.Printf("  a layer with copies                 %7.3f ms against %.3f ms (+%.0f%%)\n", ms(copyRun+router+topkTime), ms(cst), 100*(float64(copyRun+router+topkTime)/float64(cst)-1))
	}
	layer := rot + router + topkTime
	fmt.Printf("  a layer: router + top k + experts   %7.3f ms against %.3f ms (+%.0f%%)\n", ms(layer), ms(cst), 100*(float64(layer)/float64(cst)-1))
	gb := float64(Ktot*perExpert) / 1e9
	fmt.Printf("  the experts' weights read at %.1f GB/s (run time) and %.1f GB/s (constant)\n", gb/rot.Seconds(), gb/cst.Seconds())

	if *fp16 {
		gF, err := compile("experts_f16", dynF16)
		if err != nil {
			return err
		}
		fa := argIndex(gF)
		req, _ := gF.NewRequest()
		for _, a := range gF.Args() {
			b, err := alloc(gF, a.Name)
			if err != nil {
				return err
			}
			if a.Kind == types.ArgInput {
				fillF16(b, a.Bytes/2, func() float32 { return float32(rng.NormFloat64() * 0.02) })
			}
			req.Bind(a.Index, b)
		}
		// held experts in fp16 (one layer, to save memory)
		type ef struct{ gu, dn backend.Buffer }
		hf := make([]ef, E)
		for e := range hf {
			if hf[e].gu, err = alloc(gF, "gu0"); err != nil {
				return err
			}
			if hf[e].dn, err = alloc(gF, "dn0"); err != nil {
				return err
			}
			fillF16(hf[e].gu, len(hf[e].gu.Bytes())/2, func() float32 { return float32(rng.NormFloat64() * 0.02) })
			fillF16(hf[e].dn, len(hf[e].dn.Bytes())/2, func() float32 { return float32(rng.NormFloat64() * 0.02) })
		}
		rf, err := timeIt(*calls, func() error {
			p, _ := choose()
			for k, e := range p {
				req.Bind(fa[fmt.Sprintf("gu%d", k)].Index, hf[e].gu)
				req.Bind(fa[fmt.Sprintf("dn%d", k)].Index, hf[e].dn)
			}
			return runReq(req)
		})
		if err != nil {
			return err
		}
		fmt.Printf("  experts, fp16 at run time, chosen   %7.3f ms (%.1f GB/s)\n", ms(rf), 2*gb/rf.Seconds())
	}
	return nil
}

func repeat(v float32, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = v
	}
	return out
}

// topK is the indices of the k largest values.
func topK(v []float32, k int) []int {
	idx := make([]int, len(v))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return v[idx[a]] > v[idx[b]] })
	return idx[:k]
}

func cosine(a, b []float32) float64 {
	var ab, aa, bb float64
	for i := range a {
		ab += float64(a[i]) * float64(b[i])
		aa += float64(a[i]) * float64(a[i])
		bb += float64(b[i]) * float64(b[i])
	}
	return ab / math.Sqrt(aa*bb)
}

func relErr(a, b []float32) float64 {
	var d, n float64
	for i := range a {
		d += (float64(a[i]) - float64(b[i])) * (float64(a[i]) - float64(b[i]))
		n += float64(b[i]) * float64(b[i])
	}
	return math.Sqrt(d / n)
}

// locked runs a command holding bonsaiai's Neural Engine lock (for tools of other languages that
// load programs, such as tools/coreai/moebind).
func locked(args []string) error {
	if len(args) == 0 {
		return fmt.Errorf("locked: no command")
	}
	if err := llm.LockNPU(context.Background()); err != nil {
		return err
	}
	c := exec.Command(args[0], args[1:]...)
	c.Stdout, c.Stderr = os.Stdout, os.Stderr
	return c.Run()
}
