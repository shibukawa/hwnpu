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

// moePack is moe with fewer arrays to bind: an expert's weights in one int8 array (H, 3I), gate
// and up side by side with the down projection transposed, the scales applied to the products'
// columns. The scales are an fp16 array an expert (-scales expert) or one array a layer of all
// its experts', the chosen rows picked by gather_nd on the Neural Engine with the expert numbers
// an input (-scales gather). What binding costs as the arrays bound a call and held in all go
// down.
//
//	aicheck moepack [-model lfm2moe] [-layers 22] [-scales expert|gather] [-calls 200]
func moePack(argv []string) error {
	fs := flag.NewFlagSet("moepack", flag.ExitOnError)
	model := fs.String("model", "granite1b", "the layer's sizes (as moe)")
	layers := fs.Int("layers", 1, "layers' experts held")
	calls := fs.Int("calls", 200, "calls timed")
	scalesMode := fs.String("scales", "gather", "expert: an fp16 array of scales an expert; gather: a layer's in one array, picked on the device")
	topk := fs.Int("topk", 0, "experts a token takes (0: the model's)")
	two := fs.Bool("two", false, "gate|up and the down projection in two int8 arrays (no slices of the weights)")
	fs.Parse(argv)
	sh, ok := moeShapes[*model]
	if !ok {
		return fmt.Errorf("moepack: no model %s", *model)
	}
	gather := *scalesMode == "gather"
	ctx := context.Background()
	if err := llm.LockNPU(ctx); err != nil {
		return err
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "aicheck-moepack-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	H, I, E, K, T := sh.hidden, sh.inter, sh.experts, sh.topK, 1
	if *topk > 0 {
		K = *topk
	}
	S := 2*I + H // an expert's scales
	perArrays := 2
	if gather {
		perArrays = 1
	}
	if *two {
		perArrays++
	}
	fmt.Printf("%s packed: hidden %d, expert %d, %d experts, top %d; %d layers held %.2f GB in %d arrays, %d bound a call (scales: %s)\n",
		*model, H, I, E, K, *layers, float64(*layers*E*3*H*I)/1e9, *layers*E*perArrays+map[bool]int{true: *layers}[gather], K*perArrays+map[bool]int{true: 1}[gather], *scalesMode)

	body := func(g *milpart.Graph) {
		x := g.Input("x", 1, T, H)
		var sc *mlpackage.Var
		if gather {
			tbl := g.Input("S", E, S)
			i := idx16(g, "i", K, 1)
			sc = g.Op("gather_nd", map[string]any{"x": tbl, "indices": i, "batch_dims": g.B.ScalarI32(0), "validate_indices": g.B.ScalarBool(false)}, K, S)
		}
		var acc *mlpackage.Var
		for k := 0; k < K; k++ {
			cast := func(v *mlpackage.Var) *mlpackage.Var {
				return g.B.Op("cast", map[string]any{"x": v, "dtype": g.B.Str("c", "fp16")}, mlpackage.Float16, v.Shape...)
			}
			var s *mlpackage.Var
			if gather {
				s = g.Reshape(g.Slice(sc, 0, k, k+1), 1, 1, S)
			} else {
				s = g.Input(fmt.Sprintf("s%d", k), 1, 1, S)
			}
			var gu, dnT *mlpackage.Var
			if *two {
				gu = cast(g.InputI8(fmt.Sprintf("w%d", k), 1, H, 2*I))
				dnT = cast(g.InputI8(fmt.Sprintf("d%d", k), 1, H, I))
			} else {
				w := cast(g.InputI8(fmt.Sprintf("w%d", k), 1, H, 3*I))
				gu = g.Slice(w, 2, 0, 2*I)
				dnT = g.Slice(w, 2, 2*I, 3*I)
			}
			h := g.Mul(g.Op("matmul", map[string]any{"x": x, "y": gu, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(false)}, 1, T, 2*I), g.Slice(s, 2, 0, 2*I))
			a := g.Mul(g.SiLU(g.Slice(h, 2, 0, I)), g.Slice(h, 2, I, 2*I))
			d := g.Mul(g.Op("matmul", map[string]any{"x": a, "y": dnT, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(true)}, 1, T, H), g.Slice(s, 2, 2*I, S))
			y := g.Mul(d, g.Input(fmt.Sprintf("r%d", k), 1, T, 1))
			if acc == nil {
				acc = y
			} else {
				acc = g.Add(acc, y)
			}
		}
		g.Output(acc)
	}
	path := filepath.Join(dir, "packed.mlpackage")
	if _, err := llm.WritePart(path, &milpart.Part{Name: "packed", Model: "moe", Body: body}, nil, llm.Options{Backend: llm.BackendCoreML, Format: "fp16"}, nil); err != nil {
		return err
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
	rng := rand.New(rand.NewSource(1))
	type expert struct {
		w, s, d backend.Buffer
		wv      []int8
		sv      []float32
	}
	held := make([][]expert, *layers)
	tables := make([]backend.Buffer, *layers)
	tableV := make([]float32, E*S) // layer 0's
	t = time.Now()
	for l := range held {
		held[l] = make([]expert, E)
		var tb []float32
		if gather {
			if tables[l], err = alloc("S"); err != nil {
				return err
			}
			tb = make([]float32, E*S)
		}
		for e := range held[l] {
			var x expert
			if x.w, err = alloc("w0"); err != nil {
				return err
			}
			wb := make([]byte, H*3*I) // (H, 3I): gate|up, the down projection transposed
			for i := range wb {
				wb[i] = byte(rng.Intn(255) - 127)
			}
			if *two {
				if x.d, err = alloc("d0"); err != nil {
					return err
				}
				gb, db := x.w.Bytes(), x.d.Bytes()
				for h := 0; h < H; h++ {
					copy(gb[h*2*I:(h+1)*2*I], wb[h*3*I:h*3*I+2*I])
					copy(db[h*I:(h+1)*I], wb[h*3*I+2*I:(h+1)*3*I])
				}
			} else {
				copy(x.w.Bytes(), wb)
			}
			sv := make([]float32, S)
			sg, sd := float32(1/(73*math.Sqrt(float64(H)))), float32(1/(73*math.Sqrt(float64(I))))
			for j := range sv {
				if j < 2*I {
					sv[j] = sg * (0.5 + rng.Float32())
				} else {
					sv[j] = sd * (0.5 + rng.Float32())
				}
			}
			raw := make([]byte, 2*S)
			bonsaiai.EncodeF16(raw, sv)
			bonsaiai.DecodeF16(sv, raw)
			if gather {
				copy(tb[e*S:], sv)
			} else {
				if x.s, err = alloc("s0"); err != nil {
					return err
				}
				copy(x.s.Bytes(), raw)
			}
			if l == 0 && e < K {
				x.wv = make([]int8, len(wb))
				for i, v := range wb {
					x.wv[i] = int8(v)
				}
				x.sv = sv
			}
			held[l][e] = x
		}
		if gather {
			bonsaiai.EncodeF16(tables[l].Bytes(), tb)
			if l == 0 {
				copy(tableV, tb)
			}
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
	var ib backend.Buffer
	if gather {
		ib, _ = alloc("i")
		req.Bind(args["i"].Index, ib)
	}
	yb, _ := alloc("y")
	req.Bind(args["y"].Index, yb)
	bind := func(l int, chosen []int) {
		for k, e := range chosen {
			req.Bind(args[fmt.Sprintf("w%d", k)].Index, held[l][e].w)
			if *two {
				req.Bind(args[fmt.Sprintf("d%d", k)].Index, held[l][e].d)
			}
			if !gather {
				req.Bind(args[fmt.Sprintf("s%d", k)].Index, held[l][e].s)
			}
		}
		if gather {
			req.Bind(args["S"].Index, tables[l])
			f := make([]float32, K)
			for k, e := range chosen {
				f[k] = float32(e)
			}
			bonsaiai.EncodeF16(ib.Bytes(), f)
		}
	}
	run := func() error {
		f, err := req.Submit()
		if err != nil {
			return err
		}
		return f.Wait(ctx)
	}

	// correctness: layer 0's first K experts, in reverse so the gather is seen to pick
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
		sv := x.sv
		if gather {
			sv = tableV[e*S : (e+1)*S]
		}
		h := make([]float32, 2*I)
		for i := 0; i < H; i++ {
			row := x.wv[i*3*I : i*3*I+2*I]
			for j := range h {
				h[j] += xv[i] * float32(row[j])
			}
		}
		for j := range h {
			h[j] *= sv[j]
		}
		a := make([]float32, I)
		for j := range a {
			a[j] = h[j] / (1 + float32(math.Exp(float64(-h[j])))) * h[I+j]
		}
		for o := 0; o < H; o++ {
			row := x.wv[o*3*I+2*I : (o+1)*3*I]
			var s float32
			for j, v := range row {
				s += a[j] * float32(v)
			}
			want[o] += s * sv[2*I+o] / float32(K)
		}
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
