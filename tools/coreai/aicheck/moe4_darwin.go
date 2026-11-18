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

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// moe4 times experts in 4 bits on the Neural Engine: two signed codes an int8, 16 hi + (lo + 8),
// unpacked with floor_div. An expert's gate is lo and its up hi of one array (hidden, inter); its
// down projection's first half of rows lo and second half hi of (hidden / 2, inter). The scales
// are a row's (applied to the products) or a group of -group input columns' (applied to the
// weights), as fp16 arrays of the expert; lo's +8 comes off after the product (8 times x's sums
// over the groups times the scales). Against the host's float32 of the same codes, and timed as
// moepack: the same experts, chosen in one layer, chosen in any layer held.
//
//	aicheck moe4 [-model lfm2moe] [-layers 22] [-group 64] [-calls 200]
func moe4(argv []string) error {
	fs := flag.NewFlagSet("moe4", flag.ExitOnError)
	model := fs.String("model", "lfm2moe", "the layer's sizes (as moe)")
	layers := fs.Int("layers", 1, "layers' experts held")
	calls := fs.Int("calls", 200, "calls timed")
	group := fs.Int("group", 64, "input columns a scale (0: a row's, on the products)")
	castForm := fs.Bool("cast", false, "with -native -group 0: the int4 cast to fp16, the scales on the products, instead of blockwise_shift_scale")
	native := fs.Bool("native", false, "int4 inputs (Core AI's int4 arrays, two a byte) dequantized by blockwise_shift_scale, the scales on the weights")
	fs.Parse(argv)
	sh, ok := moeShapes[*model]
	if !ok {
		return fmt.Errorf("moe4: no model %s", *model)
	}
	H, I, E, K := sh.hidden, sh.inter, sh.experts, sh.topK
	G := *group
	if G > 0 && (H%G != 0 || I%G != 0) {
		return fmt.Errorf("moe4: group %d does not divide %d and %d", G, H, I)
	}
	ctx := context.Background()
	if err := llm.LockNPU(ctx); err != nil {
		return err
	}
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{})
	if err != nil {
		return err
	}
	dir, err := os.MkdirTemp("", "aicheck-moe4-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)
	// scale shapes: per row, gate|up (1, 1, 2I) and down (1, 1, H); per group, gate|up (1, H/G, 2I)
	// and down (1, H, I/G)
	sguShape, sdnShape := []int{1, 1, 2 * I}, []int{1, 1, H}
	if G > 0 {
		sguShape, sdnShape = []int{1, H / G, 2 * I}, []int{1, H, I / G}
	}
	n := func(s []int) int { return s[0] * s[1] * s[2] }
	fmt.Printf("%s in 4 bits (group %d): an expert %.1f MB packed + %.2f MB of scales; %d layers held %.2f GB\n", *model, G,
		float64(H*I+H/2*I)/1e6, float64(2*(n(sguShape)+n(sdnShape)))/1e6, *layers, float64(*layers*E*(H*I+H/2*I+2*(n(sguShape)+n(sdnShape))))/1e9)

	body := func(g *milpart.Graph) {
		f16 := func(typ string, in map[string]any, shape ...int) *mlpackage.Var {
			return g.B.Op(typ, in, mlpackage.Float16, shape...)
		}
		cast := func(v *mlpackage.Var) *mlpackage.Var {
			return f16("cast", map[string]any{"x": v, "dtype": g.B.Str("c", "fp16")}, v.Shape...)
		}
		mm := func(x, y *mlpackage.Var, ty bool, shape ...int) *mlpackage.Var {
			return g.Op("matmul", map[string]any{"x": x, "y": y, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(ty)}, shape...)
		}
		// unpack: v = 16 hi + lo8, hi in -8..7, lo8 in 0..15
		unpack := func(v *mlpackage.Var) (lo8, hi *mlpackage.Var) {
			hi = f16("floor_div", map[string]any{"x": v, "y": g.B.ScalarF16(16)}, v.Shape...)
			lo8 = g.Op("sub", map[string]any{"x": v, "y": g.Mul(hi, g.B.ScalarF16(16))}, v.Shape...)
			return lo8, hi
		}
		// group sums of a row (1, 1, n) over groups of G: (1, 1, n/G)
		gsum := func(x *mlpackage.Var, n int) *mlpackage.Var {
			r := g.Reshape(x, 1, n/G, G)
			s := f16("reduce_sum", map[string]any{"x": r, "axes": g.B.Ints(2), "keep_dims": g.B.ScalarBool(false)}, 1, n/G)
			return g.Reshape(s, 1, 1, n/G)
		}
		// w (1, rows, cols) times scales (1, rows/G, cols) a group of rows (scales along the rows)
		scaleRows := func(w, s *mlpackage.Var, rows, cols int) *mlpackage.Var {
			r := g.Reshape(w, rows/G, G, cols)
			return g.Reshape(g.Mul(r, g.Reshape(s, rows/G, 1, cols)), 1, rows, cols)
		}
		// w (1, rows, cols) times scales (1, rows, cols/G) a group of columns
		scaleCols := func(w, s *mlpackage.Var, rows, cols int) *mlpackage.Var {
			r := g.Reshape(w, rows, cols/G, G)
			return g.Reshape(g.Mul(r, g.Reshape(s, rows, cols/G, 1)), 1, rows, cols)
		}
		x := g.Input("x", 1, 1, H)
		var xg *mlpackage.Var
		if G > 0 {
			xg = gsum(x, H)
		}
		var xs *mlpackage.Var // the row's sum, for the per-row form's +8
		if G == 0 {
			xs = f16("reduce_sum", map[string]any{"x": x, "axes": g.B.Ints(2), "keep_dims": g.B.ScalarBool(true)}, 1, 1, 1)
		}
		var acc *mlpackage.Var
		for k := 0; k < K; k++ {
			gu := cast(g.InputI8(fmt.Sprintf("gu%d", k), 1, H, I))
			dn := cast(g.InputI8(fmt.Sprintf("dn%d", k), 1, H/2, I))
			sgu := g.Input(fmt.Sprintf("sgu%d", k), sguShape...)
			sdn := g.Input(fmt.Sprintf("sdn%d", k), sdnShape...)
			gl, gh := unpack(gu) // gate (lo, +8), up (hi)
			dl, dh := unpack(dn) // down's first rows (lo, +8), its second rows (hi)
			var gate, up *mlpackage.Var
			if G > 0 {
				sg, su := g.Slice(sgu, 2, 0, I), g.Slice(sgu, 2, I, 2*I)
				gate = g.Op("sub", map[string]any{"x": mm(x, scaleRows(gl, sg, H, I), false, 1, 1, I), "y": g.Mul(mm(xg, sg, false, 1, 1, I), g.B.ScalarF16(8))}, 1, 1, I)
				up = mm(x, scaleRows(gh, su, H, I), false, 1, 1, I)
			} else {
				sg, su := g.Slice(sgu, 2, 0, I), g.Slice(sgu, 2, I, 2*I)
				gate = g.Mul(g.Op("sub", map[string]any{"x": mm(x, gl, false, 1, 1, I), "y": g.Mul(xs, g.B.ScalarF16(8))}, 1, 1, I), sg)
				up = g.Mul(mm(x, gh, false, 1, 1, I), su)
			}
			a := g.Mul(g.SiLU(gate), up)
			var d1, d2 *mlpackage.Var
			if G > 0 {
				ag := gsum(a, I)
				s1, s2 := g.Slice(sdn, 1, 0, H/2), g.Slice(sdn, 1, H/2, H)
				d1 = g.Op("sub", map[string]any{"x": mm(a, scaleCols(dl, s1, H/2, I), true, 1, 1, H/2), "y": g.Mul(mm(ag, s1, true, 1, 1, H/2), g.B.ScalarF16(8))}, 1, 1, H/2)
				d2 = mm(a, scaleCols(dh, s2, H/2, I), true, 1, 1, H/2)
			} else {
				as := f16("reduce_sum", map[string]any{"x": a, "axes": g.B.Ints(2), "keep_dims": g.B.ScalarBool(true)}, 1, 1, 1)
				d1 = g.Mul(g.Op("sub", map[string]any{"x": mm(a, dl, true, 1, 1, H/2), "y": g.Mul(as, g.B.ScalarF16(8))}, 1, 1, H/2), g.Slice(sdn, 2, 0, H/2))
				d2 = g.Mul(mm(a, dh, true, 1, 1, H/2), g.Slice(sdn, 2, H/2, H))
			}
			y := g.Mul(g.Concat(2, d1, d2), g.Input(fmt.Sprintf("r%d", k), 1, 1, 1))
			if acc == nil {
				acc = y
			} else {
				acc = g.Add(acc, y)
			}
		}
		g.Output(acc)
	}
	if *native {
		body = func(g *milpart.Graph) {
			mm := func(x, y *mlpackage.Var, ty bool, shape ...int) *mlpackage.Var {
				return g.Op("matmul", map[string]any{"x": x, "y": y, "transpose_x": g.B.ScalarBool(false), "transpose_y": g.B.ScalarBool(ty)}, shape...)
			}
			deq := func(data, scale *mlpackage.Var) *mlpackage.Var {
				return g.B.Op("constexpr_blockwise_shift_scale", map[string]any{"data": data, "scale": scale}, mlpackage.Float16, data.Shape...)
			}
			x := g.Input("x", 1, 1, H)
			var acc *mlpackage.Var
			for k := 0; k < K; k++ {
				// gate|up (1, H, 2I): the input rows' groups (scales (1, H/G, 2I)); down (1, H, I),
				// the output rows by their input columns' groups (scales (1, H, I/G))
				var h, d *mlpackage.Var
				if *castForm {
					c := func(v *mlpackage.Var) *mlpackage.Var {
						return g.B.Op("cast", map[string]any{"x": v, "dtype": g.B.Str("c", "fp16")}, mlpackage.Float16, v.Shape...)
					}
					sgu := g.Input(fmt.Sprintf("sgu%d", k), 1, 1, 2*I)
					sdn := g.Reshape(g.Input(fmt.Sprintf("sdn%d", k), 1, H, 1), 1, 1, H)
					h = g.Mul(mm(x, c(g.InputI4(fmt.Sprintf("gu%d", k), 1, H, 2*I)), false, 1, 1, 2*I), sgu)
					a := g.Mul(g.SiLU(g.Slice(h, 2, 0, I)), g.Slice(h, 2, I, 2*I))
					d = g.Mul(mm(a, c(g.InputI4(fmt.Sprintf("dn%d", k), 1, H, I)), true, 1, 1, H), sdn)
				} else {
					gu := deq(g.InputI4(fmt.Sprintf("gu%d", k), 1, H, 2*I), g.Input(fmt.Sprintf("sgu%d", k), sguShape...))
					dn := deq(g.InputI4(fmt.Sprintf("dn%d", k), 1, H, I), g.Input(fmt.Sprintf("sdn%d", k), sdnShape...))
					h = mm(x, gu, false, 1, 1, 2*I)
					a := g.Mul(g.SiLU(g.Slice(h, 2, 0, I)), g.Slice(h, 2, I, 2*I))
					d = mm(a, dn, true, 1, 1, H)
				}
				y := g.Mul(d, g.Input(fmt.Sprintf("r%d", k), 1, 1, 1))
				if acc == nil {
					acc = y
				} else {
					acc = g.Add(acc, y)
				}
			}
			g.Output(acc)
		}
		if G == 0 { // a row's: gate|up's output columns, down's output rows
			sguShape, sdnShape = []int{1, 1, 2 * I}, []int{1, H, 1}
		}
	}
	path := filepath.Join(dir, "experts4.mlpackage")
	if *native {
		path = filepath.Join(dir, "experts4.aimodel")
	}
	be := llm.BackendCoreML
	if *native {
		be = llm.BackendCoreAI
	}
	if _, err := llm.WritePart(path, &milpart.Part{Name: "experts4", Model: "moe", Body: body}, nil, llm.Options{Backend: be, Format: "fp16"}, nil); err != nil {
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
		gu, dn, sgu, sdn backend.Buffer
		gate, up, down   []float32 // the values (layer 0's first K)
	}
	pack := func(lo, hi int) byte { return byte(int8(16*hi + lo + 8)) }
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
			keep := l == 0 && e < K
			sg := make([]float32, n(sguShape))
			for i := range sg {
				sg[i] = float32(1/(7*math.Sqrt(float64(H)))) * (0.5 + rng.Float32())
			}
			sd := make([]float32, n(sdnShape))
			for i := range sd {
				sd[i] = float32(1/(7*math.Sqrt(float64(I)))) * (0.5 + rng.Float32())
			}
			hwnpu.EncodeF16(x.sgu.Bytes(), sg)
			hwnpu.DecodeF16(sg, x.sgu.Bytes())
			hwnpu.EncodeF16(x.sdn.Bytes(), sd)
			hwnpu.DecodeF16(sd, x.sdn.Bytes())
			gb, db := x.gu.Bytes(), x.dn.Bytes()
			if *native {
				nib := func(buf []byte, i, c int) { // element i of a packed int4 array
					v := byte(c) & 0xF
					if i%2 == 0 {
						buf[i/2] = buf[i/2]&0xF0 | v
					} else {
						buf[i/2] = buf[i/2]&0x0F | v<<4
					}
				}
				if keep {
					x.gate, x.up, x.down = make([]float32, H*I), make([]float32, H*I), make([]float32, H*I)
				}
				for h := 0; h < H; h++ {
					for j := 0; j < 2*I; j++ {
						c := rng.Intn(16) - 8
						nib(gb, h*2*I+j, c)
						if keep {
							s := sg[j]
							if G > 0 {
								s = sg[(h/G)*2*I+j]
							}
							if j < I {
								x.gate[h*I+j] = float32(c) * s
							} else {
								x.up[h*I+j-I] = float32(c) * s
							}
						}
					}
				}
				for o := 0; o < H; o++ {
					for j := 0; j < I; j++ {
						c := rng.Intn(16) - 8
						nib(db, o*I+j, c)
						if keep {
							s := sd[o]
							if G > 0 {
								s = sd[o*(I/G)+j/G]
							}
							x.down[o*I+j] = float32(c) * s
						}
					}
				}
				held[l][e] = x
				continue
			}
			if keep {
				x.gate, x.up, x.down = make([]float32, H*I), make([]float32, H*I), make([]float32, H*I)
			}
			for h := 0; h < H; h++ {
				for j := 0; j < I; j++ {
					lo, hi := rng.Intn(16)-8, rng.Intn(16)-8
					gb[h*I+j] = pack(lo, hi)
					if keep { // gate (h, j) is the weight from input h to output j
						s1, s2 := sg[j], sg[I+j]
						if G > 0 {
							s1, s2 = sg[(h/G)*2*I+j], sg[(h/G)*2*I+I+j]
						}
						x.gate[h*I+j], x.up[h*I+j] = float32(lo)*s1, float32(hi)*s2
					}
				}
			}
			for r := 0; r < H/2; r++ {
				for j := 0; j < I; j++ {
					lo, hi := rng.Intn(16)-8, rng.Intn(16)-8
					db[r*I+j] = pack(lo, hi)
					if keep { // down (output row, input j)
						s1, s2 := sd[r], sd[H/2+r]
						if G > 0 {
							s1, s2 = sd[r*(I/G)+j/G], sd[(H/2+r)*(I/G)+j/G]
						}
						x.down[r*I+j], x.down[(H/2+r)*I+j] = float32(lo)*s1, float32(hi)*s2
					}
				}
			}
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
	hwnpu.EncodeF16(xb.Bytes(), xv)
	hwnpu.DecodeF16(xv, xb.Bytes())
	req.Bind(args["x"].Index, xb)
	for k := 0; k < K; k++ {
		r, _ := alloc(fmt.Sprintf("r%d", k))
		hwnpu.EncodeF16(r.Bytes(), []float32{1 / float32(K)})
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
	hwnpu.DecodeF16(got, yb.Bytes())
	want := make([]float32, H)
	for _, e := range chosen {
		x := held[0][e]
		gt, up := make([]float32, I), make([]float32, I)
		for h := 0; h < H; h++ {
			for j := 0; j < I; j++ {
				gt[j] += xv[h] * x.gate[h*I+j]
				up[j] += xv[h] * x.up[h*I+j]
			}
		}
		a := make([]float32, I)
		for j := range a {
			a[j] = gt[j] / (1 + float32(math.Exp(float64(-gt[j])))) * up[j]
		}
		for o := 0; o < H; o++ {
			var s float32
			for j := 0; j < I; j++ {
				s += a[j] * x.down[o*I+j]
			}
			want[o] += s / float32(K)
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
