//go:build darwin

package coreai

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// TestANEFP16ArithmeticTraps verifies how the Apple Neural Engine behaves on fp16 subnormal
// epsilons (< 2^-14 = 6.103515625e-5) for rsqrt, layer_norm, and clip, and confirms that
// clamping/guarding them at 2^-14 eliminates Inf/NaN on zero rows (e.g. padding rows).
func TestANEFP16ArithmeticTraps(t *testing.T) {
	infos, err := Provider{}.Enumerate(context.Background())
	if err != nil || len(infos) == 0 {
		t.Skip("no Core AI:", err)
	}
	t.Setenv("HWNPU_CACHE", t.TempDir())
	ctx := context.Background()
	dev, err := Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}

	dir := filepath.Join(t.TempDir(), "traps.mlpackage")
	pkg, err := mlpackage.Create(dir)
	if err != nil {
		t.Fatal(err)
	}
	b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
	f16 := mlpackage.Float16
	x := b.Input("x", f16, 1, 4, 1, 8)
	sq := b.Op("mul", map[string]any{"x": x, "y": x}, f16, 1, 4, 1, 8)
	ms := b.Op("reduce_mean", map[string]any{"x": sq, "axes": b.Ints(-1), "keep_dims": b.ScalarBool(true)}, f16, 1, 4, 1, 1)

	// Unprotected rsqrt(ms + 1e-6) reproduces the ANE fp16 subnormal +Inf bug on zero/tiny rows.
	rSub := b.NamedOp("rsqrt", "rsqrt_sub", map[string]any{"x": ms, "epsilon": b.ScalarF16(1e-6)}, f16, 1, 4, 1, 1)

	// MaxAbsRMSNorm(floor="add"): scale by max(|x|) + 2^-14 before squaring so non-zero rows
	// (even tiny 1e-3) have max(|t|) ~ 1 and ms(t) >> 2^-14, while zero rows stay finite 0.
	const fp16MinNormal = float32(1.0 / 16384.0) // 2^-14 = 0x0400
	absX := b.Op("abs", map[string]any{"x": x}, f16, 1, 4, 1, 8)
	maxAbs := b.Op("reduce_max", map[string]any{"x": absX, "axes": b.Ints(-1), "keep_dims": b.ScalarBool(true)}, f16, 1, 4, 1, 1)
	scale := b.Op("add", map[string]any{"x": maxAbs, "y": b.ScalarF16(fp16MinNormal)}, f16, 1, 4, 1, 1)
	tScaled := b.Op("real_div", map[string]any{"x": x, "y": scale}, f16, 1, 4, 1, 8)
	tSq := b.Op("mul", map[string]any{"x": tScaled, "y": tScaled}, f16, 1, 4, 1, 8)
	tMs := b.Op("reduce_mean", map[string]any{"x": tSq, "axes": b.Ints(-1), "keep_dims": b.ScalarBool(true)}, f16, 1, 4, 1, 1)
	tR := b.Op("rsqrt", map[string]any{"x": tMs, "epsilon": b.ScalarF16(fp16MinNormal)}, f16, 1, 4, 1, 1)
	yMaxAbs := b.NamedOp("mul", "maxabs_rmsnorm", map[string]any{"x": tScaled, "y": tR}, f16, 1, 4, 1, 8)

	prog := b.Finish(rSub, yMaxAbs)
	outs := []mlpackage.Feature{
		{Name: "rsqrt_sub", DType: f16, Shape: []int{1, 4, 1, 1}},
		{Name: "maxabs_rmsnorm", DType: f16, Shape: []int{1, 4, 1, 8}},
	}
	if err := pkg.Finish(prog, 9, []mlpackage.Feature{{Name: "x", DType: f16, Shape: []int{1, 4, 1, 8}}}, outs); err != nil {
		t.Fatal(err)
	}

	g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	req, err := g.NewRequest()
	if err != nil {
		t.Fatal(err)
	}
	bufs := map[string]backend.Buffer{}
	for _, a := range g.Args() {
		buf, err := dev.AllocBuffer(a.Bytes, 0)
		if err != nil {
			t.Fatal(err)
		}
		defer buf.Close()
		req.Bind(a.Index, buf)
		bufs[a.Name] = buf
		if a.Kind == types.ArgInput {
			// Row 0: all zeros (padding row)
			// Row 1: tiny 1e-3
			// Row 2: normal 1.0
			// Row 3: alternating -2, +2
			vals := make([]float32, 32)
			for i := 0; i < 8; i++ {
				vals[8+i] = 1e-3
				vals[16+i] = 1.0
				if i%2 == 0 {
					vals[24+i] = -2.0
				} else {
					vals[24+i] = 2.0
				}
			}
			for i, v := range vals {
				binary.LittleEndian.PutUint16(buf.Bytes()[2*i:], mlpackage.Float16Bits(v))
			}
		}
	}
	f, err := req.Submit()
	if err == nil {
		err = f.Wait(ctx)
	}
	if err != nil {
		t.Fatal(err)
	}
	readF16 := func(name string) []float32 {
		raw := bufs[name].Bytes()
		out := make([]float32, len(raw)/2)
		for i := range out {
			out[i] = mlpackage.Float16Value(binary.LittleEndian.Uint16(raw[2*i:]))
		}
		return out
	}
	t.Logf("onNPU=%v", g.Placement().OnNPU())
	if !g.Placement().OnNPU() && !OnCPU() {
		t.Errorf("traps graph not on NPU: %+v", g.Placement())
	}
	t.Logf("rsqrt_sub (unprotected) vals=%v", readF16("rsqrt_sub"))
	got := readF16("maxabs_rmsnorm")
	t.Logf("maxabs_rmsnorm row0=%v row1=%v row2=%v row3=%v", got[:2], got[8:10], got[16:18], got[24:26])
	for i, val := range got {
		if math.IsNaN(float64(val)) || math.IsInf(float64(val), 0) {
			t.Errorf("maxabs_rmsnorm[%d] = %v (want finite)", i, val)
		}
	}
	// Row 0 (zeros) -> 0, Row 1 (1e-3) -> ~1.0, Row 2 (1.0) -> 1.0, Row 3 (-2,+2) -> -1,+1
	if math.Abs(float64(got[0])) > 1e-4 {
		t.Errorf("row0 = %v, want 0", got[0])
	}
	if math.Abs(float64(got[8]-1.0)) > 0.02 {
		t.Errorf("row1 (tiny 1e-3) = %v, want ~1.0", got[8])
	}
	if math.Abs(float64(got[16]-1.0)) > 0.01 {
		t.Errorf("row2 (1.0) = %v, want 1.0", got[16])
	}
	if math.Abs(float64(got[24]+1.0)) > 0.01 || math.Abs(float64(got[25]-1.0)) > 0.01 {
		t.Errorf("row3 (-2,+2) = (%v, %v), want (-1, +1)", got[24], got[25])
	}
}

// TestANELayoutBC1S benchmarks ANE-native BC1S (1, C, 1, S) + 1x1 conv2d vs (1, S, C) + linear,
// and channel-axis softmax (axis=1) vs last-axis softmax (axis=-1) on real Apple Silicon ANE.
func TestANELayoutBC1S(t *testing.T) {
	infos, err := Provider{}.Enumerate(context.Background())
	if err != nil || len(infos) == 0 {
		t.Skip("no Core AI:", err)
	}
	t.Setenv("HWNPU_CACHE", t.TempDir())
	ctx := context.Background()
	dev, err := Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}

	benchPkg := func(build func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, []*mlpackage.Var)) (time.Duration, bool) {
		dir := filepath.Join(t.TempDir(), "m.mlpackage")
		pkg, err := mlpackage.Create(dir)
		if err != nil {
			t.Fatal(err)
		}
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		ins, outs, ret := build(b)
		prog := b.Finish(ret...)
		if err := pkg.Finish(prog, 9, ins, outs); err != nil {
			t.Fatal(err)
		}
		g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
		if err != nil {
			t.Fatal(err)
		}
		defer g.Close()
		req, err := g.NewRequest()
		if err != nil {
			t.Fatal(err)
		}
		for _, a := range g.Args() {
			buf, err := dev.AllocBuffer(a.Bytes, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer buf.Close()
			req.Bind(a.Index, buf)
		}
		for i := 0; i < 3; i++ {
			f, err := req.Submit()
			if err == nil {
				err = f.Wait(ctx)
			}
			if err != nil {
				t.Fatal(err)
			}
		}
		const iters = 15
		t0 := time.Now()
		for i := 0; i < iters; i++ {
			f, err := req.Submit()
			if err == nil {
				err = f.Wait(ctx)
			}
			if err != nil {
				t.Fatal(err)
			}
		}
		return time.Since(t0) / iters, g.Placement().OnNPU()
	}

	f16 := mlpackage.Float16
	rng := rand.New(rand.NewSource(99))

	// 1. Linear (1, S, Cin) -> (1, S, Cout) vs Conv1x1 BC1S (1, Cin, 1, S) -> (1, Cout, 1, S)
	for _, tc := range []struct {
		name         string
		S, Cin, Cout int
	}{
		{"decode_S8_2048x2048", 8, 2048, 2048},
		{"prefill_S128_2048x6144", 128, 2048, 6144},
	} {
		wt := make([]float32, tc.Cout*tc.Cin)
		for i := range wt {
			wt[i] = float32(rng.NormFloat64()) * 0.02
		}
		tLin, npuL := benchPkg(func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, []*mlpackage.Var) {
			x := b.Input("x", f16, 1, tc.S, tc.Cin)
			w, _ := b.F16s("w", []int{tc.Cout, tc.Cin}, wt)
			y := b.NamedOp("linear", "y", map[string]any{"x": x, "weight": w}, f16, 1, tc.S, tc.Cout)
			return []mlpackage.Feature{{Name: "x", DType: f16, Shape: []int{1, tc.S, tc.Cin}}},
				[]mlpackage.Feature{{Name: "y", DType: f16, Shape: []int{1, tc.S, tc.Cout}}},
				[]*mlpackage.Var{y}
		})
		tConv, npuC := benchPkg(func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, []*mlpackage.Var) {
			x := b.Input("x", f16, 1, tc.Cin, 1, tc.S)
			w, _ := b.F16s("w", []int{tc.Cout, tc.Cin, 1, 1}, wt)
			y := b.NamedOp("conv", "y", map[string]any{
				"x": x, "weight": w, "strides": b.Ints(1, 1), "pad_type": b.Str("c", "valid"),
				"dilations": b.Ints(1, 1), "groups": b.ScalarI32(1),
			}, f16, 1, tc.Cout, 1, tc.S)
			return []mlpackage.Feature{{Name: "x", DType: f16, Shape: []int{1, tc.Cin, 1, tc.S}}},
				[]mlpackage.Feature{{Name: "y", DType: f16, Shape: []int{1, tc.Cout, 1, tc.S}}},
				[]*mlpackage.Var{y}
		})
		t.Logf("%s: linear=%v (npu=%v) vs conv1x1_BC1S=%v (npu=%v) (%.2fx)",
			tc.name, tLin, npuL, tConv, npuC, float64(tLin)/float64(tConv))
	}

	// 2. Softmax: last-axis (1, 16, 64, 2048) axis=-1 vs channel-axis (1, 2048, 1, 1024) axis=1
	tLast, npuLast := benchPkg(func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, []*mlpackage.Var) {
		x := b.Input("x", f16, 1, 16, 64, 2048)
		y := b.NamedOp("softmax", "y", map[string]any{"x": x, "axis": b.ScalarI32(-1)}, f16, 1, 16, 64, 2048)
		return []mlpackage.Feature{{Name: "x", DType: f16, Shape: []int{1, 16, 64, 2048}}},
			[]mlpackage.Feature{{Name: "y", DType: f16, Shape: []int{1, 16, 64, 2048}}},
			[]*mlpackage.Var{y}
	})
	tChan, npuChan := benchPkg(func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, []*mlpackage.Var) {
		x := b.Input("x", f16, 1, 2048, 1, 1024)
		y := b.NamedOp("softmax", "y", map[string]any{"x": x, "axis": b.ScalarI32(1)}, f16, 1, 2048, 1, 1024)
		return []mlpackage.Feature{{Name: "x", DType: f16, Shape: []int{1, 2048, 1, 1024}}},
			[]mlpackage.Feature{{Name: "y", DType: f16, Shape: []int{1, 2048, 1, 1024}}},
			[]*mlpackage.Var{y}
	})
	t.Logf("softmax_2048x1024: last_axis(-1)=%v (npu=%v) vs channel_axis(1)=%v (npu=%v) (%.2fx)",
		tLast, npuLast, tChan, npuChan, float64(tLast)/float64(tChan))
}
