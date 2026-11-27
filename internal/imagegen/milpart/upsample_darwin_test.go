//go:build darwin

package milpart_test

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand"
	"path/filepath"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/imagegen/milpart"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// TestUpsampleConv2DFusion verifies that FuseNearest2xConv3x3 + conv_transpose(k=4, s=2, p=1)
// matches upsample_nearest_neighbor(2x) + conv3x3(pad=1) on both milcpu and Core AI ANE, and
// measures the real ANE speedup at VAE upsampler dimensions.
func TestUpsampleConv2DFusion(t *testing.T) {
	infos, err := coreai.Provider{}.Enumerate(context.Background())
	if err != nil || len(infos) == 0 {
		t.Skip("no Core AI:", err)
	}
	t.Setenv("HWNPU_CACHE", t.TempDir())
	ctx := context.Background()
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}

	// 1. Exactness check on small shape (1, 16, 15, 17) -> (1, 12, 30, 34) including odd dims
	// and boundaries, comparing unfused vs fused on both milcpu and Core AI ANE.
	{
		const cin, cout, H, W = 16, 12, 15, 17
		rng := rand.New(rand.NewSource(42))
		wt := make([]float32, cout*cin*9)
		for i := range wt {
			wt[i] = float32(rng.NormFloat64()) * 0.2
		}
		bias := make([]float32, cout)
		for i := range bias {
			bias[i] = float32(rng.NormFloat64()) * 0.1
		}
		fused := milpart.FuseNearest2xConv3x3(wt, cout, cin)

		dir := filepath.Join(t.TempDir(), "up.mlpackage")
		pkg, err := mlpackage.Create(dir)
		if err != nil {
			t.Fatal(err)
		}
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		f16 := mlpackage.Float16
		x := b.Input("x", f16, 1, cin, H, W)
		u := b.Op("upsample_nearest_neighbor", map[string]any{
			"x": x, "scale_factor_height": b.ScalarI32(2), "scale_factor_width": b.ScalarI32(2),
		}, f16, 1, cin, 2*H, 2*W)
		wUnfused, err := b.F16s("w_unfused", []int{cout, cin, 3, 3}, wt)
		if err != nil {
			t.Fatal(err)
		}
		bConst, err := b.F16s("bias", []int{cout}, bias)
		if err != nil {
			t.Fatal(err)
		}
		yUnfused := b.NamedOp("conv", "unfused", map[string]any{
			"x": u, "weight": wUnfused, "bias": bConst,
			"strides": b.Ints(1, 1), "pad_type": b.Str("c", "custom"), "pad": b.Ints(1, 1, 1, 1),
			"dilations": b.Ints(1, 1), "groups": b.ScalarI32(1),
		}, f16, 1, cout, 2*H, 2*W)

		wFused, err := b.F16s("w_fused", []int{cin, cout, 4, 4}, fused)
		if err != nil {
			t.Fatal(err)
		}
		yFused := b.NamedOp("conv_transpose", "fused", map[string]any{
			"x": x, "weight": wFused, "bias": bConst,
			"strides": b.Ints(2, 2), "pad_type": b.Str("c", "custom"), "pad": b.Ints(1, 1, 1, 1),
			"dilations": b.Ints(1, 1), "groups": b.ScalarI32(1),
		}, f16, 1, cout, 2*H, 2*W)

		prog := b.Finish(yUnfused, yFused)
		outs := []mlpackage.Feature{
			{Name: "unfused", DType: f16, Shape: []int{1, cout, 2 * H, 2 * W}},
			{Name: "fused", DType: f16, Shape: []int{1, cout, 2 * H, 2 * W}},
		}
		if err := pkg.Finish(prog, 9, []mlpackage.Feature{{Name: "x", DType: f16, Shape: []int{1, cin, H, W}}}, outs); err != nil {
			t.Fatal(err)
		}

		// CPU reference
		p, c, err := milcpu.OpenPackage(dir)
		if err != nil {
			t.Fatal(err)
		}
		inVals := make([]float32, cin*H*W)
		for i := range inVals {
			inVals[i] = mlpackage.Float16Value(mlpackage.Float16Bits(float32(rng.NormFloat64())))
		}
		cpuOut, err := p.Run(map[string]milcpu.Tensor{"x": {Shape: []int{1, cin, H, W}, Data: inVals}}, milcpu.Options{RoundF16: true})
		c.Close()
		if err != nil {
			t.Fatal(err)
		}
		var cpuDiff float64
		for i, a := range cpuOut["unfused"].Data {
			cpuDiff = max(cpuDiff, math.Abs(float64(a-cpuOut["fused"].Data[i])))
		}
		t.Logf("milcpu unfused vs fused max diff: %g", cpuDiff)
		if cpuDiff > 0.01 {
			t.Errorf("milcpu unfused vs fused max diff %g > 0.01", cpuDiff)
		}

		// Core AI ANE
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
				for i, v := range inVals {
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
		if !g.Placement().OnNPU() && !coreai.OnCPU() {
			t.Errorf("upsample fusion graph not all on NPU: %+v", g.Placement())
		}
		var aneDiff, aneVsCPU float64
		for i := range cpuOut["unfused"].Data {
			uVal := mlpackage.Float16Value(binary.LittleEndian.Uint16(bufs["unfused"].Bytes()[2*i:]))
			fVal := mlpackage.Float16Value(binary.LittleEndian.Uint16(bufs["fused"].Bytes()[2*i:]))
			aneDiff = max(aneDiff, math.Abs(float64(uVal-fVal)))
			aneVsCPU = max(aneVsCPU, math.Abs(float64(fVal-cpuOut["unfused"].Data[i])))
		}
		t.Logf("ANE unfused vs fused max diff: %g, ANE fused vs milcpu unfused: %g, onNPU=%v", aneDiff, aneVsCPU, g.Placement().OnNPU())
		if aneDiff > 0.01 || aneVsCPU > 0.01 {
			t.Errorf("ANE diff too large: unfused vs fused=%g, vs cpu=%g", aneDiff, aneVsCPU)
		}
	}

	// 2. Real ANE latency comparison at VAE upsampler sizes:
	//    FLUX  Stage 3 (1024x1024): 256 -> 256, 256x256 -> 512x512
	//    Anima Stage 5 (1024x1024): 192 -> 96, 512x512 -> 1024x1024
	type benchCase struct {
		name            string
		cin, cout, H, W int
	}
	for _, tc := range []benchCase{
		{"FLUX_256x256_to_512x512_c256", 256, 256, 256, 256},
		{"Anima_512x512_to_1024x1024_c192to96", 192, 96, 512, 512},
	} {
		rng := rand.New(rand.NewSource(7))
		wt := make([]float32, tc.cout*tc.cin*9)
		for i := range wt {
			wt[i] = float32(rng.NormFloat64()) * 0.05
		}
		bias := make([]float32, tc.cout)
		fused := milpart.FuseNearest2xConv3x3(wt, tc.cout, tc.cin)

		runOne := func(useFused bool) (time.Duration, bool) {
			dir := filepath.Join(t.TempDir(), "m.mlpackage")
			pkg, err := mlpackage.Create(dir)
			if err != nil {
				t.Fatal(err)
			}
			b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
			f16 := mlpackage.Float16
			x := b.Input("x", f16, 1, tc.cin, tc.H, tc.W)
			bConst, err := b.F16s("bias", []int{tc.cout}, bias)
			if err != nil {
				t.Fatal(err)
			}
			var y *mlpackage.Var
			if !useFused {
				u := b.Op("upsample_nearest_neighbor", map[string]any{
					"x": x, "scale_factor_height": b.ScalarI32(2), "scale_factor_width": b.ScalarI32(2),
				}, f16, 1, tc.cin, 2*tc.H, 2*tc.W)
				w, err := b.F16s("w", []int{tc.cout, tc.cin, 3, 3}, wt)
				if err != nil {
					t.Fatal(err)
				}
				y = b.NamedOp("conv", "y", map[string]any{
					"x": u, "weight": w, "bias": bConst,
					"strides": b.Ints(1, 1), "pad_type": b.Str("c", "custom"), "pad": b.Ints(1, 1, 1, 1),
					"dilations": b.Ints(1, 1), "groups": b.ScalarI32(1),
				}, f16, 1, tc.cout, 2*tc.H, 2*tc.W)
			} else {
				w, err := b.F16s("w", []int{tc.cin, tc.cout, 4, 4}, fused)
				if err != nil {
					t.Fatal(err)
				}
				y = b.NamedOp("conv_transpose", "y", map[string]any{
					"x": x, "weight": w, "bias": bConst,
					"strides": b.Ints(2, 2), "pad_type": b.Str("c", "custom"), "pad": b.Ints(1, 1, 1, 1),
					"dilations": b.Ints(1, 1), "groups": b.ScalarI32(1),
				}, f16, 1, tc.cout, 2*tc.H, 2*tc.W)
			}
			prog := b.Finish(y)
			if err := pkg.Finish(prog, 9,
				[]mlpackage.Feature{{Name: "x", DType: f16, Shape: []int{1, tc.cin, tc.H, tc.W}}},
				[]mlpackage.Feature{{Name: "y", DType: f16, Shape: []int{1, tc.cout, 2 * tc.H, 2 * tc.W}}},
			); err != nil {
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
			for i := 0; i < 2; i++ {
				f, err := req.Submit()
				if err == nil {
					err = f.Wait(ctx)
				}
				if err != nil {
					t.Fatal(err)
				}
			}
			const iters = 5
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

		tUnfused, npuU := runOne(false)
		tFused, npuF := runOne(true)
		t.Logf("%s: unfused=%v (npu=%v) -> fused=%v (npu=%v) (%.2fx)",
			tc.name, tUnfused, npuU, tFused, npuF, float64(tUnfused)/float64(tFused))
	}
}
