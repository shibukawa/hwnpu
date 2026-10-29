package lfm3

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Runs growing prefixes of conv layer 0 on the Neural Engine in fp16 and compares each with a
// float32 computation here, to find the operation that loses precision.
func TestConvPieces(t *testing.T) {
	dir := os.Getenv("HWNPU_LFM3_DIR")
	if dir == "" {
		t.Skip("HWNPU_LFM3_DIR not set")
	}
	ctx := context.Background()
	infos, err := coreml.Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	cfg, err := LoadConfig(filepath.Join(dir, "config.json"))
	if err != nil {
		t.Fatal(err)
	}
	w, err := OpenWeights(filepath.Join(dir, "LFM3-1.7B.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer w.Close()
	ids := readI32(t, filepath.Join(dir, "ref_bf16.ids.bin"))
	seq, d, k := len(ids), cfg.HiddenSize, cfg.ConvKernel
	emb, err := llm.Embed(w, cfg, ids)
	if err != nil {
		t.Fatal(err)
	}
	x := make([]float32, len(emb))
	for i, v := range emb {
		x[i] = mlpackage.Float16Value(mlpackage.Float16Bits(v))
	}
	get := func(name string) []float32 { v, _, _ := w.Get(name); return v }
	rms := func(in []float32, g []float32, n int) []float32 {
		out := make([]float32, len(in))
		for s := 0; s < len(in)/n; s++ {
			var ss float64
			for j := 0; j < n; j++ {
				ss += float64(in[s*n+j]) * float64(in[s*n+j])
			}
			r := float32(1 / math.Sqrt(ss/float64(n)+cfg.NormEps))
			for j := 0; j < n; j++ {
				out[s*n+j] = in[s*n+j] * r * g[j]
			}
		}
		return out
	}
	matmul := func(in []float32, wt []float32, inDim, outDim int) []float32 {
		out := make([]float32, seq*outDim)
		for s := 0; s < seq; s++ {
			for o := 0; o < outDim; o++ {
				var acc float64
				for j := 0; j < inDim; j++ {
					acc += float64(in[s*inDim+j]) * float64(wt[o*inDim+j])
				}
				out[s*outDim+o] = float32(acc)
			}
		}
		return out
	}
	p := "model.layers.0."
	h := rms(x, get(p+"operator_norm.weight"), d)
	bcx := matmul(h, get(p+"conv.in_proj.weight"), d, 3*d)
	bx := make([]float32, seq*d)
	cg := make([]float32, seq*d)
	for s := 0; s < seq; s++ {
		for c := 0; c < d; c++ {
			bx[s*d+c] = bcx[s*3*d+c] * bcx[s*3*d+2*d+c]
			cg[s*d+c] = bcx[s*3*d+d+c]
		}
	}
	cw := get(p + "conv.conv.weight")
	conv := make([]float32, seq*d)
	for s := 0; s < seq; s++ {
		for c := 0; c < d; c++ {
			var acc float32
			for j := 0; j < k; j++ {
				if ts := s + j - (k - 1); ts >= 0 {
					acc += cw[c*k+j] * bx[ts*d+c]
				}
			}
			conv[s*d+c] = acc
		}
	}
	gated := make([]float32, seq*d)
	for i := range gated {
		gated[i] = cg[i] * conv[i]
	}
	o := matmul(gated, get(p+"conv.out_proj.weight"), d, d)
	post := rms(o, get(p+"post_operator_norm.weight"), d)
	res := make([]float32, seq*d)
	for i := range res {
		res[i] = post[i] + x[i]
	}

	stats := func(name string, v []float32) {
		var ss float64
		var mx float32
		for _, x := range v {
			ss += float64(x) * float64(x)
			mx = max(mx, float32(math.Abs(float64(x))))
		}
		rms := math.Sqrt(ss / float64(len(v)))
		t.Logf("stats %-14s rms %.4f max %.3f ratio %.0f", name, rms, mx, float64(mx)/rms)
	}
	stats("x (embeddings)", x)
	stats("norm h", h)
	stats("in_proj bcx", bcx)
	stats("B*x", bx)
	stats("conv", conv)
	stats("C*conv", gated)
	stats("out_proj o", o)
	dev, err := coreml.Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	xb := make([]byte, 2*len(x))
	for i, v := range x {
		binary.LittleEndian.PutUint16(xb[2*i:], mlpackage.Float16Bits(v))
	}
	// stage selects how far the program goes
	stages := []struct {
		name string
		ref  []float32
	}{{"norm + in_proj", bcx}, {"B*x", bx}, {"conv", conv}, {"C*conv", gated}, {"out_proj", o}, {"post norm", post}, {"+ residual", res}}
	for si, st := range stages {
		out := filepath.Join(t.TempDir(), "c.mlpackage")
		pkg, err := mlpackage.Create(out)
		if err != nil {
			t.Fatal(err)
		}
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		m := newBuilder(b, cfg, w, Options{Seq: seq, Format: "fp16", Group: 16, ConvMode: "taps"})
		xv := b.Input("x", mlpackage.Float16, 1, seq, d)
		hv := m.rmsNorm(xv, p+"operator_norm.weight", d)
		bcxv := m.Linear(hv, p+"conv.in_proj.weight", "", 3*d)
		y := bcxv
		if si >= 1 {
			part := func(j int) *mlpackage.Var {
				return m.Op("slice_by_index", map[string]any{"x": bcxv, "begin": m.B.Ints(0, 0, int32(j*d)), "end": m.B.Ints(1, int32(seq), int32((j+1)*d)),
					"begin_mask": m.Bools(true, true, false), "end_mask": m.Bools(true, true, false)}, 1, seq, d)
			}
			bg, cgv, xg := part(0), part(1), part(2)
			y = m.Op("mul", map[string]any{"x": bg, "y": xg}, 1, seq, d)
			if si >= 2 {
				padded := m.Op("pad", map[string]any{"x": y, "pad": m.B.Ints(int32(k-1), 0, 0, 0), "mode": m.Str("constant"), "constant_val": m.B.ScalarF16(0)}, 1, seq+k-1, d)
				var acc *mlpackage.Var
				for j := 0; j < k; j++ {
					tap := make([]float32, d)
					for c := 0; c < d; c++ {
						tap[c] = cw[c*k+j]
					}
					wv, _ := m.B.F16s(fmt.Sprintf("tap%d", j), []int{1, 1, d}, tap)
					seg := m.Op("slice_by_index", map[string]any{"x": padded, "begin": m.B.Ints(0, int32(j), 0), "end": m.B.Ints(1, int32(j+seq), int32(d)),
						"begin_mask": m.Bools(true, false, true), "end_mask": m.Bools(true, false, true)}, 1, seq, d)
					term := m.Op("mul", map[string]any{"x": seg, "y": wv}, 1, seq, d)
					if acc == nil {
						acc = term
					} else {
						acc = m.Op("add", map[string]any{"x": acc, "y": term}, 1, seq, d)
					}
				}
				y = acc
				if si >= 3 {
					y = m.Op("mul", map[string]any{"x": cgv, "y": y}, 1, seq, d)
					if si >= 4 {
						y = m.Linear(y, p+"conv.out_proj.weight", "", d)
						if si >= 5 {
							y = m.rmsNorm(y, p+"post_operator_norm.weight", d)
							if si >= 6 {
								y = m.Op("add", map[string]any{"x": y, "y": xv}, 1, seq, d)
							}
						}
					}
				}
			}
		}
		y = b.NamedOp("identity", "y", map[string]any{"x": y}, mlpackage.Float16, y.Shape...)
		if err := pkg.Finish(b.Finish(y), 9, []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: xv.Shape}}, []mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}}); err != nil {
			t.Fatal(err)
		}
		g, err := dev.Compile(ctx, types.GraphSource{Path: out})
		if err != nil {
			t.Fatal(err)
		}
		pl := g.Placement()
		got := run(t, dev, g, xb)
		g.Close()
		report(fmt.Sprintf("%-14s (npu %d, elsewhere %d)", st.name, pl.NPUOps, len(pl.Ops)), got, st.ref, t)
	}
}
