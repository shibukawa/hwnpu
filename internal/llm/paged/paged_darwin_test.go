package paged_test

import (
	"context"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/llm/paged"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// On the Neural Engine, the attention over a chunk keeps the many lightly weighted keys when the
// values are small. The Neural Engine drops products under about 2^-16 inside a matrix product,
// and matmul(e, v) / l lost them (LFM3-1.7B's layer 18, |v| under 0.7: o off by up to 79 %);
// the weights e·(S/l) keep them (paged.Version). Scores spread as that layer's (largest about
// 4.5) with a few keys standing out, values of about 0.01, LFM3-1.7B's shape and chunk.
func TestSmallValuesOnNeuralEngine(t *testing.T) {
	ctx := context.Background()
	infos, err := coreai.Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Core AI:", err)
	}
	if coreai.OnCPU() {
		t.Skip("on Core AI's CPU")
	}
	t.Setenv("HWNPU_CACHE", t.TempDir())
	dev, err := coreai.Provider{}.Open(ctx, types.DeviceInfo{}, backend.Options{CacheDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	sh := paged.Shape{Heads: 16, KVHeads: 4, HeadDim: 128}
	const rows, chunk = 8, 8192
	for _, form := range []string{paged.Heads, "grouped"} {
		t.Run(form, func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "attn.mlpackage")
			if _, _, err := paged.Write(dir, sh, form, rows, chunk, "test"); err != nil {
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
			rng := rand.New(rand.NewPCG(5, 6))
			d, nh, nkv := sh.HeadDim, sh.Heads, sh.KVHeads
			vals := map[string][]float32{}
			vals["q"] = make([]float32, nh*rows*d)
			for i := range vals["q"] {
				vals["q"][i] = float32(rng.NormFloat64() * 1.2)
			}
			vals["k"] = make([]float32, nkv*chunk*d)
			for i := range vals["k"] {
				vals["k"][i] = float32(rng.NormFloat64())
			}
			for g := range nkv { // a few keys standing out for the queries of their kv head
				for _, p := range []int{100, 4000, 7900} {
					for i := range d {
						vals["k"][(g*chunk+p)*d+i] = vals["q"][(g*(nh/nkv)*rows)*d+i] * 0.4
					}
				}
			}
			vals["v"] = make([]float32, nkv*chunk*d)
			for i := range vals["v"] {
				vals["v"][i] = float32(rng.NormFloat64() * 0.01)
			}
			vals["mask"] = make([]float32, rows*chunk)
			bufs := map[string]backend.Buffer{}
			for _, a := range g.Args() {
				buf, err := dev.AllocBuffer(a.Bytes, 0)
				if err != nil {
					t.Fatal(err)
				}
				defer buf.Close()
				req.Bind(a.Index, buf)
				bufs[a.Name] = buf
				if v, ok := vals[a.Name]; ok {
					for i := range v {
						h := mlpackage.Float16Bits(v[i])
						binary.LittleEndian.PutUint16(buf.Bytes()[2*i:], h)
						v[i] = mlpackage.Float16Value(h) // the values the device sees
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
			if !g.Placement().OnNPU() {
				t.Fatal("not all on the Neural Engine")
			}
			q, k, v := vals["q"], vals["k"], vals["v"]
			scale := 1 / math.Sqrt(float64(d))
			var worst float64
			for h := range nh {
				kh := h / (nh / nkv)
				for r := range rows {
					sc := make([]float64, chunk)
					top := math.Inf(-1)
					for p := range chunk {
						var dot float64
						for i := range d {
							dot += float64(q[(h*rows+r)*d+i]) * float64(k[(kh*chunk+p)*d+i])
						}
						sc[p] = dot * scale
						top = max(top, sc[p])
					}
					var sum float64
					want := make([]float64, d)
					for p := range chunk {
						e := math.Exp(sc[p] - top)
						sum += e
						for i := range d {
							want[i] += e * float64(v[(kh*chunk+p)*d+i])
						}
					}
					var num, den float64
					for i := range d {
						got := float64(mlpackage.Float16Value(binary.LittleEndian.Uint16(bufs["o"].Bytes()[2*((h*rows+r)*d+i):])))
						num += (got - want[i]/sum) * (got - want[i]/sum)
						den += (want[i] / sum) * (want[i] / sum)
					}
					worst = max(worst, math.Sqrt(num/den))
				}
			}
			t.Logf("o off by up to %.4f", worst)
			if worst > 0.01 {
				t.Errorf("o off by up to %.3f of its size, want under 0.01", worst)
			}
		})
	}
}
