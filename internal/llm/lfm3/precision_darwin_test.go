package lfm3

import (
	"context"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Measures the Neural Engine's fp16 error on the pieces of a layer separately, against float32
// computed here from the same weights: the RMS norm alone, the in_proj linear alone, and both.
func TestANEPrecision(t *testing.T) {
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
	seq, d := len(ids), cfg.HiddenSize
	emb, err := Embed(w, d, ids)
	if err != nil {
		t.Fatal(err)
	}
	gamma, _, _ := w.Get("model.layers.0.operator_norm.weight")
	win, _, _ := w.Get("model.layers.0.conv.in_proj.weight")
	// reference pieces in float32, from fp16 rounded inputs like the device sees
	x := make([]float32, len(emb))
	for i, v := range emb {
		x[i] = mlpackage.Float16Value(mlpackage.Float16Bits(v))
	}
	normed := make([]float32, len(x))
	for s := 0; s < seq; s++ {
		var ss float64
		for j := 0; j < d; j++ {
			ss += float64(x[s*d+j]) * float64(x[s*d+j])
		}
		r := float32(1 / math.Sqrt(ss/float64(d)+cfg.NormEps))
		for j := 0; j < d; j++ {
			normed[s*d+j] = x[s*d+j] * r * gamma[j]
		}
	}
	lin := func(in []float32) []float32 {
		out := make([]float32, seq*3*d)
		for s := 0; s < seq; s++ {
			for o := 0; o < 3*d; o++ {
				var acc float64
				row := win[o*d : (o+1)*d]
				for j := 0; j < d; j++ {
					acc += float64(in[s*d+j]) * float64(row[j])
				}
				out[s*3*d+o] = float32(acc)
			}
		}
		return out
	}

	dev, err := coreml.Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	xb := make([]byte, 2*len(x))
	for i, v := range x {
		binary.LittleEndian.PutUint16(xb[2*i:], mlpackage.Float16Bits(v))
	}
	cases := []struct {
		name  string
		ref   []float32
		build func(m *builder, x *mlpackage.Var) *mlpackage.Var
	}{
		{"rmsnorm", normed, func(m *builder, x *mlpackage.Var) *mlpackage.Var {
			return m.rmsNorm(x, "model.layers.0.operator_norm.weight", d)
		}},
		{"in_proj linear fp16", lin(x), func(m *builder, x *mlpackage.Var) *mlpackage.Var {
			return m.linear(x, "model.layers.0.conv.in_proj.weight", "", 3*d)
		}},
		{"rmsnorm + in_proj fp16", lin(normed), func(m *builder, x *mlpackage.Var) *mlpackage.Var {
			return m.linear(m.rmsNorm(x, "model.layers.0.operator_norm.weight", d), "model.layers.0.conv.in_proj.weight", "", 3*d)
		}},
	}
	for _, c := range cases {
		out := filepath.Join(t.TempDir(), "p.mlpackage")
		pkg, err := mlpackage.Create(out)
		if err != nil {
			t.Fatal(err)
		}
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		m := &builder{cfg: cfg, w: w, opts: Options{Format: "fp16", Group: 16}, b: b, seq: seq, d: d}
		xv := b.Input("x", mlpackage.Float16, 1, seq, d)
		y := c.build(m, xv)
		y = b.NamedOp("identity", "y", map[string]any{"x": y}, mlpackage.Float16, y.Shape...)
		if err := pkg.Finish(b.Finish(y), 9, []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: xv.Shape}}, []mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}}); err != nil {
			t.Fatal(err)
		}
		g, err := dev.Compile(ctx, types.GraphSource{Path: out})
		if err != nil {
			t.Fatal(err)
		}
		p := g.Placement()
		got := run(t, dev, g, xb)
		g.Close()
		t.Logf("%s: npu ops %d, elsewhere %d", c.name, p.NPUOps, len(p.Ops))
		report("  "+c.name, got, c.ref, t)
	}
}
