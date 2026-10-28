package mlpackage

import (
	"context"
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Reports where Core ML plans to run a few program shapes, to learn which forms the Neural
// Engine accepts. It only logs; run with -v.
func TestPlacementVariants(t *testing.T) {
	ctx := context.Background()
	infos, err := coreml.Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	dev, err := coreml.Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	rng := rand.New(rand.NewSource(1))
	lut4 := func(b *Builder, name string, rows, cols int) *Var {
		l := make([]uint16, rows/16*16)
		for i := range l {
			l[i] = Float16Bits(float32(rng.NormFloat64()) * 0.02)
		}
		p := make([]byte, rows*cols/2)
		rng.Read(p)
		v, err := b.LUT4(name, []int{rows, cols}, []int{16, cols}, p, l)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	zeros := func(b *Builder, n int) *Var {
		v, _ := b.F16s("bias", []int{n}, make([]float32, n))
		return v
	}
	int4 := func(b *Builder, name string, rows, cols, blockCols int) *Var {
		data := make([]byte, rows*cols/2)
		rng.Read(data)
		sc := make([]uint16, rows*cols/blockCols)
		for i := range sc {
			sc[i] = Float16Bits(0.002)
		}
		v, err := b.Int4Scaled(name, []int{rows, cols}, []int{1, blockCols}, data, sc)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	type variant struct {
		name  string
		build func(b *Builder) (in []Feature, out *Var)
	}
	variants := []variant{
		{"linear 128->96 no bias", func(b *Builder) ([]Feature, *Var) {
			x := b.Input("x", Float16, 1, 8, 128)
			return []Feature{{"x", Float16, x.Shape}}, b.NamedOp("linear", "y", map[string]any{"x": x, "weight": lut4(b, "w", 96, 128)}, Float16, 1, 8, 96)
		}},
		{"linear 128->96 with bias", func(b *Builder) ([]Feature, *Var) {
			x := b.Input("x", Float16, 1, 8, 128)
			return []Feature{{"x", Float16, x.Shape}}, b.NamedOp("linear", "y", map[string]any{"x": x, "weight": lut4(b, "w", 96, 128), "bias": zeros(b, 96)}, Float16, 1, 8, 96)
		}},
		{"linear 2048->2048 seq 1", func(b *Builder) ([]Feature, *Var) {
			x := b.Input("x", Float16, 1, 1, 2048)
			return []Feature{{"x", Float16, x.Shape}}, b.NamedOp("linear", "y", map[string]any{"x": x, "weight": lut4(b, "w", 2048, 2048)}, Float16, 1, 1, 2048)
		}},
		{"linear 2048->2048 seq 32", func(b *Builder) ([]Feature, *Var) {
			x := b.Input("x", Float16, 1, 32, 2048)
			return []Feature{{"x", Float16, x.Shape}}, b.NamedOp("linear", "y", map[string]any{"x": x, "weight": lut4(b, "w", 2048, 2048)}, Float16, 1, 32, 2048)
		}},
		{"two linears 2048 seq 32", func(b *Builder) ([]Feature, *Var) {
			x := b.Input("x", Float16, 1, 32, 2048)
			h := b.Op("linear", map[string]any{"x": x, "weight": lut4(b, "w1", 2048, 2048)}, Float16, 1, 32, 2048)
			h = b.Op("silu", map[string]any{"x": h}, Float16, 1, 32, 2048)
			return []Feature{{"x", Float16, x.Shape}}, b.NamedOp("linear", "y", map[string]any{"x": h, "weight": lut4(b, "w2", 2048, 2048)}, Float16, 1, 32, 2048)
		}},
		{"int4 block 32 (Q4_0 layout) seq 32", func(b *Builder) ([]Feature, *Var) {
			x := b.Input("x", Float16, 1, 32, 2048)
			h := b.Op("linear", map[string]any{"x": x, "weight": int4(b, "w1", 2048, 2048, 32)}, Float16, 1, 32, 2048)
			h = b.Op("silu", map[string]any{"x": h}, Float16, 1, 32, 2048)
			return []Feature{{"x", Float16, x.Shape}}, b.NamedOp("linear", "y", map[string]any{"x": h, "weight": int4(b, "w2", 2048, 2048, 32)}, Float16, 1, 32, 2048)
		}},
		{"int4 per row scale seq 32", func(b *Builder) ([]Feature, *Var) {
			x := b.Input("x", Float16, 1, 32, 2048)
			h := b.Op("linear", map[string]any{"x": x, "weight": int4(b, "w1", 2048, 2048, 2048)}, Float16, 1, 32, 2048)
			h = b.Op("silu", map[string]any{"x": h}, Float16, 1, 32, 2048)
			return []Feature{{"x", Float16, x.Shape}}, b.NamedOp("linear", "y", map[string]any{"x": h, "weight": int4(b, "w2", 2048, 2048, 2048)}, Float16, 1, 32, 2048)
		}},
		{"int4 block 128 seq 32", func(b *Builder) ([]Feature, *Var) {
			x := b.Input("x", Float16, 1, 32, 2048)
			h := b.Op("linear", map[string]any{"x": x, "weight": int4(b, "w1", 2048, 2048, 128)}, Float16, 1, 32, 2048)
			h = b.Op("silu", map[string]any{"x": h}, Float16, 1, 32, 2048)
			return []Feature{{"x", Float16, x.Shape}}, b.NamedOp("linear", "y", map[string]any{"x": h, "weight": int4(b, "w2", 2048, 2048, 128)}, Float16, 1, 32, 2048)
		}},
		{"gather 128000x2048 lut4", func(b *Builder) ([]Feature, *Var) {
			ids := b.Input("ids", Int32, 1, 32)
			return []Feature{{"ids", Int32, ids.Shape}}, b.NamedOp("gather", "y", map[string]any{"x": lut4(b, "table", 128000, 2048), "indices": ids, "axis": b.ScalarI32(0), "validate_indices": b.ScalarBool(false)}, Float16, 1, 32, 2048)
		}},
		{"gather then linear", func(b *Builder) ([]Feature, *Var) {
			ids := b.Input("ids", Int32, 1, 32)
			e := b.Op("gather", map[string]any{"x": lut4(b, "table", 4096, 2048), "indices": ids, "axis": b.ScalarI32(0), "validate_indices": b.ScalarBool(false)}, Float16, 1, 32, 2048)
			return []Feature{{"ids", Int32, ids.Shape}}, b.NamedOp("linear", "y", map[string]any{"x": e, "weight": lut4(b, "w", 2048, 2048)}, Float16, 1, 32, 2048)
		}},
	}
	for i, v := range variants {
		dir := filepath.Join(t.TempDir(), "v.mlpackage")
		pkg, err := Create(dir)
		if err != nil {
			t.Fatal(err)
		}
		b := NewBuilder("CoreML8", pkg.Blobs)
		in, out := v.build(b)
		if err := pkg.Finish(b.Finish(out), 9, in, []Feature{{out.Name, out.DType, out.Shape}}); err != nil {
			t.Fatal(err)
		}
		g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
		if err != nil {
			t.Errorf("%d %s: %v", i, v.name, err)
			continue
		}
		p := g.Placement()
		t.Logf("%-28s npu %d, elsewhere %v", v.name, p.NPUOps, p.Ops)
		g.Close()
	}
}
