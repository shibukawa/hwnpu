package mlpackage

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/types"
)

// Compiles one small program per operation the converters emit, selected by HWNPU_VARIANT, so a
// loader crash on one of them can be pinned down from a shell loop.
func TestOpVariant(t *testing.T) {
	name := os.Getenv("HWNPU_VARIANT")
	if name == "" {
		t.Skip("HWNPU_VARIANT not set")
	}
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
	const s, d = 8, 32
	builds := map[string]func(b *Builder) *Var{
		"rmsnorm": func(b *Builder) *Var {
			x := b.Input("x", Float16, 1, s, d)
			xs := b.Op("mul", map[string]any{"x": x, "y": b.ScalarF16(0.0625)}, Float16, 1, s, d)
			sq := b.Op("mul", map[string]any{"x": xs, "y": xs}, Float16, 1, s, d)
			ms := b.Op("reduce_mean", map[string]any{"x": sq, "axes": b.Ints(-1), "keep_dims": b.ScalarBool(true)}, Float16, 1, s, 1)
			r := b.Op("rsqrt", map[string]any{"x": ms, "epsilon": b.ScalarF16(1e-5)}, Float16, 1, s, 1)
			return b.NamedOp("mul", "y", map[string]any{"x": xs, "y": r}, Float16, 1, s, d)
		},
		"transpose_slice": func(b *Builder) *Var {
			x := b.Input("x", Float16, 1, s, d)
			tr := b.Op("transpose", map[string]any{"x": x, "perm": b.Ints(0, 2, 1)}, Float16, 1, d, s)
			bools := func(v ...bool) *Var { r, _ := b.Bools("m", []int{len(v)}, v); return r }
			a := b.Op("slice_by_index", map[string]any{"x": tr, "begin": b.Ints(0, 0, 0), "end": b.Ints(1, d/2, s), "begin_mask": bools(true, false, true), "end_mask": bools(true, false, true)}, Float16, 1, d/2, s)
			c := b.Op("slice_by_index", map[string]any{"x": tr, "begin": b.Ints(0, d/2, 0), "end": b.Ints(1, d, s), "begin_mask": bools(true, false, true), "end_mask": bools(true, false, true)}, Float16, 1, d/2, s)
			return b.NamedOp("mul", "y", map[string]any{"x": a, "y": c}, Float16, 1, d/2, s)
		},
		"pad_conv": func(b *Builder) *Var {
			x := b.Input("x", Float16, 1, d, s)
			p := b.Op("pad", map[string]any{"x": x, "pad": b.Ints(6, 0), "mode": b.Str("mode", "constant"), "constant_val": b.ScalarF16(0)}, Float16, 1, d, s+6)
			w, _ := b.F16s("w", []int{d, 1, 7}, make([]float32, d*7))
			return b.NamedOp("conv", "y", map[string]any{"x": p, "weight": w, "strides": b.Ints(1), "pad_type": b.Str("pt", "valid"), "pad": b.Ints(0, 0), "dilations": b.Ints(1), "groups": b.ScalarI32(d)}, Float16, 1, d, s)
		},
		"pad_only": func(b *Builder) *Var {
			x := b.Input("x", Float16, 1, d, s)
			return b.NamedOp("pad", "y", map[string]any{"x": x, "pad": b.Ints(6, 0), "mode": b.Str("mode", "constant"), "constant_val": b.ScalarF16(0)}, Float16, 1, d, s+6)
		},
		"conv_only": func(b *Builder) *Var {
			x := b.Input("x", Float16, 1, d, s+6)
			w, _ := b.F16s("w", []int{d, 1, 7}, make([]float32, d*7))
			return b.NamedOp("conv", "y", map[string]any{"x": x, "weight": w, "strides": b.Ints(1), "pad_type": b.Str("pt", "valid"), "pad": b.Ints(0, 0), "dilations": b.Ints(1), "groups": b.ScalarI32(d)}, Float16, 1, d, s)
		},
		"silu_add_identity": func(b *Builder) *Var {
			x := b.Input("x", Float16, 1, s, d)
			a := b.Op("silu", map[string]any{"x": x}, Float16, 1, s, d)
			a = b.Op("add", map[string]any{"x": a, "y": x}, Float16, 1, s, d)
			return b.NamedOp("identity", "y", map[string]any{"x": a}, Float16, 1, s, d)
		},
		"linear_nobias_blob": func(b *Builder) *Var {
			x := b.Input("x", Float16, 1, s, d)
			w, _ := b.F16s("w", []int{d, d}, make([]float32, d*d))
			return b.NamedOp("linear", "y", map[string]any{"x": x, "weight": w}, Float16, 1, s, d)
		},
	}
	build, ok := builds[name]
	if !ok {
		t.Fatalf("no variant %q", name)
	}
	dir := filepath.Join(t.TempDir(), "v.mlpackage")
	pkg, err := Create(dir)
	if err != nil {
		t.Fatal(err)
	}
	b := NewBuilder("CoreML8", pkg.Blobs)
	y := build(b)
	in := b.inputs[0]
	inShape := []int{}
	for _, dim := range in.Type.GetTensorType().Dimensions {
		inShape = append(inShape, int(dim.GetConstant().Size))
	}
	if err := pkg.Finish(b.Finish(y), 9, []Feature{{in.Name, Float16, inShape}}, []Feature{{y.Name, y.DType, y.Shape}}); err != nil {
		t.Fatal(err)
	}
	g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	p := g.Placement()
	t.Logf("%s: ok, npu %d, elsewhere %v", name, p.NPUOps, p.Ops)
	g.Close()
}
