package blocks

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/types"
)

// ChunkedSDPA over a KV cache in chunks of up to the Neural Engine's dimension limit is placed on
// it, for one token per call and for a prompt's 64. With the sums of the exps and the weighted
// values taken elementwise and by reduce_sum, the placement of 64 tokens per call changed from
// one compile to the next (M3, macOS 27.0.1); as matrix products it held in every compile.
func TestChunkedSDPAOnNeuralEngine(t *testing.T) {
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
	for _, c := range []struct{ q, chunk, n int }{{1, 16384, 2}, {16, 16384, 2}, {64, 16384, 2}, {64, 16384, 4}, {64, 4096, 8}} {
		dir := filepath.Join(t.TempDir(), "m.mlpackage")
		pkg, err := mlpackage.Create(dir)
		if err != nil {
			t.Fatal(err)
		}
		b := mlpackage.NewBuilder("CoreML8", pkg.Blobs)
		g := &Graph{Name: "t", B: b}
		var ins []mlpackage.Feature
		in := func(name string, sh ...int) *mlpackage.Var {
			ins = append(ins, mlpackage.Feature{Name: name, DType: mlpackage.Float16, Shape: sh})
			return b.Input(name, mlpackage.Float16, sh...)
		}
		q := in("q", 1, 16, c.q, 128)
		var ks, vs, ms []*mlpackage.Var
		for j := 0; j < c.n; j++ {
			ks = append(ks, in(fmt.Sprintf("k%d", j), 1, 4, c.chunk, 128))
			vs = append(vs, in(fmt.Sprintf("v%d", j), 1, 4, c.chunk, 128))
			ms = append(ms, in(fmt.Sprintf("m%d", j), 1, 1, c.q, c.chunk))
		}
		ks = append(ks, in("k_self", 1, 4, c.q, 128))
		vs = append(vs, in("v_self", 1, 4, c.q, 128))
		ms = append(ms, in("m_self", 1, 1, c.q, c.q))
		o := g.ChunkedSDPA(q, ks, vs, ms)
		y := b.NamedOp("identity", "y", map[string]any{"x": o}, mlpackage.Float16, o.Shape...)
		if err := pkg.Finish(b.Finish(y), 9, ins, []mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: o.Shape}}); err != nil {
			t.Fatal(err)
		}
		gr, err := dev.Compile(ctx, types.GraphSource{Path: dir})
		if err != nil {
			t.Fatalf("q %d, %d chunks of %d: %v", c.q, c.n, c.chunk, err)
		}
		if p := gr.Placement(); p.Verified && !p.OnNPU() {
			t.Errorf("q %d, %d chunks of %d: %d operations off the Neural Engine", c.q, c.n, c.chunk, len(p.Ops))
		}
		gr.Close()
	}
}
