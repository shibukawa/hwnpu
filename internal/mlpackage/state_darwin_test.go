package mlpackage

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
	"github.com/shibukawa/hwnpu/internal/types"
)

// A state input keeps its value from one prediction to the next, and ResetState zeros it:
// y = s + x, then s = y, so the outputs are x, 2x, 3x, and x again after a reset.
func TestState(t *testing.T) {
	ctx := context.Background()
	infos, err := coreml.Provider{}.Enumerate(ctx)
	if err != nil || len(infos) == 0 {
		t.Skip("no Neural Engine")
	}
	const n = 64
	dir := filepath.Join(t.TempDir(), "state.mlpackage")
	pkg, err := Create(dir)
	if err != nil {
		t.Fatal(err)
	}
	b := NewBuilder("CoreML8", pkg.Blobs)
	x := b.Input("x", Float16, 1, n)
	s := b.State("s", Float16, 1, n)
	cur := b.Op("read_state", map[string]any{"input": s}, Float16, 1, n)
	sum := b.Op("add", map[string]any{"x": cur, "y": x}, Float16, 1, n)
	y := b.NamedOp("identity", "y", map[string]any{"x": b.UpdateState(s, sum)}, Float16, 1, n)
	pkg.States = []Feature{{"s", Float16, []int{1, n}}}
	if err := pkg.Finish(b.Finish(y), 9, []Feature{{"x", Float16, []int{1, n}}}, []Feature{{"y", Float16, []int{1, n}}}); err != nil {
		t.Fatal(err)
	}
	dev, err := coreml.Provider{}.Open(ctx, infos[0], backend.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer dev.Close()
	g, err := dev.Compile(ctx, types.GraphSource{Path: dir})
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	if len(g.Args()) != 2 {
		t.Fatalf("arguments %v: the state must not be one", g.Args())
	}
	req, err := g.NewRequest()
	if err != nil {
		t.Fatal(err)
	}
	defer req.Close()
	xb, _ := dev.AllocBuffer(2*n, types.BufferDefault)
	yb, _ := dev.AllocBuffer(2*n, types.BufferDefault)
	defer xb.Close()
	defer yb.Close()
	for i := 0; i < n; i++ {
		h := Float16Bits(float32(i%7) - 3)
		xb.Bytes()[2*i], xb.Bytes()[2*i+1] = byte(h), byte(h>>8)
	}
	req.Bind(0, xb)
	req.Bind(1, yb)
	run := func(want float32) {
		t.Helper()
		f, err := req.Submit()
		if err == nil {
			err = f.Wait(ctx)
		}
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < n; i++ {
			got := Float16Value(uint16(yb.Bytes()[2*i]) | uint16(yb.Bytes()[2*i+1])<<8)
			if w := want * (float32(i%7) - 3); got != w {
				t.Fatalf("y[%d] = %v, want %v", i, got, w)
			}
		}
	}
	run(1)
	run(2)
	run(3)
	if err := req.(interface{ ResetState() error }).ResetState(); err != nil {
		t.Fatal(err)
	}
	run(1)
}
