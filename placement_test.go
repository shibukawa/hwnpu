package hwnpu

import (
	"context"
	"errors"
	"testing"

	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/types"
)

type fakeDevice struct{ rep types.PlacementReport }

func (d fakeDevice) AllocBuffer(int, types.BufferKind) (backend.Buffer, error) {
	return nil, ErrUnsupported
}
func (d fakeDevice) Close() error { return nil }
func (d fakeDevice) Compile(context.Context, types.GraphSource) (backend.Graph, error) {
	return &fakeGraph{rep: d.rep}, nil
}

type fakeGraph struct {
	rep    types.PlacementReport
	closed bool
}

func (g *fakeGraph) Args() []types.ArgInfo                { return nil }
func (g *fakeGraph) Cached() bool                         { return false }
func (g *fakeGraph) Placement() types.PlacementReport     { return g.rep }
func (g *fakeGraph) NewRequest() (backend.Request, error) { return nil, ErrUnsupported }
func (g *fakeGraph) Close() error                         { g.closed = true; return nil }

func compileWith(p Placement, rep types.PlacementReport) (*Graph, error) {
	d := &Device{info: DeviceInfo{Caps: CapGraph}, be: fakeDevice{rep: rep}, opts: options{placement: p}}
	gr, _ := d.Graph()
	return gr.Compile(context.Background(), GraphSource{})
}

// Acceptance requirement:npu-only-execution/unverifiable-strict.
func TestStrictRejectsUnverifiedPlacement(t *testing.T) {
	rep := types.PlacementReport{Verified: false, Reason: "no compute plan"}
	_, err := compileWith(PlacementStrict, rep)
	var pe *PlacementError
	if !errors.Is(err, ErrNotOnNPU) || !errors.As(err, &pe) || pe.Report.Verified {
		t.Fatalf("strict: err = %v", err)
	}
	g, err := compileWith(PlacementPrefer, rep)
	if err != nil || g.Placement().Verified {
		t.Fatalf("prefer: g=%v err=%v", g, err)
	}
	if g, err := compileWith(PlacementStrict, types.PlacementReport{Verified: true, NPUOps: 3}); err != nil || !g.Placement().OnNPU() {
		t.Fatalf("verified on NPU: g=%v err=%v", g, err)
	}
}
