package levelzero

import (
	"testing"

	"github.com/shibukawa/hwnpu/internal/ovir"
	"github.com/shibukawa/hwnpu/internal/types"
	"github.com/shibukawa/hwnpu/internal/ze"
)

// The arguments take the IR's names: the driver's name when it is one, else a debug or tensor
// name that is, else the IR's Parameter or Result of the same place.
func TestArgNames(t *testing.T) {
	model := &ovir.Model{
		Inputs:  []ovir.Port{{Name: "x"}, {Name: "kv_k_1"}},
		Outputs: []ovir.Port{{Name: "y"}, {Name: "kv_k_1_new"}},
	}
	prop := func(name, debug string, tensors ...string) *ze.GraphArgumentProperties3 {
		p := &ze.GraphArgumentProperties3{}
		copy(p.Name[:], name)
		copy(p.DebugFriendlyName[:], debug)
		for i, n := range tensors {
			copy(p.AssociatedTensorNames[i][:], n)
		}
		p.AssociatedTensorNamesCount = uint32(len(tensors))
		return p
	}
	props := []*ze.GraphArgumentProperties3{
		prop("x", "x"),
		prop("Parameter_12", "Parameter_12", "kv_k_1"),
		prop("Result_40", "y/sink_port_0"),
		prop("Result_41", "Result_41"),
	}
	args := []types.ArgInfo{
		{Name: "x"}, {Name: "Parameter_12"},
		{Name: "Result_40", Kind: types.ArgOutput}, {Name: "Result_41", Kind: types.ArgOutput},
	}
	argNames(args, props, model)
	want := []string{"x", "kv_k_1", "y", "kv_k_1_new"}
	for i, a := range args {
		if a.Name != want[i] {
			t.Errorf("argument %d: %q, want %q", i, a.Name, want[i])
		}
	}
}
