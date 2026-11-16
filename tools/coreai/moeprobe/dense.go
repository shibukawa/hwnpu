package main

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// writeDense writes one layer's 32 experts as one program that computes them all for every row
// and weighs them by the input r (1, seq, 32), zero for an expert not routed to: the routing
// decides nothing about what the Neural Engine reads. The experts go in groups of per experts,
// each a linear of the group's gates and ups and one of its downs over the weighted rows, under
// the Neural Engine's 16384 per axis.
func writeDense(path string, seq, experts, per, layer int) error {
	_, err := llm.WriteAI(path, nil, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
		g := &blocks.Graph{Name: "moeprobe", B: b, W: synth{}, Opts: llm.Options{Format: "lut4", Group: 16}}
		var ins, outs []mlpackage.Feature
		err := blocks.Run(func() {
			x := b.Input("x", mlpackage.Float16, 1, seq, d)
			r := b.Input("r", mlpackage.Float16, 1, seq, experts)
			var sum *mlpackage.Var
			for e0 := 0; e0 < experts; e0 += per {
				var parts []*mlpackage.Var
				for e := e0; e < e0+per; e++ {
					h := expertHidden(g, x, layer*experts+e)
					we := g.Op("slice_by_index", map[string]any{"x": r, "begin": b.Ints(0, 0, int32(e)), "end": b.Ints(1, int32(seq), int32(e+1))}, 1, seq, 1)
					parts = append(parts, g.Op("mul", map[string]any{"x": h, "y": we}, h.Shape...))
				}
				cat := parts[0]
				if len(parts) > 1 {
					cat = g.Op("concat", map[string]any{"values": parts, "axis": b.ScalarI32(2), "interleave": b.ScalarBool(false)}, 1, seq, per*inter)
				}
				name := fmt.Sprintf("down%d", e0)
				y := g.Op("linear", map[string]any{"x": cat, "weight": g.Matrix(name, denseDown(layer*experts+e0, per), d, per*inter)}, 1, seq, d)
				if sum == nil {
					sum = y
				} else {
					sum = g.Op("add", map[string]any{"x": sum, "y": y}, sum.Shape...)
				}
			}
			y := g.B.NamedOp("identity", "y", map[string]any{"x": sum}, mlpackage.Float16, sum.Shape...)
			ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}, {Name: "r", DType: mlpackage.Float16, Shape: r.Shape}}
			outs = []mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}}
		})
		return ins, outs, err
	})
	return err
}

// expertHidden is an expert before its down: (up + 1) * gate * sigmoid(1.702 gate).
func expertHidden(g *blocks.Graph, x *mlpackage.Var, e int) *mlpackage.Var {
	p := fmt.Sprintf("e%d.", e)
	gate := g.LinearBias(x, p+"gate", p+"gate_b", "", inter)
	up := g.LinearBias(x, p+"up", p+"up_b", "", inter)
	gate = g.Op("minimum", map[string]any{"x": gate, "y": g.B.ScalarF16(7)}, gate.Shape...)
	up = g.Op("clip", map[string]any{"x": up, "alpha": g.B.ScalarF16(-7), "beta": g.B.ScalarF16(7)}, up.Shape...)
	neg := g.Op("mul", map[string]any{"x": gate, "y": g.B.ScalarF16(-1.702)}, gate.Shape...)
	den := g.Op("add", map[string]any{"x": g.Op("exp", map[string]any{"x": neg}, gate.Shape...), "y": g.B.ScalarF16(1)}, gate.Shape...)
	glu := g.Op("real_div", map[string]any{"x": gate, "y": den}, gate.Shape...)
	up1 := g.Op("add", map[string]any{"x": up, "y": g.B.ScalarF16(1)}, up.Shape...)
	return g.Op("mul", map[string]any{"x": up1, "y": glu}, gate.Shape...)
}

// denseDown is the downs of experts e0.. side by side: (d, per*inter).
func denseDown(e0, per int) []float32 {
	out := make([]float32, d*per*inter)
	for k := 0; k < per; k++ {
		w, _, _ := synth{}.Get(fmt.Sprintf("e%d.down", e0+k))
		for r := 0; r < d; r++ {
			copy(out[r*per*inter+k*inter:], w[r*inter:(r+1)*inter])
		}
	}
	return out
}
