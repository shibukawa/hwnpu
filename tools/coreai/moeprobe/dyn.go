package main

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/blocks"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// writeDyn writes a program of one (rows, cols) linear whose weight is lut8 with its indices
// (uint8, one a weight) and tables (fp16, one of 256 entries a 16 rows) given as inputs, and the
// same linear with constant lut8 weights: whether the Neural Engine runs a lookup table it is
// given at run time.
func writeDyn(dir string, seq, rows, cols int) error {
	_, err := llm.WriteAI(dir+"/dyn.aimodel", nil, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
		x := b.Input("x", mlpackage.Float16, 1, seq, cols)
		idx := b.Input("idx", mlpackage.UInt8, rows, cols)
		lut := b.Input("lut", mlpackage.Float16, rows/16, 1, 256, 1)
		w := b.Op("constexpr_lut_to_dense", map[string]any{"indices": idx, "lut": lut}, mlpackage.Float16, rows, cols)
		y := b.NamedOp("linear", "y", map[string]any{"x": x, "weight": w}, mlpackage.Float16, 1, seq, rows)
		return []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}, {Name: "idx", DType: mlpackage.UInt8, Shape: idx.Shape}, {Name: "lut", DType: mlpackage.Float16, Shape: lut.Shape}},
			[]mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}}, nil
	})
	if err != nil {
		return fmt.Errorf("dyn: %w", err)
	}
	_, err = llm.WriteAI(dir+"/const.aimodel", nil, func(b *mlpackage.Builder) ([]mlpackage.Feature, []mlpackage.Feature, error) {
		g := &blocks.Graph{Name: "moeprobe", B: b, W: synth{}, Opts: llm.Options{Format: "lut8", Group: 16}}
		var ins, outs []mlpackage.Feature
		err := blocks.Run(func() {
			x := b.Input("x", mlpackage.Float16, 1, seq, cols)
			y := g.Linear(x, "e0.down", "y", rows)
			ins = []mlpackage.Feature{{Name: "x", DType: mlpackage.Float16, Shape: x.Shape}}
			outs = []mlpackage.Feature{{Name: "y", DType: mlpackage.Float16, Shape: y.Shape}}
		})
		return ins, outs, err
	})
	return err
}
