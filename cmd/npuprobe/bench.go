package main

import (
	"fmt"
)

// bench runs every format at the given layer counts on the NPU and, on Apple, on the CPU and the
// GPU through Core ML for comparison. A model too large for the NPU is still measured: the table
// then shows where Core ML put it.
func (p *prober) bench(rep *report, formats []string, layers []int, baselines []string, save func()) {
	units := append([]string{"npu"}, baselines...)
	for _, n := range layers {
		for _, f := range formats {
			fmt.Printf("bench, %s %d layers:\n", f, n)
			for _, u := range units {
				if p.ctx.Err() != nil {
					return
				}
				if p.coreml && u == "npu" && f == "int4" {
					continue // Core ML never places blockwise int4 on the Neural Engine
				}
				r := p.isolated(childSpec{Kind: "try", Format: f, Layers: n, Units: u})[0]
				logResult(r)
				rep.Bench = append(rep.Bench, r)
			}
			save()
		}
	}
}
