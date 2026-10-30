// Package ovrun runs a model converted for Intel's NPU (hwnpu-convert -backend intel) as a
// decode.Program with OpenVINO's CPU plugin (internal/ovcpu), so that the programs hwnpu-chat
// runs on Intel's NPU can be stepped in tests on any machine.
package ovrun

import (
	"fmt"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/ovcpu"
)

// Program loads a model written for Intel's NPU in parts (hwnpu-convert -backend intel,
// out.parts.json, its first tier) as a decode.Program run by ovcpu.Run, part after part: the y of a
// part is the x of the next, and every other input and output is the program's. Each run starts
// a Python process per part, which is slow but enough for the tests' small models.
func Program(python, manifest string, opts ovcpu.Options) (decode.Program, error) {
	man, err := llm.ReadManifest(manifest)
	if err != nil {
		return decode.Program{}, err
	}
	parts := man.Tiered()[0].Parts
	p := decode.Program{In: map[string][]byte{}, Out: map[string][]byte{}, InShape: map[string][]int{}, OutShape: map[string][]int{}}
	for i, part := range parts {
		for _, f := range part.Inputs {
			if f.Name == "x" && i > 0 {
				continue
			}
			p.InShape[f.Name] = f.Shape
			p.In[f.Name] = make([]byte, 2*count(f.Shape))
		}
		outs := part.State
		if i == len(parts)-1 {
			outs = append([]llm.FeatureRecord{part.Output}, outs...)
		}
		for _, f := range outs {
			p.OutShape[f.Name] = f.Shape
			p.Out[f.Name] = make([]byte, 2*count(f.Shape))
		}
	}
	p.Run = func() error {
		var y []float32
		for i, part := range parts {
			in := map[string][]float32{}
			for _, f := range part.Inputs {
				if f.Name == "x" && i > 0 {
					in["x"] = y
					continue
				}
				v := make([]float32, count(f.Shape))
				hwnpu.DecodeF16(v, p.In[f.Name])
				in[f.Name] = v
			}
			res, err := ovcpu.Run(python, part.Path, in, opts)
			if err != nil {
				return fmt.Errorf("%s: %w", part.Path, err)
			}
			for name, v := range res {
				if name == "y" && i < len(parts)-1 {
					y = v
					continue
				}
				if out, ok := p.Out[name]; ok {
					hwnpu.EncodeF16(out, v)
				}
			}
		}
		return nil
	}
	return p, nil
}

func count(sh []int) int {
	n := 1
	for _, d := range sh {
		n *= d
	}
	return n
}
