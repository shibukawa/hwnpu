package flux2klein

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/backend/amd"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagegen/aspect"
)

// NPUGeneration is the generation of this machine's NPU of vendor as models.yaml names it: the
// chip on Apple, xdna1 or xdna2 on AMD, "" when it is not known (the default sizes then).
func NPUGeneration(vendor string) string {
	switch vendor {
	case "apple":
		return chat.Chip()
	case "amd":
		return amd.Generation()
	}
	return ""
}

// NPUAspects are the picture sizes of m's export for vendor's NPU on generation gen: models.yaml
// npu's sizes, with the generation's overrides, in the order of aspect.All.
func NPUAspects(m chat.Model, vendor, gen string) ([]aspect.Aspect, error) {
	spec, ok := m.NPU[vendor]
	if !ok {
		return nil, fmt.Errorf("%s has no %s sizes in models.yaml (npu)", m.Name, vendor)
	}
	sizes := spec.SizesOn(gen)
	out := make([]aspect.Aspect, 0, len(aspect.All))
	for _, a := range aspect.All {
		wh, ok := sizes[a.Name]
		if !ok {
			return nil, fmt.Errorf("%s: no %s size for %s in models.yaml (npu)", m.Name, vendor, a.Name)
		}
		out = append(out, aspect.Aspect{Name: a.Name, Width: wh[0], Height: wh[1]})
	}
	return out, nil
}
