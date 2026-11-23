package text

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/airole"
)

// noAMDForm is why a decision model without an export for AMD's NPU (its recipe, model-recipes.yaml)
// does not run there: hwnpu has no form of it, said at once instead of looking for a device that
// runs the converted programs ("no NPU device found" on a machine that has one). Programs given
// (packages) and the CPU interpreter still run.
func noAMDForm(e airole.PrepareEnv) error {
	if !e.NPU.AMD() || len(e.Model.Programs["packages"]) > 0 || e.NPU.CPU(e.Model) {
		return nil
	}
	return fmt.Errorf("%s has no form for AMD's NPU: it has no recipe of iron_export_lm.py in model-recipes.yaml (device: cpu in a definition of your own runs it on the CPU interpreter, slowly)", e.Model.Name)
}
