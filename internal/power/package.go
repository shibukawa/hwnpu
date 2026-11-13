package power

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/shibukawa/hwnpu/internal/types"
)

// PackagePower reads the package power the amdgpu driver reports on an AMD APU (its hwmon's
// power1_input, PPT: the whole socket, CPU cores, GPU and NPU), readable without root, for an
// Integrator. An error when the machine has none (not Linux, no AMD APU).
func PackagePower() (Reader, error) {
	dirs, _ := filepath.Glob("/sys/class/hwmon/hwmon*")
	for _, d := range dirs {
		name, err := os.ReadFile(filepath.Join(d, "name"))
		if err != nil || strings.TrimSpace(string(name)) != "amdgpu" {
			continue
		}
		p := filepath.Join(d, "power1_input")
		read := func() (float64, types.PowerScope, string, error) {
			b, err := os.ReadFile(p)
			if err != nil {
				return 0, "", "", err
			}
			v, err := strconv.ParseFloat(strings.TrimSpace(string(b)), 64)
			if err != nil {
				return 0, "", "", err
			}
			return v / 1e6, types.ScopePackage, "amdgpu PPT", nil
		}
		if _, _, _, err := read(); err == nil {
			return read, nil
		}
	}
	return nil, fmt.Errorf("power: no package power meter (amdgpu's hwmon power1_input): %w", types.ErrUnsupported)
}
