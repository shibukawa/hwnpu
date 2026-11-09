package chat

import (
	"fmt"
	"os"
	"slices"
	"strings"
)

// Vendors are the NPU vendors a model's Device names (types.Vendor's names).
var Vendors = []string{"amd", "apple", "intel"}

// Devices are where a model runs: the NPU, or the CPU beside it (a small model can cost less
// there: YuNet on AMD's NPU took 16 times the energy of the CPU, docs/amd-npu-models.md 17).
var Devices = []string{"npu", "cpu"}

// DeviceFor is where m runs on a machine whose NPU is vendor's: its definition's Device, "npu"
// where it names none.
func (m Model) DeviceFor(vendor string) string {
	if d, ok := m.Device[vendor]; ok {
		return d
	}
	return "npu"
}

// ChooseDevice is where m runs on a machine whose NPU is vendor's: flag (a command's --device)
// when given, else HWNPU_DEVICE, else the definitions' (DeviceFor). why says which.
func ChooseDevice(m Model, vendor, flag string) (device, why string, err error) {
	pick := func(d, from string) (string, string, error) {
		d = strings.ToLower(strings.TrimSpace(d))
		if !slices.Contains(Devices, d) {
			return "", "", fmt.Errorf("%s %q is not one of %s", from, d, strings.Join(Devices, ", "))
		}
		return d, from, nil
	}
	switch {
	case flag != "":
		return pick(flag, "--device")
	case os.Getenv("HWNPU_DEVICE") != "":
		return pick(os.Getenv("HWNPU_DEVICE"), "HWNPU_DEVICE")
	}
	if _, ok := m.Device[vendor]; ok {
		return m.DeviceFor(vendor), "models.yaml (" + vendor + ")", nil
	}
	return "npu", "the default", nil
}
