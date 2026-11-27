package chat

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
)

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

// HereVendor is the vendor of this machine's NPU ("amd", "apple", "intel"), "" without one.
// HWNPU_BACKEND naming a backend of Intel's programs (intel, levelzero, openvino) makes it
// "intel", and coreml or coreai "apple": with openvino, a machine without Intel's NPU runs the commands'
// Intel path on OpenVINO's CPU, the IR it converts included.
var HereVendor = sync.OnceValue(hereVendor)

func hereVendor() string {
	switch b := strings.TrimSpace(os.Getenv(llm.BackendEnv)); {
	case b == llm.BackendIntel || llm.ProgramBackend(b) == llm.BackendIntel:
		return "intel"
	case llm.ProgramBackend(b) == llm.BackendCoreML: // coreml, coreai
		return "apple"
	}
	infos, _ := hwnpu.Enumerate(context.Background())
	for _, info := range infos {
		if v := info.Vendor.String(); slices.Contains(Vendors, v) {
			return v
		}
	}
	return ""
}

// ConvertedDevice is where m, a model hwnpu converts from ONNX for Core ML and OpenVINO (RTMDet,
// RTMO, PP-OCR, SwiftF0, Basic Pitch), runs on a machine whose NPU is vendor's: ChooseDevice's
// pick, except that an NPU with no converted form (AMD's, or none) runs it on the CPU with ONNX
// Runtime unless --device npu (flag) asks for the NPU.
func ConvertedDevice(m Model, vendor, flag string) (device, why string, err error) {
	device, why, err = ChooseDevice(m, vendor, flag)
	if err != nil {
		return "", "", err
	}
	if device == "npu" && flag != "npu" && vendor != "apple" && vendor != "intel" {
		v := vendor
		if v == "" {
			v = "none"
		}
		device, why = "cpu", fmt.Sprintf("no converted form of %s for this NPU (%s)", m.Name, v)
	}
	return device, why, nil
}
