package chat

import (
	"context"
	"fmt"
	"maps"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
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

// PerVendor is a value by the vendor of the machine's NPU (Vendors); "" holds the one for the
// vendors it does not name. A definition writes it as one number (every vendor) or as a table
// of vendors (memory: {amd: 3.5, apple: 4.1}): what a model takes and downloads can differ by
// NPU (Anima-Turbo: 3.5 GB on AMD's, 4.1 on Apple's).
type PerVendor map[string]float64

// For is the value for vendor: its own, else the one for every vendor, else the largest of the
// table (a vendor it leaves out is not assumed to need less).
func (p PerVendor) For(vendor string) float64 {
	if v, ok := p[vendor]; ok {
		return v
	}
	if v, ok := p[""]; ok {
		return v
	}
	m := 0.0
	for _, v := range p {
		m = max(m, v)
	}
	return m
}

// UnmarshalYAML reads a number or a table of vendors.
func (p *PerVendor) UnmarshalYAML(unmarshal func(any) error) error {
	var one float64
	if err := unmarshal(&one); err == nil {
		*p = PerVendor{"": one}
		return nil
	}
	var by map[string]float64
	if err := unmarshal(&by); err != nil {
		return fmt.Errorf("a number or a table of vendors (%s)", strings.Join(Vendors, ", "))
	}
	for v := range by {
		if !slices.Contains(Vendors, v) {
			return fmt.Errorf("vendor %q is not one of %s", v, strings.Join(Vendors, ", "))
		}
	}
	*p = by
	return nil
}

// over is p with q written over it: q's number replaces p, q's table changes the vendors it
// names.
func (p PerVendor) over(q PerVendor) PerVendor {
	if _, one := q[""]; one || p == nil {
		return maps.Clone(q)
	}
	r := maps.Clone(p)
	maps.Copy(r, q)
	return r
}

// MemoryFor is what m takes running on a machine whose NPU is vendor's, in GB (0: unknown).
func (m Model) MemoryFor(vendor string) float64 { return m.MemoryBy.For(vendor) }

// BytesFor is the size of m's download on a machine whose NPU is vendor's (0: unknown).
func (m Model) BytesFor(vendor string) float64 { return m.BytesBy.For(vendor) }

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
