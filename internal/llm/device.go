package llm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/backend/openvino"
)

// The program formats hwnpu-convert writes (Options.Backend).
const (
	BackendCoreML = "coreml" // .mlpackage, for the Neural Engine
	BackendIntel  = "intel"  // OpenVINO IR, for Intel's NPU
)

// BackendEnv names the environment variable that picks the device to run on when a command is not
// told: a device backend (coreml, levelzero, openvino) or a program format (intel: levelzero).
const BackendEnv = "HWNPU_BACKEND"

// ProgramBackend is the program format a device backend runs (Options.Backend): coreml for Core
// ML, intel for Level Zero and OpenVINO; "" for a backend that runs neither.
func ProgramBackend(device string) string {
	switch device {
	case "coreml":
		return BackendCoreML
	case "levelzero", "openvino":
		return BackendIntel
	}
	return ""
}

// deviceBackends are the device backends that run a program format, in the order they are
// preferred when the device is not named.
func deviceBackends(name string) ([]string, error) {
	switch name {
	case "":
		return []string{"coreml", "levelzero", "openvino"}, nil
	case BackendIntel:
		return []string{"levelzero", "openvino"}, nil
	case "coreml", "levelzero", "openvino":
		return []string{name}, nil
	}
	return nil, fmt.Errorf("llm: unknown backend %q (coreml, intel, levelzero or openvino)", name)
}

// FindDevice finds the device a converted model runs on: the one of the backend named (a device
// backend, or a program format), else of HWNPU_BACKEND, else the first device found of Core ML or
// Level Zero (Intel's NPU). It returns the device and its program format (ProgramBackend).
func FindDevice(ctx context.Context, name string) (hwnpu.DeviceInfo, string, error) {
	if name == "" {
		name = strings.TrimSpace(os.Getenv(BackendEnv))
	}
	want, err := deviceBackends(name)
	if err != nil {
		return hwnpu.DeviceInfo{}, "", err
	}
	if name == "openvino" {
		openvino.Enable()
	}
	infos, enumErr := hwnpu.Enumerate(ctx)
	for _, b := range want {
		for _, info := range infos {
			if info.Backend == b {
				return info, ProgramBackend(b), nil
			}
		}
	}
	err = fmt.Errorf("no NPU that runs the converted models (Apple's Neural Engine or Intel's NPU)")
	if name != "" {
		err = fmt.Errorf("no device of the backend %s", name)
	}
	if enumErr != nil {
		err = fmt.Errorf("%w: %w", err, enumErr)
	}
	return hwnpu.DeviceInfo{}, "", errors.Join(hwnpu.ErrNoDevice, err)
}

// ProgramFormat is the program format (Options.Backend) to convert for: name when it is a format
// (coreml, intel), the format of a device backend named, else that of the device FindDevice
// finds. A format named needs no device, so that a model can be converted for another machine.
func ProgramFormat(ctx context.Context, name string) (string, error) {
	if name == "" {
		name = strings.TrimSpace(os.Getenv(BackendEnv))
	}
	switch name {
	case BackendCoreML, BackendIntel:
		return name, nil
	case "levelzero", "openvino":
		return ProgramBackend(name), nil
	}
	_, backend, err := FindDevice(ctx, name)
	return backend, err
}

// OpenDevice opens the device FindDevice finds, with the compile cache in cacheDir ("" for none),
// and returns it with its program format. Graphs are compiled with PlacementPrefer: Core ML may put
// a few operations on the CPU, which the caller reports (chain.Chain.OnNPU); Intel's compiler
// puts everything on the NPU or fails.
func OpenDevice(ctx context.Context, name, cacheDir string, opts ...hwnpu.Option) (*hwnpu.Device, string, error) {
	info, backend, err := FindDevice(ctx, name)
	if err != nil {
		return nil, "", err
	}
	if backend == "coreml" {
		if err := lockNPU(ctx); err != nil {
			return nil, "", err
		}
	}
	opts = append([]hwnpu.Option{hwnpu.WithPlacement(hwnpu.PlacementPrefer)}, opts...)
	if cacheDir != "" {
		opts = append(opts, hwnpu.WithCacheDir(cacheDir))
	}
	dev, err := hwnpu.OpenDevice(ctx, info, opts...)
	if err != nil {
		return nil, "", err
	}
	return dev, backend, nil
}

// DeviceName is what logs call the device: the Neural Engine or the device's own name.
func DeviceName(dev *hwnpu.Device) string {
	info := dev.Info()
	if info.Backend == "coreml" {
		return "the Neural Engine"
	}
	return info.Name
}
