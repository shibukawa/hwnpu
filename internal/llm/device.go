package llm

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/shibukawa/hwnpu"
)

// BackendEnv names the environment variable that overrides HostBackend.
const BackendEnv = "HWNPU_BACKEND"

// HostBackend is the backend (Options.Backend) of the programs this machine's NPU runs: "coreml"
// on macOS (the Neural Engine) and "intel" elsewhere (Intel's NPU through Level Zero; the
// writer for AMD's is not there yet). HWNPU_BACKEND overrides it.
func HostBackend() string {
	if b := os.Getenv(BackendEnv); b != "" {
		return b
	}
	if runtime.GOOS == "darwin" {
		return "coreml"
	}
	return "intel"
}

// CheckBackend rejects a backend there is no converter and runtime for.
func CheckBackend(backend string) error {
	switch backend {
	case "coreml", "intel":
		return nil
	}
	return fmt.Errorf("backend %q: coreml or intel", backend)
}

// ProgramExt is the file extension of a program written for backend: .mlpackage for Core ML,
// .xml (with its .bin) for Intel's NPU.
func ProgramExt(backend string) string {
	if backend == "intel" {
		return ".xml"
	}
	return ".mlpackage"
}

// TrimProgramExt removes a program's or a manifest's extension from path.
func TrimProgramExt(path string) string {
	for _, ext := range []string{".parts.json", ".mlpackage", ".xml"} {
		if strings.HasSuffix(path, ext) {
			return strings.TrimSuffix(path, ext)
		}
	}
	return path
}

// OpenDevice opens the NPU that runs programs written for backend, with opts besides. Core ML is
// opened with PlacementPrefer, since it may place an operation off the Neural Engine and the
// loaders report it; Intel's compiler does not, so a graph that compiles runs all on the NPU.
func OpenDevice(ctx context.Context, backend string, opts ...hwnpu.Option) (*hwnpu.Device, error) {
	var o []hwnpu.Option
	switch backend {
	case "", "coreml":
		o = []hwnpu.Option{hwnpu.WithBackend("coreml"), hwnpu.WithPlacement(hwnpu.PlacementPrefer)}
	case "intel":
		o = []hwnpu.Option{hwnpu.WithBackend("levelzero"), hwnpu.WithVendor(hwnpu.VendorIntel)}
	default:
		return nil, CheckBackend(backend)
	}
	d, err := hwnpu.Open(ctx, append(o, opts...)...)
	if err != nil {
		return nil, fmt.Errorf("opening the NPU for %s programs: %w", backend, err)
	}
	return d, nil
}

// ProgramBackend tells the backend a converted program was written for from its path: a .xml
// is Intel's, a manifest says (or its first part's extension does), anything else is Core ML's.
func ProgramBackend(path string) string {
	if strings.HasSuffix(path, ".xml") {
		return "intel"
	}
	if strings.HasSuffix(path, ".json") {
		if m, err := ReadManifest(path); err == nil {
			if m.Backend != "" {
				return m.Backend
			}
			if t := m.Tiered(); len(t) > 0 && len(t[0].Parts) > 0 && strings.HasSuffix(t[0].Parts[0].Path, ".xml") {
				return "intel"
			}
		}
	}
	return "coreml"
}
