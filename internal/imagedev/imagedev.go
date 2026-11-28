// Package imagedev is where hwnpu's models of pictures run and how their devices are opened, as
// hwnpu-image and the ai/vision package pick them: the device backend (on a Mac Core AI's, which
// needs macOS 27), the program format converted for it, the exports of IRON that AMD's NPU runs
// (made with IRON on the first run where none is here), and the CPU's three ways (ONNX Runtime,
// hwnpu's own executor, the CPU interpreter).
package imagedev

import (
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/mem"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/onnxrt"
	"github.com/shibukawa/hwnpu/internal/vision/ironcnn"
)

// Device is the device backend opened for the one named (name, else HWNPU_BACKEND, else the
// NPU found): on a Mac Core AI's, whatever names Apple's Neural Engine (coreml included, as
// HWNPU_BACKEND may for the other commands); Core ML is not used.
func Device(name string) string {
	if name == "" {
		name = strings.TrimSpace(os.Getenv(llm.BackendEnv))
	}
	if runtime.GOOS == "darwin" {
		switch name {
		case "", llm.BackendCoreML, llm.BackendCoreAI:
			return llm.BackendCoreAI
		}
	}
	return name
}

// ErrNoCoreAI explains that a Mac without Core AI has no device for the models of pictures.
func ErrNoCoreAI(err error) error {
	return fmt.Errorf("hwnpu-image runs on Apple's Neural Engine through Core AI, which needs macOS 27 (--cpu runs on the CPU): %w", err)
}

// Format is the program format converted for on the device named: coreai on a Mac (an error
// without Core AI), else llm.ProgramFormat's (intel; an error where neither is found, AMD's NPU
// running exports of IRON).
func Format(ctx context.Context, name string) (string, error) {
	d := Device(name)
	if d != llm.BackendCoreAI {
		return llm.ProgramFormat(ctx, d)
	}
	if _, _, err := llm.FindDevice(ctx, d); err != nil {
		return "", ErrNoCoreAI(err)
	}
	return llm.BackendCoreAI, nil
}

// Open opens the device of Device(name), with the compile cache in cacheDir, and returns it with
// the program format it runs (coreai on a Mac).
func Open(ctx context.Context, name, cacheDir string, opts ...hwnpu.Option) (*hwnpu.Device, string, error) {
	d := Device(name)
	dev, format, err := llm.OpenDevice(ctx, d, cacheDir, opts...)
	if d == llm.BackendCoreAI {
		if err != nil {
			return nil, "", ErrNoCoreAI(err)
		}
		format = llm.BackendCoreAI
	}
	return dev, format, err
}

// ExportDevice is the device backend to open for an export of the program format: Core AI's for
// an export for Apple's Neural Engine (an older Core ML export is lowered to Core AI as it
// loads), else llm.ExportDevice's.
func ExportDevice(format, name string) string {
	if llm.AppleFormat(format) {
		return llm.BackendCoreAI
	}
	return llm.ExportDevice(format, Device(name))
}

// OpenProgramDevice opens the NPU for converted programs (Core AI's or OpenVINO IR), or none for
// the CPU interpreter (cpu, which runs Core ML packages): the device backend named ("":
// HWNPU_BACKEND or the NPU found; a Mac's is Core AI's), with the compile cache in cacheDir (""
// for the user cache directory's). what names the model for the error on an NPU that runs
// neither. The closer closes the device.
func OpenProgramDevice(ctx context.Context, cpu bool, cacheDir, name, what string) (*hwnpu.Device, string, func(), error) {
	if cpu {
		return nil, llm.BackendCoreML, func() {}, nil
	}
	if cacheDir == "" {
		cacheDir = cache.Root()
	}
	dev, backend, err := Open(ctx, name, cacheDir)
	if err != nil {
		return nil, "", nil, err
	}
	if backend == "" {
		dev.Close()
		return nil, "", nil, fmt.Errorf("%s runs neither Core AI programs nor OpenVINO IR (--device cpu runs %s on the CPU)", dev.Info().Backend, what)
	}
	return dev, backend, func() { dev.Close() }, nil
}

// Placement is where a loaded program runs, for the log.
func Placement(cpu, onNPU bool) string {
	switch {
	case cpu:
		return "on the CPU interpreter"
	case onNPU:
		return "all on the NPU"
	}
	return "NOT all on the NPU"
}

// NPUHeld logs, when called after the pictures, how much more memory the Neural Engine holds
// than before (mem.NPUWired: the programs' weights and buffers, outside the process's
// footprint).
func NPUHeld(logf func(string, ...any)) func() {
	before, ok := mem.NPUWired()
	return func() {
		if after, ok2 := mem.NPUWired(); ok && ok2 && after > before {
			logf("the Neural Engine holds %.0f MB more", float64(after-before)/1e6)
		}
	}
}

// OnlyFile is the one file with the extension that models.yaml's sha256 of m names: the file of
// a repository with several models (bukuroo/RTMDet-ONNX).
func OnlyFile(m chat.Model, ext string) (string, error) {
	var files []string
	for f := range maps.Keys(m.SHA256) {
		if strings.EqualFold(filepath.Ext(f), ext) && !strings.Contains(f, "/") {
			files = append(files, f)
		}
	}
	if len(files) != 1 {
		return "", fmt.Errorf("%s: models.yaml's sha256 must name its one %s file (it names %d)", m.Name, ext, len(files))
	}
	return files[0], nil
}

// ProgramDevice reports whether a model of ONNX files that is converted on Apple's and Intel's
// NPU (EfficientSAM's encoder, YuNet) runs as a converted program on this machine's NPU
// (vendor): on Apple's and Intel's, unless flag (a command's --device), HWNPU_DEVICE or
// models.yaml says cpu.
func ProgramDevice(m chat.Model, vendor, flag string) bool {
	if vendor != "apple" && vendor != "intel" {
		return false
	}
	d, _, err := chat.ChooseDevice(m, vendor, flag)
	return err == nil && d == "npu"
}

// CPURoute is where an ONNX model m runs (chat.ConvertedDevice) and logs it: ort, ONNX Runtime
// on the CPU when its library is here; builtin, hwnpu's own CPU executor on the converted
// program (internal/milcpu, on threads goroutines) when it is not, or with interp (the CPU
// interpreter, a command's --cpu: all the cores); neither, the NPU. flag is a command's --device.
func CPURoute(ctx context.Context, m chat.Model, flag string, interp bool, threads int, logf func(string, ...any)) (ort, builtin bool) {
	if interp {
		milcpu.Threads = 0
		return false, true
	}
	device, why, err := chat.ConvertedDevice(m, chat.HereVendor(), flag)
	if err != nil {
		logf("%v; the NPU", err)
		return false, false
	}
	if device != "cpu" {
		logf("%s on the %s (%s)", m.Name, device, why)
		return false, false
	}
	if onnxrt.Available() {
		logf("%s on the CPU with ONNX Runtime (%s)", m.Name, why)
		return true, false
	}
	milcpu.Threads = max(1, threads)
	logf("%s on the CPU with hwnpu's own executor (%s; ONNX Runtime, faster, is not here: HWNPU_ONNXRUNTIME)", m.Name, why)
	return false, true
}

// AMDCNNExport is the export of tools/amd/iron_onnxcnn.py that model m runs from on AMD's NPU
// (ironcnn.Dir: NAME.amd-npu2 in the cache's entries/amd-xdna2, or ~/models), from the shared
// layers or made on the first run with IRON when it is not there (minutes): fetch downloads the
// ONNX file, export makes the export of it in dir, version is the export's (ironcnn.Version or
// the model's own). It is "" where m runs elsewhere: another NPU, interp (the CPU interpreter),
// flag cpu (a command's --device; or models.yaml's device for AMD, HWNPU_DEVICE), or no export
// and no IRON to make one with (then the CPU, said through logf). log takes the shared layers'
// lines.
func AMDCNNExport(ctx context.Context, m chat.Model, flag string, interp bool, logf func(string, ...any), log io.Writer, version int, fetch func() (string, error), export func(python []string, model, dir string) error) string {
	return AMDFirstRunExport(ctx, m, flag, interp, logf, log, func(dir string) error { _, err := ironcnn.Read(dir); return err },
		"iron_onnxcnn.py", version, fetch, export)
}

// AMDFirstRunExport is AMDCNNExport for any tool's export: have says whether dir holds one,
// tool is the script, version the export's version (chat.PinnedRecipe).
func AMDFirstRunExport(ctx context.Context, m chat.Model, flag string, interp bool, logf func(string, ...any), log io.Writer, have func(dir string) error, tool string, version int, fetch func() (string, error), export func(python []string, model, dir string) error) string {
	if interp || chat.HereVendor() != "amd" {
		return ""
	}
	if device, _, err := chat.ChooseDevice(m, "amd", flag); err != nil || device != "npu" {
		return ""
	}
	dir := ironcnn.Dir(m.Name)
	if have(dir) == nil {
		// one made by hand here goes in the sources written back to, described as a first run's
		// (as hwnpu-dev cache import describes it), so that other machines' first runs find it
		chat.ShareMadeHere(m, dir, chat.ExportKind(tool), chat.PinnedRecipe(m, tool, version), log)
		return dir
	}
	// none here: from the shared layers, else made as hwnpu-chat makes its models'
	// (chat.PrepareAMD), with IRON
	errNoIRON := errors.New("no IRON")
	err := chat.SharedExport(ctx, chat.ExportKind(tool), chat.PinnedRecipe(m, tool, version), dir, log, func() error {
		python, err := chat.IRONPython(ctx)
		if err != nil {
			logf("%s: no export for AMD's NPU in %s or the shared layers, and making one needs IRON (%v; INSTALL_AMD.md, %s): the CPU", m.Name, dir, err, chat.IRONEnv)
			return errNoIRON
		}
		model, err := fetch()
		if err == nil {
			err = export(python, model, dir)
		}
		return err
	})
	if err == nil {
		err = have(dir)
	}
	if err != nil {
		if !errors.Is(err, errNoIRON) {
			logf("%s: %v; the CPU", m.Name, err)
		}
		return ""
	}
	return dir
}

// DeviceFlag is the device the definition of m gives for every NPU, as an app's model of its own
// does (chat.DefinedDevice): what the roles of pictures pass as a command's --device. "" when it
// gives none, or not for every NPU.
func DeviceFlag(m chat.Model) string { return chat.DefinedDevice(m) }

// YuNetFile is the ONNX file of YuNet's repository (opencv/face_detection_yunet).
const YuNetFile = "face_detection_yunet_2023mar.onnx"
