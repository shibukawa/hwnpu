package engine

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
)

// cnn is how a model of one ONNX file that is converted for the NPU (RTMDet, RTMO) is made
// ready and loaded, as hwnpu-image does: on AMD's NPU an export of tools/amd/iron_onnxcnn.py
// (from the shared layers, or made with IRON on the first run), on Apple's and Intel's NPU the
// file converted, on the CPU the file with ONNX Runtime, or the converted program on hwnpu's own
// executor without ONNX Runtime, or on the CPU interpreter (settings cpu: interpreter).
type cnn[T io.Closer] struct {
	// what names the model in the errors (RTMDet).
	what string
	// version is the version of the export of AMD's NPU (chat.PinnedRecipe).
	version int
	// fetch downloads the ONNX file of model m; export makes the export of AMD's NPU of it in
	// dir; convert downloads and converts it for the program format, in the directory of base.
	fetch   func(ctx context.Context, m chat.Model, log io.Writer) (string, error)
	export  func(ctx context.Context, python []string, name, model, dir string, log io.Writer) error
	convert func(ctx context.Context, m, base chat.Model, backend string, log io.Writer) (model, program string, err error)
	// openNPU loads an export of AMD's NPU, openCPU the file into ONNX Runtime, open a
	// converted program (dev nil: on the CPU); onNPU says whether that one runs all on the NPU.
	openNPU func(ctx context.Context, dev *hwnpu.Device, dir, model string) (T, error)
	openCPU func(model string, threads int) (T, error)
	open    func(ctx context.Context, dev *hwnpu.Device, model, program string) (T, error)
	onNPU   func(T) bool
	// programOnly says open runs a converted Core ML package without the ONNX file (convert gives
	// no model's path for it: the program alone, or with its part file of what the host reads):
	// on the CPU one from the shared layers (chat.CoreMLKind) is run on hwnpu's own executor in
	// place of the ONNX file, even where ONNX Runtime is, and the ONNX file is not downloaded
	programOnly bool
}

func (c cnn[T]) prepare(e airole.PrepareEnv) (f airole.Files, err error) {
	m := e.Model
	flag, interp := imagedev.DeviceFlag(m), interpreter(m)
	fetch := func() (string, error) { return c.fetch(e.Ctx, m, e.Log) }
	export := func(python []string, model, dir string) error {
		return c.export(e.Ctx, python, e.Base.Name, model, dir, e.Log)
	}
	// the export of AMD's NPU is the one of the model the app's was made from
	if dir := imagedev.AMDCNNExport(e.Ctx, e.Base, flag, interp, logf(e.Log), e.Log, c.version, fetch, export); dir != "" {
		model, err := fetch()
		if err != nil {
			return f, err
		}
		f.Set("export", dir)
		f.Set("onnx", model)
		return f, nil
	}
	ort, builtin := imagedev.CPURoute(e.Ctx, m, flag, interp, threads(m), quiet)
	if ort && c.programOnly {
		// a prebuilt Core ML package, here or in the shared layers, before the ONNX file
		if model, program, err := c.convert(llm.WithFetchOnly(e.Ctx), m, e.Base, llm.BackendCoreML, e.Log); err == nil && model == "" {
			f.Set("program", program)
			f.SetValue("runtime", "hwnpu")
			return f, nil
		}
	}
	if ort {
		model, err := fetch()
		if err != nil {
			return f, err
		}
		f.Set("onnx", model)
		return f, nil
	}
	backend := e.NPU.Format
	if interp || builtin {
		backend = llm.BackendCoreML // what the CPU interpreter and hwnpu's executor run
	}
	if backend == "" {
		return f, fmt.Errorf("this machine's NPU runs neither Core AI programs nor OpenVINO IR (--device cpu runs %s on the CPU)", c.what)
	}
	model, program, err := c.convert(e.Ctx, m, e.Base, backend, e.Log)
	if err != nil {
		return f, err
	}
	f.Set("onnx", model)
	f.Set("program", program)
	return f, nil
}

// load loads the model where prepare made it ready for, and says where it runs: its device (npu
// or cpu), and the words of the log ("on the CPU (ONNX Runtime, 1 threads)"). The device closes
// with closeDev.
func (c cnn[T]) load(ctx context.Context, f airole.Files, e airole.LoadEnv) (t T, device, where string, closeDev func(), err error) {
	m := e.Def
	n := threads(m)
	if dir := f.Path("export"); dir != "" {
		dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
		if err != nil {
			return t, "", "", nil, err
		}
		if t, err = c.openNPU(ctx, dev, dir, f.Path("onnx")); err != nil {
			dev.Close()
			return t, "", "", nil, err
		}
		return t, "npu", fmt.Sprintf("on AMD's NPU (%s)", filepath.Base(dir)), func() { dev.Close() }, nil
	}
	interp := interpreter(m)
	ort, builtin := imagedev.CPURoute(ctx, named(m), imagedev.DeviceFlag(m), interp, n, logf(e.Log))
	if ort && f.Values["runtime"] == "hwnpu" {
		// prepared with a prebuilt Core ML package in place of the ONNX file (programOnly)
		ort, builtin = false, true
		milcpu.Threads = max(1, n)
	}
	if ort {
		if t, err = c.openCPU(f.Path("onnx"), n); err != nil {
			return t, "", "", nil, err
		}
		return t, "cpu", fmt.Sprintf("on the CPU (ONNX Runtime, %d threads)", n), func() {}, nil
	}
	program := f.Path("program")
	if program == "" {
		return t, "", "", nil, errors.New("prepared for ONNX Runtime, which is not here now: prepare it again")
	}
	cpu := interp || builtin
	dev, _, closeDev, err := imagedev.OpenProgramDevice(ctx, cpu, e.Cache, e.NPU.Device, c.what)
	if err != nil {
		return t, "", "", nil, err
	}
	if t, err = c.open(ctx, dev, f.Path("onnx"), program); err != nil {
		closeDev()
		return t, "", "", nil, err
	}
	device = "npu"
	if cpu {
		device = "cpu"
	}
	return t, device, imagedev.Placement(cpu, c.onNPU(t)), closeDev, nil
}
