package sound

import (
	"context"
	"fmt"
	"io"
	"strconv"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/onnxrt"
)

// The pitch and the notes are small ONNX models (SwiftF0, Basic Pitch) that hwnpu converts for
// Core ML and OpenVINO. Where one runs is picked as it is loaded (chat.ConvertedDevice): the
// definition's device (the app's Definition.Device), else HWNPU_DEVICE, else models.yaml's;
// an NPU with no converted form of it (AMD's, or none) runs it on the CPU. On the CPU it runs
// with ONNX Runtime when its library is here (HWNPU_ONNXRUNTIME), else on hwnpu's own executor
// (internal/milcpu) on the converted program.
//
// The settings their definitions may give (ai.Definition):
//
//	settings: threads  the CPU's threads (1: the least CPU time and energy, the default)
//	          cpu      hwnpu: on the CPU, hwnpu's own executor on all the cores, ONNX Runtime
//	                   or not (for checking a conversion)

// convertedFiles are the files a converted model's preparation makes: the ONNX file, and its
// program for the CPU interpreter and for this machine's NPU.
func convertedPrepare(e airole.PrepareEnv, prepare func(backend string) (model, program string, err error)) (f airole.Files, err error) {
	// the CPU interpreter's (Core ML packages), which runs it on the CPU without ONNX Runtime
	model, program, err := prepare(llm.BackendCoreML)
	if err != nil {
		return f, err
	}
	f.Set("onnx", model)
	f.Set("program-"+llm.BackendCoreML, program)
	if format, err := programFormat(e.NPU); err == nil && format != llm.BackendCoreML {
		if _, program, err = prepare(format); err != nil {
			return f, err
		}
		f.Set("program-"+format, program)
	}
	return f, nil
}

// route is where a converted model is loaded.
type route struct {
	// device is npu or cpu; runtime onnxruntime, hwnpu (its own executor on the CPU) or npu
	device, runtime string
	threads         int
	// line is the log's line of it
	line string
}

// Where says where the model runs, for a log.
func (r route) Where() string {
	switch r.runtime {
	case "onnxruntime":
		return fmt.Sprintf("on the CPU (ONNX Runtime, %d threads)", max(1, r.threads))
	case "hwnpu":
		return "on the CPU interpreter"
	}
	return ""
}

// convertedRoute is where model m (its definition, of the model named name) runs here.
func convertedRoute(m chat.Model, name string, n airole.NPU) (route, error) {
	threads := 1
	if s := m.Settings["threads"]; s != "" {
		t, err := strconv.Atoi(s)
		if err != nil || t < 0 {
			return route{}, fmt.Errorf("sound: %s's threads %q: a number", name, s)
		}
		threads = t
	}
	if m.Settings["cpu"] == "hwnpu" {
		// all the cores unless the definition says
		if m.Settings["threads"] == "" {
			threads = 0
		}
		return route{device: "cpu", runtime: "hwnpu", threads: threads}, nil
	}
	device, why, err := chat.ConvertedDevice(m, n.Vendor, m.Device[n.Vendor])
	if err != nil {
		return route{device: "npu", runtime: "npu", line: fmt.Sprintf("%v; the NPU", err)}, nil
	}
	switch {
	case device != "cpu":
		return route{device: device, runtime: "npu", line: fmt.Sprintf("%s on the %s (%s)", name, device, why)}, nil
	case onnxrt.Available():
		return route{device: "cpu", runtime: "onnxruntime", threads: threads, line: fmt.Sprintf("%s on the CPU with ONNX Runtime (%s)", name, why)}, nil
	}
	return route{device: "cpu", runtime: "hwnpu", threads: max(1, threads),
		line: fmt.Sprintf("%s on the CPU with hwnpu's own executor (%s; ONNX Runtime, faster, is not here: HWNPU_ONNXRUNTIME)", name, why)}, nil
}

// convertedOpen loads a converted model on its route: open with the ONNX file on ONNX Runtime,
// else with the program of the CPU interpreter (dev nil) or of the NPU (dev open).
type convertedOpener struct {
	cpu func(model string, threads int) (io.Closer, error)
	npu func(ctx context.Context, dev *hwnpu.Device, model, program string) (io.Closer, error)
}

// convertedLoad loads the model of the files f on r: the loaded model, and the device it holds
// (nil on the CPU), which the caller closes after it.
func convertedLoad(ctx context.Context, f airole.Files, e airole.LoadEnv, r route, o convertedOpener) (io.Closer, *hwnpu.Device, error) {
	if r.line != "" {
		logf(e.Log)("%s", r.line)
	}
	switch r.runtime {
	case "onnxruntime":
		m, err := o.cpu(f.Path("onnx"), r.threads)
		return m, nil, err
	case "hwnpu":
		milcpu.Threads = r.threads
		m, err := o.npu(ctx, nil, f.Path("onnx"), f.Path("program-"+llm.BackendCoreML))
		return m, nil, err
	}
	format, err := programFormat(e.NPU)
	if err != nil {
		return nil, nil, err
	}
	program := f.Path("program-" + format)
	if program == "" {
		return nil, nil, fmt.Errorf("sound: %s has no program for %s prepared", e.Model, format)
	}
	dev, err := openNPU(ctx, e.NPU, e.Cache)
	if err != nil {
		return nil, nil, err
	}
	m, err := o.npu(ctx, dev, f.Path("onnx"), program)
	if err != nil {
		dev.Close()
		return nil, nil, err
	}
	return m, dev, nil
}
