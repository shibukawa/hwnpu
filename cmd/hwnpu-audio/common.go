package main

import (
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/onnxrt"

	"cmp"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu/internal/chat"
)

// modelsDir is --dir, or where the models are kept by default (as hwnpu-chat's --dir).
func modelsDir(dir string) string { return cmp.Or(dir, chat.ModelsDir()) }

// cacheDir is --cache, or hwnpu in the user cache directory.
func cacheDir(dir string) string {
	if dir != "" {
		return dir
	}
	if d, err := os.UserCacheDir(); err == nil {
		return filepath.Join(d, "hwnpu")
	}
	return ""
}

// placement is how a log line says where a model runs.
// cpuRoute is where an ONNX model m runs (chat.ConvertedDevice) and logs it: ort, ONNX
// Runtime on the CPU when its library is here; builtin, hwnpu's own CPU executor on the
// converted program (internal/milcpu, on threads goroutines) when it is not, or with --cpu
// (interp: all the cores); neither, the NPU.
func cpuRoute(m chat.Model, flag string, interp bool, threads int, logf func(string, ...any)) (ort, builtin bool) {
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

func placement(cpu, onNPU bool) string {
	switch {
	case cpu:
		return "on the CPU interpreter"
	case onNPU:
		return "all on the NPU"
	}
	return "NOT all on the NPU"
}

// resolveKind is the model of the definitions --model means for kind (with --models' files read
// first); when there is none, what there is is listed before the error.
func resolveKind(name string, kind chat.Kind, models string, lw io.Writer) (chat.Model, error) {
	if err := chat.LoadUserModels(models, lw); err != nil {
		return chat.Model{}, err
	}
	m, err := chat.ResolveFor(name, kind, lw)
	if e := (*chat.NoModelError)(nil); errors.As(err, &e) {
		err = fmt.Errorf("no %s model, family or use %q (hwnpu-audio list %s shows them)", kind, name, kind)
		fmt.Fprintf(os.Stderr, "no %s model %q; the %s models of the definitions:\n\n", kind, name, kind)
		chat.List(os.Stderr, chat.ModelsDir(), kind)
		fmt.Fprintln(os.Stderr)
	}
	return m, err
}
