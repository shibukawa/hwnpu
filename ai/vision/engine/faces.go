package engine

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"os"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/onnxrt"
	"github.com/shibukawa/hwnpu/internal/vision/face"
)

func init() {
	r := bridge.Roles["faces"]
	r.Usable = usableFaces
	r.Prepare = prepareFaces
	r.Load = loadFaces
}

// usableFaces: prepared for the CPU's runtime that was here, Prepare decides again once ONNX
// Runtime (HWNPU_ONNXRUNTIME) is gone (hwnpu's own executor in its place) or has come.
func usableFaces(f airole.Files) error {
	if f.Values["device"] != "cpu" {
		return nil
	}
	switch ort := onnxrt.Available(); {
	case f.Values["runtime"] != "hwnpu" && !ort:
		return errors.New("it runs on the CPU with ONNX Runtime, whose library does not load (HWNPU_ONNXRUNTIME)")
	case f.Values["runtime"] == "hwnpu" && ort && f.Values["prebuilt"] == "":
		return errors.New("it runs on hwnpu's own executor, and ONNX Runtime is here now")
	}
	return nil
}

func prepareFaces(e airole.PrepareEnv) (f airole.Files, err error) {
	m := e.Model
	flag := imagedev.DeviceFlag(m)
	if dir := program(m, "export"); dir != "" {
		if !face.IsExport(dir) {
			return f, fmt.Errorf("%s is not an export of tools/amd/iron_yunet.py", dir)
		}
		if flag == "cpu" {
			return f, fmt.Errorf("an export of tools/amd/iron_yunet.py (%s) runs on AMD's NPU, not the CPU", dir)
		}
		f.SetValue("device", "npu")
		f.Set("export", dir)
		return f, nil
	}
	// where the definitions say it runs on this machine's NPU (models.yaml's device: the CPU
	// beside AMD's NPU, where it takes a tenth of the time and energy)
	device, why, err := chat.ChooseDevice(m, e.NPU.Vendor, flag)
	if path := program(m, "onnx"); path != "" {
		why = "" // a file of the app's: not said, as the command never said it
		if flag == "" {
			device = "cpu" // an ONNX file runs on the CPU unless the device is given
		}
	}
	if err != nil {
		return f, err
	}
	if device == "npu" && e.NPU.Vendor != "amd" && e.NPU.Vendor != "apple" && e.NPU.Vendor != "intel" {
		// AMD's NPU has YuNet's graph (iron_yunet.py), Apple's and Intel's the ONNX graph
		// converted; no graph of it for another NPU
		if flag == "npu" {
			return f, fmt.Errorf("%s runs on AMD's, Apple's and Intel's NPU (this machine's NPU: %s); --device cpu runs it on the CPU", shown(m.Name), orNone(e.NPU.Vendor))
		}
		device, why = "cpu", fmt.Sprintf("no %s graph for this NPU (%s)", shown(m.Name), orNone(e.NPU.Vendor))
	}
	// the CPU runs the ONNX file with ONNX Runtime where its library loads, else the program
	// converted from it on hwnpu's own executor (internal/milcpu): slower than ONNX Runtime
	// and still far less time and energy than AMD's NPU (docs/amd-npu-models.md 17)
	builtin := device == "cpu" && !onnxrt.Available()
	if builtin {
		f.SetValue("runtime", "hwnpu")
		why = strings.TrimPrefix(why+"; hwnpu's own executor: ONNX Runtime, faster, is not here (HWNPU_ONNXRUNTIME)", "; ")
	}
	f.SetValue("device", device)
	f.SetValue("why", why)
	path := program(m, "onnx")
	if device == "npu" && e.NPU.AMD() {
		if path != "" {
			return f, fmt.Errorf("AMD's NPU runs an export of tools/amd/iron_yunet.py, not the ONNX file %s", path)
		}
		dir, err := chat.AMDPictureExport(e.Ctx, e.Base, "tools/amd/iron_yunet.py", 0, e.Log)
		if err != nil {
			return f, err
		}
		iron.ShareExport(dir, logf(e.Log))
		f.Set("export", dir)
		return f, nil
	}
	fetchONNX := func() (string, error) { return chat.Fetch(e.Ctx, m.Source(), imagedev.YuNetFile, e.Log) }
	if device == "cpu" && path == "" {
		// the Core ML programs of every size made before (hwnpu-dev models package
		// YuNet/coreml), here or in the shared layers: run on hwnpu's own executor even where
		// ONNX Runtime is, and the ONNX file is not downloaded
		dir := chat.ModelDir(e.Base)
		if _, err := face.PreparePrograms(llm.WithFetchOnly(e.Ctx), llm.BackendCoreML, fetchONNX, dir, face.CPUSizes); err == nil {
			f.SetValue("runtime", "hwnpu")
			f.SetValue("prebuilt", "1")
			f.SetValue("why", strings.TrimPrefix(f.Values["why"]+"; hwnpu's own executor on the prebuilt programs", "; "))
			f.Set("programs", dir)
			return f, nil
		}
	}
	if builtin {
		// the Core ML programs of every size now, which the executor runs (as the NPU's below);
		// the ONNX file downloaded only for one converted here
		dir := chat.ModelDir(e.Base)
		if path == "" {
			if path, err = face.PreparePrograms(e.Ctx, llm.BackendCoreML, fetchONNX, dir, face.CPUSizes); err != nil {
				return f, err
			}
		} else if err := face.ConvertPrograms(e.Ctx, llm.BackendCoreML, path, dir, face.CPUSizes); err != nil {
			return f, err
		}
		f.Set("onnx", path)
		f.Set("programs", dir)
		return f, nil
	}
	if path == "" {
		if path, err = fetchONNX(); err != nil {
			return f, err
		}
	}
	f.Set("onnx", path)
	if device == "npu" {
		// Apple's and Intel's NPU run the ONNX graph converted (at its first load) into the
		// model's directory of programs, made now: the store finds a use whose files are all
		// there, and the directory is empty until that load
		dir := chat.ModelDir(e.Base)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return f, err
		}
		// every size's program now, not at the first frame of each: after "models
		// prepare" a run needs no network nor conversion (Warm compiles them)
		if err := face.ConvertPrograms(e.Ctx, e.NPU.Format, path, dir, face.CPUSizes); err != nil {
			return f, err
		}
		f.Set("programs", dir)
		f.SetValue("format", e.NPU.Format)
	}
	return f, nil
}

func loadFaces(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	var (
		d   *face.Detector
		err error
	)
	if why := f.Values["why"]; why != "" {
		logf(e.Log)("%s on the %s (%s)", shown(e.Model), f.Values["device"], why)
	}
	n := threads(e.Def)
	switch {
	case f.Values["device"] == "cpu" && f.Values["runtime"] == "hwnpu":
		milcpu.Threads = n
		if d, err = face.OpenProgram(ctx, nil, llm.BackendCoreML, f.Path("onnx"), f.Path("programs")); err != nil {
			return nil, err
		}
		d.Device = "cpu"
	case f.Values["device"] == "cpu":
		if d, err = face.OpenCPU(f.Path("onnx"), n); err != nil {
			return nil, fmt.Errorf("%s runs on the CPU with ONNX Runtime here: %w", shown(e.Model), err)
		}
	default:
		dev, err := openDevice(ctx, e)
		if err != nil {
			return nil, err
		}
		if dir := f.Path("export"); dir != "" {
			d, err = face.Open(ctx, dev, dir)
		} else {
			d, err = face.OpenProgram(ctx, dev, f.Values["format"], f.Path("onnx"), f.Path("programs"))
		}
		if err != nil {
			dev.Close()
			return nil, err
		}
		d.OnClose = func() { dev.Close() }
	}
	return bridge.Wrap["faces"](&faceDetector{d: d, name: e.Model, log: e.Log, score: d.Score, nms: d.NMS, threads: n}), nil
}

func orNone(v string) string {
	if v == "" {
		return "none"
	}
	return v
}

// faceDetector is a face model of the engine (vision.FaceDetector's).
type faceDetector struct {
	mu         sync.Mutex
	d          *face.Detector
	name       string
	log        io.Writer
	score, nms float32
	threads    int
}

func (d *faceDetector) Model() string  { return d.name }
func (d *faceDetector) Device() string { return d.d.Device }

func (d *faceDetector) Where() string {
	if d.d.Device == "cpu" {
		return fmt.Sprintf("the CPU (ONNX Runtime, %d threads)", d.threads)
	}
	return "the NPU"
}

func (d *faceDetector) Warm(ctx context.Context) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.d.Warm(ctx)
}

func (d *faceDetector) DetectWith(ctx context.Context, img image.Image, o vision.FaceOptions) ([]vision.Face, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	if o.Size != [2]int{} && (o.Size[0] <= 0 || o.Size[1] <= 0 || o.Size[0]%32 != 0 || o.Size[1]%32 != 0) {
		return nil, fmt.Errorf("vision: a face model's size %dx%d is not of multiples of 32", o.Size[0], o.Size[1])
	}
	d.d.Score, d.d.NMS, d.d.Size = d.score, d.nms, o.Size
	if o.Score != 0 {
		d.d.Score = o.Score
	}
	if o.NMS != 0 {
		d.d.NMS = o.NMS
	}
	d.d.Logf = nil
	if o.Log {
		d.d.Logf = logf(d.log)
	}
	found, err := d.d.Detect(ctx, img)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	out := make([]vision.Face, len(found))
	for i, f := range found {
		pt := func(p [2]float32) vision.Point {
			return vision.Point{X: float64(p[0]) + float64(b.Min.X), Y: float64(p[1]) + float64(b.Min.Y)}
		}
		x, y := float64(f.X)+float64(b.Min.X), float64(f.Y)+float64(b.Min.Y)
		out[i] = vision.Face{
			Box:      vision.Box{Min: vision.Point{X: x, Y: y}, Max: vision.Point{X: x + float64(f.W), Y: y + float64(f.H)}},
			Score:    float64(f.Score),
			RightEye: pt(f.Landmarks[0]), LeftEye: pt(f.Landmarks[1]), Nose: pt(f.Landmarks[2]),
			MouthRight: pt(f.Landmarks[3]), MouthLeft: pt(f.Landmarks[4]),
		}
	}
	return out, nil
}

func (d *faceDetector) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.d.Close()
}
