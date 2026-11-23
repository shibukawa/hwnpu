package vision

import (
	"context"
	"errors"
	"fmt"
	"image"
	"io"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/onnxrt"
	"github.com/shibukawa/hwnpu/internal/vision/face"
)

// Faces is the use of a model that finds faces (YuNet) for an app: a model, a family or a use
// of the definitions ("face-detect").
func Faces(name string) ai.Use[*FaceDetector] {
	return ai.Define[*FaceDetector](facesRole, name)
}

// The definition of a face model may give the settings threads, the programs export (a
// directory of tools/amd/iron_yunet.py: AMD's NPU) or onnx (YuNet's ONNX file: the CPU, or
// converted for Apple's and Intel's NPU), and the device (common.go).
var facesRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &FaceDetector{remote: h} },
	Name:   "faces",
	// 1: the programs of every size made at Prepare; 2: hwnpu's own executor on the CPU where
	// ONNX Runtime is not here (AMD's NPU before)
	Revision: 2,
	// prepared for the CPU's runtime that was here: Prepare decides again once ONNX Runtime
	// (HWNPU_ONNXRUNTIME) is gone (hwnpu's own executor in its place) or has come
	Usable: func(f airole.Files) error {
		if f.Values["device"] != "cpu" {
			return nil
		}
		switch ort := onnxrt.Available(); {
		case f.Values["runtime"] != "hwnpu" && !ort:
			return errors.New("it runs on the CPU with ONNX Runtime, whose library does not load (HWNPU_ONNXRUNTIME)")
		case f.Values["runtime"] == "hwnpu" && ort:
			return errors.New("it runs on hwnpu's own executor, and ONNX Runtime is here now")
		}
		return nil
	},
	Kinds: []chat.Kind{chat.KindFace},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
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
		if path == "" {
			if path, err = chat.Fetch(e.Ctx, m.Source(), imagedev.YuNetFile, e.Log); err != nil {
				return f, err
			}
		}
		f.Set("onnx", path)
		if builtin {
			// the Core ML programs of every size now, which the executor runs (as the NPU's below)
			dir := chat.ModelDir(e.Base)
			if err := face.ConvertPrograms(e.Ctx, llm.BackendCoreML, path, dir, face.CPUSizes); err != nil {
				return f, err
			}
			f.Set("programs", dir)
			return f, nil
		}
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
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
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
		return &FaceDetector{d: d, name: e.Model, log: e.Log, score: d.Score, nms: d.NMS, threads: n}, nil
	},
}

func orNone(v string) string {
	if v == "" {
		return "none"
	}
	return v
}

// Warm loads the program of every size once, which compiles it for the NPU (ai.Warmer: the
// load of "models prepare"), so that a camera of any size does not wait for that.
func (d *FaceDetector) Warm(ctx context.Context) error {
	if d.remote != nil {
		return airpc.Call0(d.remote, ctx, "Warm")
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.d.Warm(ctx)
}

// Face is a face found in a picture, in the picture's pixels.
type Face struct {
	// Box is the face's box.
	Box Box
	// Score is how sure the model is, 0 to 1.
	Score float64
	// RightEye and LeftEye are the person's own right and left eye (the right eye is on the
	// picture's left), Nose the tip of the nose, MouthRight and MouthLeft the corners of the
	// mouth.
	RightEye, LeftEye, Nose, MouthRight, MouthLeft Point
}

// FaceDetector finds the faces in pictures. It runs one picture at a time; others wait.
type FaceDetector struct {
	remote     *airpc.Handle // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
	mu         sync.Mutex
	d          *face.Detector
	name       string
	log        io.Writer
	score, nms float32
	threads    int
}

// FaceOptions are how faces are found. The zero value of a field takes its default.
type FaceOptions struct {
	// Score is the least score a face is kept with (default 0.9, OpenCV's; a negative value
	// keeps every face), NMS the overlap (intersection over union) above which the lower of two
	// faces goes (default 0.3).
	Score, NMS float32
	// Size is the model's input size every picture runs at, multiples of 32 (default: of the
	// model's sizes the one the picture is scaled down least to fit).
	Size [2]int
	// Log writes each picture's sizes and times to the Runtime's log (ai.WithLog).
	Log bool
}

// Model is the name of the model in the definitions.
func (d *FaceDetector) Model() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Model")
	}
	return d.name
}

// Device is where the model runs: npu or cpu (YuNet is a small graph that the definitions put
// on the CPU beside AMD's NPU).
func (d *FaceDetector) Device() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Device")
	}
	return d.d.Device
}

// Where says where the model runs, for a log: "the CPU (ONNX Runtime, 1 threads)", "the NPU".
func (d *FaceDetector) Where() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Where")
	}
	if d.d.Device == "cpu" {
		return fmt.Sprintf("the CPU (ONNX Runtime, %d threads)", d.threads)
	}
	return "the NPU"
}

// Detect finds the faces in a picture, the surest first.
func (d *FaceDetector) Detect(ctx context.Context, img image.Image) ([]Face, error) {
	return d.DetectWith(ctx, img, FaceOptions{})
}

// DetectWith is Detect with options.
func (d *FaceDetector) DetectWith(ctx context.Context, img image.Image, o FaceOptions) ([]Face, error) {
	if d.remote != nil {
		return airpc.Call1[[]Face](d.remote, ctx, "DetectWith", img, o)
	}
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
	out := make([]Face, len(found))
	for i, f := range found {
		pt := func(p [2]float32) Point {
			return Point{float64(p[0]) + float64(b.Min.X), float64(p[1]) + float64(b.Min.Y)}
		}
		x, y := float64(f.X)+float64(b.Min.X), float64(f.Y)+float64(b.Min.Y)
		out[i] = Face{
			Box:      Box{Point{x, y}, Point{x + float64(f.W), y + float64(f.H)}},
			Score:    float64(f.Score),
			RightEye: pt(f.Landmarks[0]), LeftEye: pt(f.Landmarks[1]), Nose: pt(f.Landmarks[2]),
			MouthRight: pt(f.Landmarks[3]), MouthLeft: pt(f.Landmarks[4]),
		}
	}
	return out, nil
}

// Found is what was found in a frame: its faces, the surest first (none for a frame without
// one), and the frame's bounds, which say where in the camera's view they are.
type Found struct {
	Frame image.Rectangle
	Faces []Face
	// Took is how long finding them took.
	Took time.Duration
}

// DetectStream finds the faces in each picture of a stream (a camera's frames), in a chunk of
// the picture's times. A stream whose pictures come faster than they are read is to be read
// through flow.Latest, or over an edge of flow.Newest.
func (d *FaceDetector) DetectStream(ctx context.Context, s flow.Stream[image.Image]) flow.Stream[Found] {
	return flow.Map(s, func(c flow.Chunk[image.Image]) (Found, error) {
		t0 := time.Now()
		faces, err := d.Detect(ctx, c.Data)
		return Found{Frame: c.Data.Bounds(), Faces: faces, Took: time.Since(t0)}, err
	})
}

// FacesNode is the node of a face detector in a network or a call (flow, ai.App.From,
// ai.NewCall): pictures in, what was found in each out.
func FacesNode(u ai.Use[*FaceDetector]) *flow.Node[image.Image, Found] {
	return ai.Node(u, func(ctx context.Context, d *FaceDetector, in flow.Stream[image.Image]) flow.Stream[Found] {
		return d.DetectStream(ctx, in)
	})
}

// HeadNode is a node of the program that works out where the nearest head is (NearestHead) in
// what a face detector found: the node after FacesNode in a network that follows a head. It
// passes nothing on for a frame without a face.
func HeadNode(o HeadOptions) *flow.Node[Found, Head] {
	return flow.Each("head", func(c flow.Chunk[Found], emit func(flow.Chunk[Head])) error {
		if h, ok := NearestHead(c.Data.Faces, c.Data.Frame, o); ok {
			emit(flow.Chunk[Head]{Data: h, Start: c.Start, Len: c.Len, Final: true})
		}
		return nil
	})
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (d *FaceDetector) Close() error {
	if d.remote != nil {
		return d.remote.Release() // the Runtime in hwnpud unloads it
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.d.Close()
}

func init() { airole.Register(facesRole) }
