package vision

import (
	"context"
	"fmt"
	"image"
	"io"
	"os"
	"sync"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/vision/face"
)

// Faces is the use of a model that finds faces (YuNet) for an app: a model, a family or a use
// of the definitions ("face-detect").
func Faces(name string) ai.Use[*FaceDetector] {
	return ai.Define[*FaceDetector](facesRole, name)
}

// faceONNX is the file of YuNet's repository.
const faceONNX = "face_detection_yunet_2023mar.onnx"

var facesRole = &airole.Role{
	Name:     "faces",
	Revision: 1, // 1: the programs of every size made at Prepare
	Kinds:    []chat.Kind{chat.KindFace},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		m := e.Model
		// where the definitions say it runs on this machine's NPU (models.yaml's device: the CPU
		// beside AMD's NPU, where it takes a tenth of the time and energy)
		device, _, err := chat.ChooseDevice(m, e.NPU.Vendor, "")
		if err != nil {
			return f, err
		}
		switch e.NPU.Vendor {
		case "amd", "apple", "intel":
		default:
			device = "cpu" // no graph of it for another NPU
		}
		f.SetValue("device", device)
		if device == "npu" && e.NPU.AMD() {
			dir, err := chat.AMDPictureExport(e.Ctx, m, "tools/amd/iron_yunet.py", 0, e.Log)
			if err != nil {
				return f, err
			}
			iron.ShareExport(dir, logf(e.Log))
			f.Set("export", dir)
			return f, nil
		}
		path, err := chat.Fetch(e.Ctx, m.Source(), faceONNX, e.Log)
		if err != nil {
			return f, err
		}
		f.Set("onnx", path)
		if device == "npu" {
			// Apple's and Intel's NPU run the ONNX graph converted (at its first load) into the
			// model's directory of programs, made now: the store finds a use whose files are all
			// there, and the directory is empty until that load
			if err := os.MkdirAll(e.Dir, 0o755); err != nil {
				return f, err
			}
			// every size's program now, not at the first frame of each: after "models
			// prepare" a run needs no network nor conversion (Warm compiles them)
			if err := face.ConvertPrograms(e.Ctx, e.NPU.Format, path, e.Dir, face.CPUSizes); err != nil {
				return f, err
			}
			f.Set("programs", e.Dir)
			f.SetValue("format", e.NPU.Format)
		}
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		var (
			d   *face.Detector
			err error
		)
		switch {
		case f.Values["device"] == "cpu":
			if d, err = face.OpenCPU(f.Path("onnx"), 1); err != nil {
				return nil, fmt.Errorf("%s runs on the CPU with ONNX Runtime here: %w", e.Model, err)
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
		return &FaceDetector{d: d, name: e.Model}, nil
	},
}

// Warm loads the program of every size once, which compiles it for the NPU (ai.Warmer: the
// load of "models prepare"), so that a camera of any size does not wait for that.
func (d *FaceDetector) Warm(ctx context.Context) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.d.Warm(ctx)
}

// Face is a face found in a picture, in the picture's pixels.
type Face struct {
	// Box is the face's box.
	Box image.Rectangle
	// Score is how sure the model is, 0 to 1.
	Score float64
	// RightEye and LeftEye are the person's own right and left eye (the right eye is on the
	// picture's left), Nose the tip of the nose, MouthRight and MouthLeft the corners of the
	// mouth.
	RightEye, LeftEye, Nose, MouthRight, MouthLeft Point
}

// Point is a point of a picture, in its pixels.
type Point struct{ X, Y float64 }

// FaceDetector finds the faces in pictures. It runs one picture at a time; others wait.
type FaceDetector struct {
	mu   sync.Mutex
	d    *face.Detector
	name string
}

// Model is the name of the model in the definitions.
func (d *FaceDetector) Model() string { return d.name }

// Device is where the model runs: npu or cpu (YuNet is a small graph that the definitions put
// on the CPU beside AMD's NPU).
func (d *FaceDetector) Device() string { return d.d.Device }

// Detect finds the faces in a picture, the surest first.
func (d *FaceDetector) Detect(ctx context.Context, img image.Image) ([]Face, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
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
		out[i] = Face{
			Box:      image.Rect(int(f.X), int(f.Y), int(f.X+f.W+0.5), int(f.Y+f.H+0.5)).Add(b.Min),
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
}

// DetectStream finds the faces in each picture of a stream (a camera's frames), in a chunk of
// the picture's times. A stream whose pictures come faster than they are read is to be read
// through flow.Latest, or over an edge of flow.Newest.
func (d *FaceDetector) DetectStream(ctx context.Context, s flow.Stream[image.Image]) flow.Stream[Found] {
	return flow.Map(s, func(c flow.Chunk[image.Image]) (Found, error) {
		faces, err := d.Detect(ctx, c.Data)
		return Found{Frame: c.Data.Bounds(), Faces: faces}, err
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
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.d.Close()
}
