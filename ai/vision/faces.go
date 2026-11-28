package vision

import (
	"context"
	"image"
	"io"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
)

// Faces is the use of a model that finds faces (YuNet) for an app: a model, a family or a use
// of the definitions ("face-detect").
func Faces(name string) ai.Use[*FaceDetector] {
	return ai.Define[*FaceDetector](facesRole, name)
}

// The definition of a face model may give the settings threads, the programs export (a
// directory of tools/amd/iron_yunet.py: AMD's NPU) or onnx (YuNet's ONNX file: the CPU, or
// converted for Apple's and Intel's NPU), and the device (common.go). facesRole's Prepare, Load
// and Usable are the engine's (ai/vision/engine).
var facesRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &FaceDetector{p: airpc.RemoteProxy[faceDetectorImpl](h)} },
	Name:   "faces",
	// 1: the programs of every size made at Prepare; 2: hwnpu's own executor on the CPU where
	// ONNX Runtime is not here (AMD's NPU before)
	Revision: 2,
	Kinds:    []catalog.Kind{catalog.KindFace},
}

// Warm loads the program of every size once, which compiles it for the NPU (ai.Warmer: the
// load of "models prepare"), so that a camera of any size does not wait for that.
func (d *FaceDetector) Warm(ctx context.Context) error {
	return airpc.ProxyCall0(d.p, ctx, "Warm", faceDetectorImpl.Warm)
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
	p airpc.Proxy[faceDetectorImpl]
}

// faceDetectorImpl is a face model of the engine in this process.
type faceDetectorImpl interface {
	Model() string
	Device() string
	Where() string
	DetectWith(ctx context.Context, img image.Image, o FaceOptions) ([]Face, error)
	Warm(ctx context.Context) error
	Close() error
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
func (d *FaceDetector) Model() string { return airpc.ProxyValue(d.p, "Model", faceDetectorImpl.Model) }

// Device is where the model runs: npu or cpu (YuNet is a small graph that the definitions put
// on the CPU beside AMD's NPU).
func (d *FaceDetector) Device() string {
	return airpc.ProxyValue(d.p, "Device", faceDetectorImpl.Device)
}

// Where says where the model runs, for a log: "the CPU (ONNX Runtime, 1 threads)", "the NPU".
func (d *FaceDetector) Where() string { return airpc.ProxyValue(d.p, "Where", faceDetectorImpl.Where) }

// Detect finds the faces in a picture, the surest first.
func (d *FaceDetector) Detect(ctx context.Context, img image.Image) ([]Face, error) {
	return d.DetectWith(ctx, img, FaceOptions{})
}

// DetectWith is Detect with options.
func (d *FaceDetector) DetectWith(ctx context.Context, img image.Image, o FaceOptions) ([]Face, error) {
	return airpc.ProxyCall1(d.p, ctx, "DetectWith", func(l faceDetectorImpl) ([]Face, error) {
		return l.DetectWith(ctx, img, o)
	}, img, o)
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
func (d *FaceDetector) Close() error { return d.p.Close() }

func init() {
	airole.Register(facesRole)
	bridge.Roles[facesRole.Name] = facesRole
	bridge.Wrap[facesRole.Name] = func(l any) io.Closer { return &FaceDetector{p: airpc.LocalProxy(l.(faceDetectorImpl))} }
}
