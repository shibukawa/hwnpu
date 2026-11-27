package vision

import (
	"context"
	"image"
	"io"
	"slices"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/visionwire"
)

// Detect is the use of a model that finds objects and their labels (RTMDet, COCO's 80 classes)
// for an app: a model, a family or a use of the definitions ("detect", "rtmdet:tiny").
func Detect(name string) ai.Use[*Detector] {
	return ai.Define[*Detector](detectRole, name)
}

// COCO are the classes of the COCO models (Detector.Classes): person, bicycle, car, ...
var COCO = slices.Clone(visionwire.COCO)

// The definition of a detection model may give the settings threads and cpu, and the device
// (common.go). detectRole's Prepare and Load are the engine's (ai/vision/engine).
var detectRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &Detector{remote: h} },
	Name:   "detect",
	Kinds:  []catalog.Kind{catalog.KindDetect},
}

// Object is an object found in a picture, in its pixels.
type Object struct {
	Box Box
	// Class is the class's index in Detector.Classes, Label its name.
	Class int
	Label string
	// Score is how sure the model is, 0 to 1.
	Score float64
}

// Objects are the objects found in a picture, the surest first, and what finding them took.
type Objects struct {
	Objects []Object
	// Took is how long it all took: Resize the picture's resizing, Network the model's network
	// (on the NPU, or the CPU), Host decoding the boxes and suppressing the overlaps.
	Took, Resize, Network, Host time.Duration
}

// DetectOptions are how objects are found. The zero value of a field takes its default.
type DetectOptions struct {
	// Score is the least score of an object kept (default 0.3; a negative value keeps every
	// one), NMS the overlap (intersection over union) above which the lower of two objects of a
	// class goes (default 0.65, MMDetection's).
	Score, NMS float32
	// Classes keeps the objects of these classes only (Detector.Classes' names); none keeps all.
	Classes []string
}

// Detector finds the objects in pictures. It runs one picture at a time; others wait.
type Detector struct {
	remote *airpc.Handle // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
	local  detectorImpl  // the engine's (ai/vision/engine), when it is here
}

// detectorImpl is a detection model of the engine in this process.
type detectorImpl interface {
	Model() string
	Device() string
	Where() string
	Classes() []string
	Detect(ctx context.Context, img image.Image, o DetectOptions) (*Objects, error)
	Close() error
}

// Model is the name of the model in the definitions.
func (d *Detector) Model() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Model")
	}
	return d.local.Model()
}

// Device is where the model's network runs: npu or cpu.
func (d *Detector) Device() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Device")
	}
	return d.local.Device()
}

// Where says where the model runs, for a log: "on AMD's NPU (rtmdet-m.amd-npu2)", "on the CPU
// (ONNX Runtime, 1 threads)", "all on the NPU", "on the CPU interpreter".
func (d *Detector) Where() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Where")
	}
	return d.local.Where()
}

// Classes are the names of the model's classes (COCO's).
func (d *Detector) Classes() []string {
	if d.remote != nil {
		return airpc.Value[[]string](d.remote, "Classes")
	}
	return d.local.Classes()
}

// Detect finds the objects in a picture, the surest first.
func (d *Detector) Detect(ctx context.Context, img image.Image, o DetectOptions) (*Objects, error) {
	if d.remote != nil {
		return airpc.Call1[*Objects](d.remote, ctx, "Detect", img, o)
	}
	return d.local.Detect(ctx, img, o)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (d *Detector) Close() error {
	if d.remote != nil {
		return d.remote.Release() // the Runtime in hwnpud unloads it
	}
	return d.local.Close()
}

func init() {
	airole.Register(detectRole)
	bridge.Roles[detectRole.Name] = detectRole
	bridge.Wrap[detectRole.Name] = func(l any) io.Closer { return &Detector{local: l.(detectorImpl)} }
}
