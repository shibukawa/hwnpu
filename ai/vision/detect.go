package vision

import (
	"context"
	"fmt"
	"image"
	"io"
	"slices"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/vision/detect"
)

// Detect is the use of a model that finds objects and their labels (RTMDet, COCO's 80 classes)
// for an app: a model, a family or a use of the definitions ("detect", "rtmdet:tiny").
func Detect(name string) ai.Use[*Detector] {
	return ai.Define[*Detector](detectRole, name)
}

// COCO are the classes of the COCO models (Detector.Classes): person, bicycle, car, ...
var COCO = slices.Clone(detect.COCO)

// detectSetup is what detect downloads and converts for model m, in the directory of base.
func detectSetup(m, base chat.Model, log io.Writer) (detect.Setup, error) {
	file, err := imagedev.OnlyFile(m, ".onnx")
	if err != nil {
		return detect.Setup{}, err
	}
	return detect.Setup{Dir: chat.ModelDir(base), Repo: m.Source(), File: file, SHA256: m.SHA256[file], Log: log}, nil
}

var detectCNN = cnn[*detect.Detector]{
	what:    "RTMDet",
	version: detect.ExportVersion,
	fetch: func(ctx context.Context, m chat.Model, log io.Writer) (string, error) {
		s, err := detectSetup(m, m, log)
		if err != nil {
			return "", err
		}
		return detect.Fetch(ctx, s)
	},
	export: func(ctx context.Context, python []string, name, model, dir string, log io.Writer) error {
		return detect.ExportNPU(ctx, python, name, model, dir, log)
	},
	convert: func(ctx context.Context, m, base chat.Model, backend string, log io.Writer) (string, string, error) {
		s, err := detectSetup(m, base, log)
		if err != nil {
			return "", "", err
		}
		s.Backend = backend
		return detect.Prepare(ctx, s)
	},
	openNPU: detect.OpenNPU,
	openCPU: detect.OpenCPU,
	open: func(ctx context.Context, dev *hwnpu.Device, _, program string) (*detect.Detector, error) {
		return detect.Open(ctx, dev, program)
	},
	onNPU: (*detect.Detector).OnNPU,
}

// The definition of a detection model may give the settings threads and cpu, and the device
// (common.go).
var detectRole = &airole.Role{
	Remote:  func(h *airpc.Handle) io.Closer { return &Detector{remote: h} },
	Name:    "detect",
	Kinds:   []chat.Kind{chat.KindDetect},
	Prepare: detectCNN.prepare,
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		d, device, where, closeDev, err := detectCNN.load(ctx, f, e)
		if err != nil {
			return nil, err
		}
		return &Detector{d: d, closeDev: closeDev, name: e.Model, device: device, where: where, score: d.Score, nms: d.NMS}, nil
	},
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
	remote        *airpc.Handle // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
	mu            sync.Mutex
	d             *detect.Detector
	closeDev      func()
	name          string
	device, where string
	score, nms    float32
}

// Model is the name of the model in the definitions.
func (d *Detector) Model() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Model")
	}
	return d.name
}

// Device is where the model's network runs: npu or cpu.
func (d *Detector) Device() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Device")
	}
	return d.device
}

// Where says where the model runs, for a log: "on AMD's NPU (rtmdet-m.amd-npu2)", "on the CPU
// (ONNX Runtime, 1 threads)", "all on the NPU", "on the CPU interpreter".
func (d *Detector) Where() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Where")
	}
	return d.where
}

// Classes are the names of the model's classes (COCO's).
func (d *Detector) Classes() []string {
	if d.remote != nil {
		return airpc.Value[[]string](d.remote, "Classes")
	}
	return slices.Clone(d.d.Classes)
}

// Detect finds the objects in a picture, the surest first.
func (d *Detector) Detect(ctx context.Context, img image.Image, o DetectOptions) (*Objects, error) {
	if d.remote != nil {
		return airpc.Call1[*Objects](d.remote, ctx, "Detect", img, o)
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	keep := map[string]bool{}
	for _, c := range o.Classes {
		if !slices.Contains(d.d.Classes, c) {
			return nil, fmt.Errorf("vision: %q is not one of %s's classes", c, shown(d.name))
		}
		keep[c] = true
	}
	d.d.Score, d.d.NMS = d.score, d.nms
	if o.Score != 0 {
		d.d.Score = o.Score
	}
	if o.NMS != 0 {
		d.d.NMS = o.NMS
	}
	t0 := time.Now()
	objs, err := d.d.Detect(ctx, img)
	if err != nil {
		return nil, err
	}
	out := &Objects{Took: time.Since(t0), Resize: d.d.Times[0], Network: d.d.Times[1], Host: d.d.Times[2], Objects: []Object{}}
	b := img.Bounds()
	for _, ob := range objs {
		if len(keep) > 0 && !keep[ob.Label] {
			continue
		}
		bx := box(ob.Box[0], ob.Box[1], ob.Box[2], ob.Box[3])
		if b.Min != (image.Point{}) {
			bx = bx.add(b.Min)
		}
		out.Objects = append(out.Objects, Object{Box: bx, Class: ob.Class, Label: ob.Label, Score: float64(ob.Score)})
	}
	return out, nil
}

// add is the box moved by p.
func (b Box) add(p image.Point) Box {
	x, y := float64(p.X), float64(p.Y)
	return Box{Point{b.Min.X + x, b.Min.Y + y}, Point{b.Max.X + x, b.Max.Y + y}}
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (d *Detector) Close() error {
	if d.remote != nil {
		return d.remote.Release() // the Runtime in hwnpud unloads it
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	err := d.d.Close()
	d.closeDev()
	return err
}

func init() {
	airole.Register(detectRole)
}
