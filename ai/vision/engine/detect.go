package engine

import (
	"context"
	"fmt"
	"image"
	"io"
	"slices"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/vision/detect"
)

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
	// Open reads the program alone: a Core ML package of the shared layers spares the ONNX file
	programOnly: true,
	open: func(ctx context.Context, dev *hwnpu.Device, _, program string) (*detect.Detector, error) {
		return detect.Open(ctx, dev, program)
	},
	onNPU: (*detect.Detector).OnNPU,
}

func init() {
	r := bridge.Roles["detect"]
	r.Prepare = detectCNN.prepare
	r.Load = func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		d, device, where, closeDev, err := detectCNN.load(ctx, f, e)
		if err != nil {
			return nil, err
		}
		return bridge.Wrap["detect"](&detector{d: d, closeDev: closeDev, name: e.Model, device: device, where: where, score: d.Score, nms: d.NMS}), nil
	}
}

// detector is a detection model of the engine (vision.Detector's).
type detector struct {
	mu            sync.Mutex
	d             *detect.Detector
	closeDev      func()
	name          string
	device, where string
	score, nms    float32
}

func (d *detector) Model() string     { return d.name }
func (d *detector) Device() string    { return d.device }
func (d *detector) Where() string     { return d.where }
func (d *detector) Classes() []string { return slices.Clone(d.d.Classes) }

func (d *detector) Detect(ctx context.Context, img image.Image, o vision.DetectOptions) (*vision.Objects, error) {
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
	out := &vision.Objects{Took: time.Since(t0), Resize: d.d.Times[0], Network: d.d.Times[1], Host: d.d.Times[2], Objects: []vision.Object{}}
	b := img.Bounds()
	for _, ob := range objs {
		if len(keep) > 0 && !keep[ob.Label] {
			continue
		}
		bx := box(ob.Box[0], ob.Box[1], ob.Box[2], ob.Box[3])
		if b.Min != (image.Point{}) {
			bx = add(bx, b.Min)
		}
		out.Objects = append(out.Objects, vision.Object{Box: bx, Class: ob.Class, Label: ob.Label, Score: float64(ob.Score)})
	}
	return out, nil
}

// add is box b moved by p.
func add(b vision.Box, p image.Point) vision.Box {
	x, y := float64(p.X), float64(p.Y)
	return vision.Box{Min: vision.Point{X: b.Min.X + x, Y: b.Min.Y + y}, Max: vision.Point{X: b.Max.X + x, Y: b.Max.Y + y}}
}

func (d *detector) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	err := d.d.Close()
	d.closeDev()
	return err
}
