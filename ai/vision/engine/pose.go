package engine

import (
	"context"
	"image"
	"io"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/pose"
)

func poseSetup(m, base chat.Model, log io.Writer) pose.Setup {
	return pose.Setup{Dir: chat.ModelDir(base), Repo: m.Source(), SHA256: m.SHA256[pose.ModelFile], Log: log}
}

var poseCNN = cnn[*pose.Estimator]{
	what:    "RTMO",
	version: pose.ExportVersion,
	fetch: func(ctx context.Context, m chat.Model, log io.Writer) (string, error) {
		return pose.Fetch(ctx, poseSetup(m, m, log))
	},
	export: func(ctx context.Context, python []string, name, model, dir string, log io.Writer) error {
		return pose.ExportNPU(ctx, python, name, model, dir, log)
	},
	convert: func(ctx context.Context, m, base chat.Model, backend string, log io.Writer) (string, string, error) {
		s := poseSetup(m, base, log)
		s.Backend = backend
		return pose.Prepare(ctx, s)
	},
	openNPU: pose.OpenNPU,
	openCPU: pose.OpenCPU,
	open: func(ctx context.Context, dev *hwnpu.Device, model, program string) (*pose.Estimator, error) {
		return pose.Open(ctx, dev, model, program)
	},
	onNPU: (*pose.Estimator).OnNPU,
}

func init() {
	r := bridge.Roles["pose"]
	r.Prepare = poseCNN.prepare
	r.Load = func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		p, device, where, closeDev, err := poseCNN.load(ctx, f, e)
		if err != nil {
			return nil, err
		}
		return bridge.Wrap["pose"](&poseEstimator{e: p, closeDev: closeDev, name: e.Model, device: device, where: where}), nil
	}
}

// poseEstimator is a pose model of the engine (vision.PoseEstimator's).
type poseEstimator struct {
	mu            sync.Mutex
	e             *pose.Estimator
	closeDev      func()
	name          string
	device, where string
}

func (p *poseEstimator) Model() string  { return p.name }
func (p *poseEstimator) Device() string { return p.device }
func (p *poseEstimator) Where() string  { return p.where }

func (p *poseEstimator) Estimate(ctx context.Context, img image.Image, o vision.PoseOptions) (*vision.People, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	score := o.Score
	if score == 0 {
		score = 0.3
	}
	t0 := time.Now()
	found, err := p.e.Estimate(ctx, img, score)
	if err != nil {
		return nil, err
	}
	out := &vision.People{People: []vision.Person{}, Took: time.Since(t0), Resize: p.e.Times[0], Network: p.e.Times[1], Host: p.e.Times[2]}
	b := img.Bounds().Min
	x0, y0 := float64(b.X), float64(b.Y)
	for _, f := range found {
		q := vision.Person{Box: box(f.Box[0]+x0, f.Box[1]+y0, f.Box[2]+x0, f.Box[3]+y0), Score: f.Score}
		for _, k := range f.Keypoints {
			q.Keypoints = append(q.Keypoints, vision.Keypoint{Name: k.Name, Point: vision.Point{X: k.X + x0, Y: k.Y + y0}, Score: k.Score})
		}
		out.People = append(out.People, q)
	}
	return out, nil
}

func (p *poseEstimator) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	err := p.e.Close()
	p.closeDev()
	return err
}
