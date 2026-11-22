package vision

import (
	"context"
	"image"
	"io"
	"slices"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/pose"
)

// Pose is the use of a model that finds people and their 17 keypoints (RTMO) for an app: a
// model, a family or a use of the definitions ("pose", "rtmo:s").
func Pose(name string) ai.Use[*PoseEstimator] {
	return ai.Define[*PoseEstimator](poseRole, name)
}

// Keypoints are the names of the keypoints of a Person, in order (COCO's): nose, left_eye, ...
var Keypoints = slices.Clone(pose.Names)

// Skeleton are COCO's limbs, by the keypoints they join.
var Skeleton = [][2]int{{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}}

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

// The definition of a pose model may give the settings threads and cpu, and the device
// (common.go).
var poseRole = &airole.Role{
	Name:    "pose",
	Kinds:   []chat.Kind{chat.KindPose},
	Prepare: poseCNN.prepare,
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		p, device, where, closeDev, err := poseCNN.load(ctx, f, e)
		if err != nil {
			return nil, err
		}
		return &PoseEstimator{e: p, closeDev: closeDev, name: e.Model, device: device, where: where}, nil
	},
}

// Person is a person found in a picture, in its pixels: the box, how sure the model is (0 to
// 1), and the keypoints, in the order of Keypoints.
type Person struct {
	Box       Box
	Score     float64
	Keypoints []Keypoint
}

// Keypoint is a keypoint of a person: its name (Keypoints), place and score (visible from about
// 0.3).
type Keypoint struct {
	Name string
	Point
	Score float64
}

// People are the people found in a picture, the surest first, and what finding them took.
type People struct {
	People []Person
	// Took is how long it all took: Resize the picture's resizing, Network the model's network,
	// Host the people and keypoints worked out on the host.
	Took, Resize, Network, Host time.Duration
}

// PoseOptions are how people are found.
type PoseOptions struct {
	// Score is the least score of a person kept (0: the default, 0.3; a negative value keeps
	// every one).
	Score float64
}

// PoseEstimator finds the people in pictures and their keypoints. It runs one picture at a
// time; others wait.
type PoseEstimator struct {
	mu            sync.Mutex
	e             *pose.Estimator
	closeDev      func()
	name          string
	device, where string
}

// Model is the name of the model in the definitions.
func (p *PoseEstimator) Model() string { return p.name }

// Device is where the model's network runs: npu or cpu.
func (p *PoseEstimator) Device() string { return p.device }

// Where says where the model runs, for a log (Detector.Where).
func (p *PoseEstimator) Where() string { return p.where }

// Estimate finds the people in a picture and their keypoints.
func (p *PoseEstimator) Estimate(ctx context.Context, img image.Image, o PoseOptions) (*People, error) {
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
	out := &People{People: []Person{}, Took: time.Since(t0), Resize: p.e.Times[0], Network: p.e.Times[1], Host: p.e.Times[2]}
	b := img.Bounds().Min
	x0, y0 := float64(b.X), float64(b.Y)
	for _, f := range found {
		q := Person{Box: box(f.Box[0]+x0, f.Box[1]+y0, f.Box[2]+x0, f.Box[3]+y0), Score: f.Score}
		for _, k := range f.Keypoints {
			q.Keypoints = append(q.Keypoints, Keypoint{Name: k.Name, Point: Point{k.X + x0, k.Y + y0}, Score: k.Score})
		}
		out.People = append(out.People, q)
	}
	return out, nil
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (p *PoseEstimator) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	err := p.e.Close()
	p.closeDev()
	return err
}
