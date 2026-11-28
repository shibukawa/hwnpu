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

// Pose is the use of a model that finds people and their 17 keypoints (RTMO) for an app: a
// model, a family or a use of the definitions ("pose", "rtmo:s").
func Pose(name string) ai.Use[*PoseEstimator] {
	return ai.Define[*PoseEstimator](poseRole, name)
}

// Keypoints are the names of the keypoints of a Person, in order (COCO's): nose, left_eye, ...
var Keypoints = slices.Clone(visionwire.Keypoints)

// Skeleton are COCO's limbs, by the keypoints they join.
var Skeleton = [][2]int{{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}}

// The definition of a pose model may give the settings threads and cpu, and the device
// (common.go). poseRole's Prepare and Load are the engine's (ai/vision/engine).
var poseRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &PoseEstimator{p: airpc.RemoteProxy[poseEstimatorImpl](h)} },
	Name:   "pose",
	Kinds:  []catalog.Kind{catalog.KindPose},
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
	p airpc.Proxy[poseEstimatorImpl]
}

// poseEstimatorImpl is a pose model of the engine in this process.
type poseEstimatorImpl interface {
	Model() string
	Device() string
	Where() string
	Estimate(ctx context.Context, img image.Image, o PoseOptions) (*People, error)
	Close() error
}

// Model is the name of the model in the definitions.
func (p *PoseEstimator) Model() string {
	return airpc.ProxyValue(p.p, "Model", poseEstimatorImpl.Model)
}

// Device is where the model's network runs: npu or cpu.
func (p *PoseEstimator) Device() string {
	return airpc.ProxyValue(p.p, "Device", poseEstimatorImpl.Device)
}

// Where says where the model runs, for a log (Detector.Where).
func (p *PoseEstimator) Where() string {
	return airpc.ProxyValue(p.p, "Where", poseEstimatorImpl.Where)
}

// Estimate finds the people in a picture and their keypoints.
func (p *PoseEstimator) Estimate(ctx context.Context, img image.Image, o PoseOptions) (*People, error) {
	return airpc.ProxyCall1(p.p, ctx, "Estimate", func(l poseEstimatorImpl) (*People, error) {
		return l.Estimate(ctx, img, o)
	}, img, o)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (p *PoseEstimator) Close() error { return p.p.Close() }

func init() {
	airole.Register(poseRole)
	bridge.Roles[poseRole.Name] = poseRole
	bridge.Wrap[poseRole.Name] = func(l any) io.Closer { return &PoseEstimator{p: airpc.LocalProxy(l.(poseEstimatorImpl))} }
}
