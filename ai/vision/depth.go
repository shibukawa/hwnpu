package vision

import (
	"context"
	"image"
	"io"
	"math"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/visionwire"
)

// Depth is the use of a model of a picture's depth (Depth Anything 3, MoGe-2) for an app: a
// model, a family or a use of the definitions ("depth", "DA3-SMALL", "moge").
func Depth(name string) ai.Use[*DepthEstimator] {
	return ai.Define[*DepthEstimator](depthRole, name)
}

// The definition of a depth model may give the programs export: a directory of
// tools/amd/iron_da3.py (one of --views N among them: a multi-view export, EstimateViews),
// iron_moge.py, or one Prepare made for Apple's or Intel's NPU. depthRole's Prepare and Load
// are the engine's (ai/vision/engine).
var depthRole = &airole.Role{
	Name:     "depth",
	Remote:   func(h *airpc.Handle) io.Closer { return &DepthEstimator{remote: h} },
	Revision: 1, // 1: every size's programs made at Prepare
	Kinds:    []catalog.Kind{catalog.KindDepth},
}

// DepthMap is the depth of a picture.
type DepthMap struct {
	// W and H are the map's size: that of the model's input the picture was stretched to, not
	// the picture's (Near and NearImage give it at any size). The maps below are of that size,
	// row major.
	W, H int
	// Depth is how far each point is: larger is farther. It is in meters when Metric (MoGe-2,
	// and Depth Anything 3 Metric given the camera's focal length or field of view), else in a
	// unit of the picture's own (Depth Anything 3; the views of a multi-view model share it).
	// Where MoGe-2's Mask is off it is 0, no depth: the farthest for Near.
	Depth  []float32
	Metric bool
	// Sky is Depth Anything 3 Metric's sky, from 0 to 1 (SkyThreshold and more is sky), nil for
	// the others.
	Sky []float32
	// Conf is how sure Depth Anything 3 Small, Base and their views are of each point (exp + 1),
	// nil for the others.
	Conf []float32
	// Mask, Normal and Points are MoGe-2's: Mask true where a point has a depth (not the sky or
	// a background past it), Normal each point's normal (x, y, z of unit length: x right, y
	// down, z forward) and Points its place (x, y, z in meters, z the depth); nil for the
	// others.
	Mask           []bool
	Normal, Points []float32
	// FovX is the camera's horizontal field of view in degrees and Fx, Fy its focal lengths in
	// the map's pixels: MoGe-2's (solved, or the one given), 0 for the others.
	FovX, Fx, Fy float64
	// Camera is the view's camera of a multi-view model (EstimateViews), nil for the others.
	Camera *Camera
	// Took is how long the estimate took (for EstimateViews, all the views').
	Took time.Duration
}

// SkyThreshold is the least of DepthMap.Sky that is sky.
const SkyThreshold = visionwire.SkyThreshold

// Camera is a view's camera, as a multi-view model gives it: its world-to-camera [R | t] (3 x 4,
// row major) in the reference view's frame, and its intrinsics fx, fy, cx, cy at the map's size.
type Camera struct {
	W2C            [12]float64
	Fx, Fy, Cx, Cy float64
}

// Near is how near each pixel of the picture at w x h is, row major: 1 the nearest point of
// the picture, 0 the farthest, in proportion to the inverse of the depth (what a point moves
// by on the screen when the eye moves sideways). It is what a parallax shader reads.
func (d *DepthMap) Near(w, h int) []float32 {
	inv := make([]float32, len(d.Depth))
	lo, hi := float32(math.Inf(1)), float32(math.Inf(-1))
	for i, z := range d.Depth {
		if z > 0 && !math.IsInf(float64(z), 0) && !math.IsNaN(float64(z)) {
			inv[i] = 1 / z
			lo, hi = min(lo, inv[i]), max(hi, inv[i])
		}
	}
	if !(hi > lo) {
		return make([]float32, w*h)
	}
	for i, v := range inv {
		inv[i] = min(max((v-lo)/(hi-lo), 0), 1) // a point without a depth (0) is the farthest
	}
	if w == d.W && h == d.H {
		return inv
	}
	return Bilinear(inv, d.W, d.H, w, h)
}

// NearImage is Near as a gray picture of w x h: white the nearest, black the farthest.
func (d *DepthMap) NearImage(w, h int) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	for i, v := range d.Near(w, h) {
		img.Pix[i] = uint8(v*255 + 0.5)
	}
	return img
}

// Bilinear resizes a map xs of w x h (row major) to ow x oh, the pixels' centres aligned (as
// PyTorch's interpolate without align_corners): a copy of xs when the sizes are the same.
func Bilinear(xs []float32, w, h, ow, oh int) []float32 {
	return visionwire.Bilinear(xs, w, h, ow, oh)
}

// DepthOptions are how a picture's depth is estimated. The zero value estimates it as the model
// does alone.
type DepthOptions struct {
	// Focal is the camera's focal length in the picture's pixels (fx = fy), FovX its horizontal
	// field of view in degrees in place of it: Depth Anything 3 Metric then gives meters, and
	// MoGe-2 takes the field of view instead of solving it.
	Focal, FovX float64
}

// DepthEstimator estimates the depth of pictures. It runs one picture at a time; others wait.
type DepthEstimator struct {
	remote *airpc.Handle      // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
	local  depthEstimatorImpl // the engine's (ai/vision/engine), when it is here
}

// depthEstimatorImpl is a depth model of the engine in this process.
type depthEstimatorImpl interface {
	Model() string
	Export() string
	Kind() string
	Family() string
	Sizes() [][2]int
	Views() int
	EstimateWith(ctx context.Context, img image.Image, o DepthOptions) (*DepthMap, error)
	EstimateViews(ctx context.Context, imgs []image.Image, o ViewOptions) ([]*DepthMap, error)
	Warm(ctx context.Context) error
	Close() error
}

// Model is the name of the model in the definitions.
func (d *DepthEstimator) Model() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Model")
	}
	return d.local.Model()
}

// Export is the directory of the export or the programs the model runs from.
func (d *DepthEstimator) Export() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Export")
	}
	return d.local.Export()
}

// Kind is what the model gives: "relative depth" (Depth Anything 3, as one view), "metric
// depth" (Depth Anything 3 Metric, given the focal length; MoGe-2), or "views" (a multi-view
// export: EstimateViews).
func (d *DepthEstimator) Kind() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Kind")
	}
	return d.local.Kind()
}

// Family is the model's family: "moge" for MoGe-2, "da3" for Depth Anything 3.
func (d *DepthEstimator) Family() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Family")
	}
	return d.local.Family()
}

// Sizes are the input sizes of the model (W x H): a picture is stretched to the one nearest its
// shape.
func (d *DepthEstimator) Sizes() [][2]int {
	if d.remote != nil {
		return airpc.Value[[][2]int](d.remote, "Sizes")
	}
	return d.local.Sizes()
}

// Views is how many pictures of one scene a multi-view export takes together (EstimateViews), 0
// for a model of one picture.
func (d *DepthEstimator) Views() int {
	if d.remote != nil {
		return airpc.Value[int](d.remote, "Views")
	}
	return d.local.Views()
}

// Estimate estimates the depth of a picture. The picture is stretched to the size of the
// model's that is nearest its shape; on Apple's Neural Engine and Intel's NPU the program of a
// size is converted when a picture first takes it (seconds, and its first load compiles it).
func (d *DepthEstimator) Estimate(ctx context.Context, img image.Image) (*DepthMap, error) {
	return d.EstimateWith(ctx, img, DepthOptions{})
}

// EstimateWith is Estimate with options.
func (d *DepthEstimator) EstimateWith(ctx context.Context, img image.Image, o DepthOptions) (*DepthMap, error) {
	if d.remote != nil {
		return airpc.Call1[*DepthMap](d.remote, ctx, "EstimateWith", img, o)
	}
	return d.local.EstimateWith(ctx, img, o)
}

// ViewOptions are how the pictures of a scene are estimated together.
type ViewOptions struct {
	// Ref is the reference view's index, whose frame the cameras are in: -1 picks it as Depth
	// Anything 3 does (of 3 views or more); the zero value is the first view's.
	Ref int
}

// EstimateViews estimates the depth of the pictures of one scene together, with a multi-view
// export (Views of them, each stretched to its size): each view's depth, of one scale for all,
// its confidence and its camera (DepthMap.Camera). Took is of all of them.
func (d *DepthEstimator) EstimateViews(ctx context.Context, imgs []image.Image, o ViewOptions) ([]*DepthMap, error) {
	if d.remote != nil {
		return airpc.Call1[[]*DepthMap](d.remote, ctx, "EstimateViews", imgs, o)
	}
	return d.local.EstimateViews(ctx, imgs, o)
}

// Warm loads the programs of every size once, which compiles them for the NPU (ai.Warmer: the
// load of "models prepare"), so that a picture of any aspect does not wait for that.
func (d *DepthEstimator) Warm(ctx context.Context) error {
	if d.remote != nil {
		return airpc.Call0(d.remote, ctx, "Warm")
	}
	return d.local.Warm(ctx)
}

// EstimateStream estimates the depth of each picture of a stream, in a chunk of the picture's
// times.
func (d *DepthEstimator) EstimateStream(ctx context.Context, s flow.Stream[image.Image]) flow.Stream[*DepthMap] {
	return flow.Map(s, func(c flow.Chunk[image.Image]) (*DepthMap, error) { return d.Estimate(ctx, c.Data) })
}

// DepthNode is the node of a depth estimator in a network or a call (flow, ai.App.From,
// ai.NewCall): pictures in, their depth out.
func DepthNode(u ai.Use[*DepthEstimator]) *flow.Node[image.Image, *DepthMap] {
	return ai.Node(u, func(ctx context.Context, d *DepthEstimator, in flow.Stream[image.Image]) flow.Stream[*DepthMap] {
		return d.EstimateStream(ctx, in)
	})
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (d *DepthEstimator) Close() error {
	if d.remote != nil {
		return d.remote.Release() // the Runtime in hwnpud unloads it
	}
	return d.local.Close()
}

func init() {
	airole.Register(depthRole)
	bridge.Roles[depthRole.Name] = depthRole
	bridge.Wrap[depthRole.Name] = func(l any) io.Closer { return &DepthEstimator{local: l.(depthEstimatorImpl)} }
}
