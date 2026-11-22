package vision

import (
	"context"
	"fmt"
	"image"
	"io"
	"math"
	"path/filepath"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/vision/depth"
	"github.com/shibukawa/hwnpu/internal/vision/moge"
)

// Depth is the use of a model of a picture's depth (Depth Anything 3, MoGe-2) for an app: a
// model, a family or a use of the definitions ("depth", "DA3-SMALL", "moge").
func Depth(name string) ai.Use[*DepthEstimator] {
	return ai.Define[*DepthEstimator](depthRole, name)
}

// The definition of a depth model may give the programs export: a directory of
// tools/amd/iron_da3.py (one of --views N among them: a multi-view export, EstimateViews),
// iron_moge.py, or one Prepare made for Apple's or Intel's NPU.
var depthRole = &airole.Role{
	Name:     "depth",
	Revision: 1, // 1: every size's programs made at Prepare
	Kinds:    []chat.Kind{chat.KindDepth},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		m := e.Model
		if dir := program(m, "export"); dir != "" {
			if !depth.IsExport(dir) && !depth.IsMultiExport(dir) && !moge.IsExport(dir) {
				return f, fmt.Errorf("%s is not an export of Depth Anything 3 or MoGe-2", dir)
			}
			if e.NPU.AMD() {
				iron.ShareExport(dir, logf(e.Log))
			}
			f.Set("export", dir)
			return f, nil
		}
		var dir string
		switch mg := m.Family == "moge"; {
		case e.NPU.AMD():
			// AMD's NPU runs an export of IRON (tools/amd/iron_da3.py, iron_moge.py), here or in
			// the sources: the one of the model the app's was made from
			tool := "tools/amd/iron_da3.py"
			if mg {
				tool = "tools/amd/iron_moge.py"
			}
			if dir, err = chat.AMDPictureExport(e.Ctx, e.Base, tool, 0, e.Log); err == nil {
				iron.ShareExport(dir, logf(e.Log))
			}
		case mg:
			dir, err = moge.Prepare(e.Ctx, moge.Setup{Dir: chat.ModelsDir(), Repo: m.Repo, Name: e.Base.Name, Backend: e.NPU.Format, Log: e.Log})
		default:
			dir, err = depth.Prepare(e.Ctx, depth.Setup{Dir: chat.ModelsDir(), Repo: m.Repo, Name: e.Base.Name, Backend: e.NPU.Format, Log: e.Log})
		}
		if err == nil && m.Family != "moge" && !e.NPU.AMD() {
			// every aspect's programs now, not at the first picture of each: after "models
			// prepare" a run needs no network nor conversion (Warm compiles them)
			err = depth.ConvertSizes(e.Ctx, dir, e.Log)
		}
		if err != nil {
			return f, err
		}
		f.Set("export", dir)
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		dir := f.Path("export")
		dev, err := openDevice(ctx, e)
		if err != nil {
			return nil, err
		}
		d := &DepthEstimator{dev: dev, name: e.Model, dir: dir}
		d.model, _ = chat.Lookup(e.Model)
		ctx = chat.WithModel(ctx, d.model)
		lf := logf(e.Log)
		switch {
		case depth.IsMultiExport(dir):
			if d.mv, err = depth.OpenMulti(ctx, dev, dir); err == nil {
				d.mv.Logf = lf
			}
		case moge.IsExport(dir):
			if d.mg, err = moge.Open(ctx, dev, dir); err == nil {
				d.mg.Logf = lf
			}
		default:
			if d.da, err = depth.Open(ctx, dev, dir); err == nil {
				d.da.Logf = lf
			}
		}
		if err != nil {
			dev.Close()
			return nil, err
		}
		return d, nil
	},
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
const SkyThreshold = depth.SkyThreshold

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
func Bilinear(xs []float32, w, h, ow, oh int) []float32 { return moge.Bilinear(xs, w, h, ow, oh) }

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
	mu    sync.Mutex
	da    *depth.Estimator
	mg    *moge.Estimator
	mv    *depth.Multi
	dev   *hwnpu.Device
	model chat.Model
	name  string
	dir   string
}

// Model is the name of the model in the definitions.
func (d *DepthEstimator) Model() string { return d.name }

// Export is the directory of the export or the programs the model runs from.
func (d *DepthEstimator) Export() string { return d.dir }

// Kind is what the model gives: "relative depth" (Depth Anything 3, as one view), "metric
// depth" (Depth Anything 3 Metric, given the focal length; MoGe-2), or "views" (a multi-view
// export: EstimateViews).
func (d *DepthEstimator) Kind() string {
	switch {
	case d.mv != nil:
		return "views"
	case d.mg != nil || d.da.Metric:
		return "metric depth"
	}
	return "relative depth"
}

// Family is the model's family: "moge" for MoGe-2, "da3" for Depth Anything 3.
func (d *DepthEstimator) Family() string {
	if d.mg != nil {
		return "moge"
	}
	return "da3"
}

// Sizes are the input sizes of the model (W x H): a picture is stretched to the one nearest its
// shape.
func (d *DepthEstimator) Sizes() [][2]int {
	switch {
	case d.mg != nil:
		return d.mg.Sizes
	case d.da != nil:
		return d.da.Sizes
	}
	return [][2]int{{d.mv.Width, d.mv.Height}}
}

// Views is how many pictures of one scene a multi-view export takes together (EstimateViews), 0
// for a model of one picture.
func (d *DepthEstimator) Views() int {
	if d.mv == nil {
		return 0
	}
	return d.mv.Views
}

// Estimate estimates the depth of a picture. The picture is stretched to the size of the
// model's that is nearest its shape; on Apple's Neural Engine and Intel's NPU the program of a
// size is converted when a picture first takes it (seconds, and its first load compiles it).
func (d *DepthEstimator) Estimate(ctx context.Context, img image.Image) (*DepthMap, error) {
	return d.EstimateWith(ctx, img, DepthOptions{})
}

// EstimateWith is Estimate with options.
func (d *DepthEstimator) EstimateWith(ctx context.Context, img image.Image, o DepthOptions) (*DepthMap, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	ctx = chat.WithModel(ctx, d.model)
	t0 := time.Now()
	switch {
	case d.mv != nil:
		return nil, fmt.Errorf("vision: %s takes %d pictures of a scene together (EstimateViews)", filepath.Base(d.dir), d.mv.Views)
	case d.mg != nil:
		d.mg.FovX = o.FovX
		r, err := d.mg.Estimate(ctx, img)
		if err != nil {
			return nil, err
		}
		out := &DepthMap{W: r.W, H: r.H, Depth: append([]float32(nil), r.Depth...), Metric: true, Mask: r.Mask,
			Normal: r.Normal, Points: r.Points, FovX: r.FovX, Fx: r.Fx, Fy: r.Fy}
		for i, ok := range r.Mask {
			if !ok {
				out.Depth[i] = 0 // no depth there (the sky): the farthest
			}
		}
		out.Took = time.Since(t0)
		return out, nil
	}
	r, err := d.da.Estimate(ctx, img)
	if err != nil {
		return nil, err
	}
	out := &DepthMap{W: r.W, H: r.H, Depth: r.Depth, Sky: r.Sky, Conf: r.Conf}
	if r.Metric && (o.Focal > 0 || o.FovX > 0) {
		ow, oh := img.Bounds().Dx(), img.Bounds().Dy()
		f := o.Focal
		if f == 0 {
			f = float64(ow) / 2 / math.Tan(o.FovX*math.Pi/360)
		}
		r.Meters(f, f, ow, oh)
		out.Metric = true
	}
	out.Took = time.Since(t0)
	return out, nil
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
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.mv == nil {
		return nil, fmt.Errorf("vision: %s is not a multi-view export", filepath.Base(d.dir))
	}
	if len(imgs) != d.mv.Views {
		return nil, fmt.Errorf("vision: the export takes %d pictures together, got %d", d.mv.Views, len(imgs))
	}
	t0 := time.Now()
	d.mv.Ref = o.Ref
	rs, cams, err := d.mv.Estimate(chat.WithModel(ctx, d.model), imgs)
	if err != nil {
		return nil, err
	}
	took := time.Since(t0)
	out := make([]*DepthMap, len(rs))
	for i, r := range rs {
		c := Camera(cams[i])
		out[i] = &DepthMap{W: r.W, H: r.H, Depth: r.Depth, Conf: r.Conf, Camera: &c, Took: took}
	}
	return out, nil
}

// Warm loads the programs of every size once, which compiles them for the NPU (ai.Warmer: the
// load of "models prepare"), so that a picture of any aspect does not wait for that.
func (d *DepthEstimator) Warm(ctx context.Context) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.da == nil {
		return nil
	}
	return d.da.Warm(chat.WithModel(ctx, d.model))
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
	d.mu.Lock()
	defer d.mu.Unlock()
	var err error
	if d.mg != nil {
		err = d.mg.Close()
	}
	if d.da != nil {
		err = d.da.Close()
	}
	if d.mv != nil {
		err = d.mv.Close()
	}
	if e := d.dev.Close(); err == nil {
		err = e
	}
	return err
}

func init() { airole.Register(depthRole) }
