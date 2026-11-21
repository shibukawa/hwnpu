package vision

import (
	"context"
	"fmt"
	"image"
	"io"
	"math"
	"sync"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/vision/depth"
	"github.com/shibukawa/hwnpu/internal/vision/moge"
)

// Depth is the use of a model of a picture's depth (Depth Anything 3, MoGe-2) for an app: a
// model, a family or a use of the definitions ("depth", "DA3-SMALL", "moge").
func Depth(name string) ai.Use[*DepthEstimator] {
	return ai.Define[*DepthEstimator](depthRole, name)
}

var depthRole = &airole.Role{
	Name:  "depth",
	Kinds: []chat.Kind{chat.KindDepth},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		m := e.Model
		var dir string
		switch mg := m.Family == "moge"; {
		case e.NPU.AMD():
			// AMD's NPU runs an export of IRON (tools/amd/iron_da3.py, iron_moge.py), here or in
			// the sources
			tool := "tools/amd/iron_da3.py"
			if mg {
				tool = "tools/amd/iron_moge.py"
			}
			if dir, err = chat.AMDPictureExport(e.Ctx, m, tool, 0, e.Log); err == nil {
				iron.ShareExport(dir, logf(e.Log))
			}
		case mg:
			dir, err = moge.Prepare(e.Ctx, moge.Setup{Dir: chat.ModelsDir(), Repo: m.Repo, Name: m.Name, Backend: e.NPU.Format, Log: e.Log})
		default:
			dir, err = depth.Prepare(e.Ctx, depth.Setup{Dir: chat.ModelsDir(), Repo: m.Repo, Name: m.Name, Backend: e.NPU.Format, Log: e.Log})
		}
		if err != nil {
			return f, err
		}
		f.Set("export", dir)
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		dev, err := openDevice(ctx, e)
		if err != nil {
			return nil, err
		}
		d := &DepthEstimator{dev: dev, name: e.Model}
		d.model, _ = chat.Lookup(e.Model)
		ctx = chat.WithModel(ctx, d.model)
		dir := f.Path("export")
		if moge.IsExport(dir) {
			d.mg, err = moge.Open(ctx, dev, dir)
		} else {
			d.da, err = depth.Open(ctx, dev, dir)
		}
		if err != nil {
			dev.Close()
			return nil, err
		}
		return d, nil
	},
}

// openDevice opens the NPU a model of pictures runs on: AMD's for an export of IRON, else the
// one of the programs' format, where Apple's Neural Engine is to run every operation.
func openDevice(ctx context.Context, e airole.LoadEnv) (*hwnpu.Device, error) {
	if e.NPU.AMD() {
		return hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	}
	var opts []hwnpu.Option
	if llm.AppleFormat(e.NPU.Format) {
		opts = append(opts, hwnpu.WithPlacement(hwnpu.PlacementStrict))
	}
	dev, _, err := llm.OpenDevice(ctx, e.NPU.Device, e.Cache, opts...)
	return dev, err
}

// logf writes lines to w (nil for nowhere).
func logf(w io.Writer) func(string, ...any) {
	return func(format string, args ...any) {
		if w != nil {
			fmt.Fprintf(w, format+"\n", args...)
		}
	}
}

// DepthMap is the depth of a picture.
type DepthMap struct {
	// W and H are the map's size: that of the model's input the picture was stretched to, not
	// the picture's (Near and NearImage give it at any size).
	W, H int
	// Depth is how far each point is, row major: larger is farther. It is in meters when Metric
	// (MoGe-2), else in a unit of the picture's own (Depth Anything 3).
	Depth  []float32
	Metric bool
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
	return moge.Bilinear(inv, d.W, d.H, w, h)
}

// NearImage is Near as a gray picture of w x h: white the nearest, black the farthest.
func (d *DepthMap) NearImage(w, h int) *image.Gray {
	img := image.NewGray(image.Rect(0, 0, w, h))
	for i, v := range d.Near(w, h) {
		img.Pix[i] = uint8(v*255 + 0.5)
	}
	return img
}

// DepthEstimator estimates the depth of pictures. It runs one picture at a time; others wait.
type DepthEstimator struct {
	mu    sync.Mutex
	da    *depth.Estimator
	mg    *moge.Estimator
	dev   *hwnpu.Device
	model chat.Model
	name  string
}

// Model is the name of the model in the definitions.
func (d *DepthEstimator) Model() string { return d.name }

// Estimate estimates the depth of a picture. The picture is stretched to the size of the
// model's that is nearest its shape; on Apple's Neural Engine and Intel's NPU the program of a
// size is converted when a picture first takes it (seconds, and its first load compiles it).
func (d *DepthEstimator) Estimate(ctx context.Context, img image.Image) (*DepthMap, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	ctx = chat.WithModel(ctx, d.model)
	if d.mg != nil {
		r, err := d.mg.Estimate(ctx, img)
		if err != nil {
			return nil, err
		}
		out := &DepthMap{W: r.W, H: r.H, Depth: append([]float32(nil), r.Depth...), Metric: true}
		for i, ok := range r.Mask {
			if !ok {
				out.Depth[i] = 0 // no depth there (the sky): the farthest
			}
		}
		return out, nil
	}
	r, err := d.da.Estimate(ctx, img)
	if err != nil {
		return nil, err
	}
	return &DepthMap{W: r.W, H: r.H, Depth: r.Depth}, nil
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
	if e := d.dev.Close(); err == nil {
		err = e
	}
	return err
}
