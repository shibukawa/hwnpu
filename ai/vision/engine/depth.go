package engine

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
	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/vision/depth"
	"github.com/shibukawa/hwnpu/internal/vision/moge"
)

func init() {
	r := bridge.Roles["depth"]
	r.Prepare = prepareDepth
	r.Load = loadDepth
}

func prepareDepth(e airole.PrepareEnv) (f airole.Files, err error) {
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
}

func loadDepth(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	dir := f.Path("export")
	dev, err := openDevice(ctx, e)
	if err != nil {
		return nil, err
	}
	d := &depthEstimator{dev: dev, name: e.Model, dir: dir}
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
	return bridge.Wrap["depth"](d), nil
}

// depthEstimator is a depth model of the engine (vision.DepthEstimator's).
type depthEstimator struct {
	mu    sync.Mutex
	da    *depth.Estimator
	mg    *moge.Estimator
	mv    *depth.Multi
	dev   *hwnpu.Device
	model chat.Model
	name  string
	dir   string
}

func (d *depthEstimator) Model() string  { return d.name }
func (d *depthEstimator) Export() string { return d.dir }

func (d *depthEstimator) Kind() string {
	switch {
	case d.mv != nil:
		return "views"
	case d.mg != nil || d.da.Metric:
		return "metric depth"
	}
	return "relative depth"
}

func (d *depthEstimator) Family() string {
	if d.mg != nil {
		return "moge"
	}
	return "da3"
}

func (d *depthEstimator) Sizes() [][2]int {
	switch {
	case d.mg != nil:
		return d.mg.Sizes
	case d.da != nil:
		return d.da.Sizes
	}
	return [][2]int{{d.mv.Width, d.mv.Height}}
}

func (d *depthEstimator) Views() int {
	if d.mv == nil {
		return 0
	}
	return d.mv.Views
}

func (d *depthEstimator) EstimateWith(ctx context.Context, img image.Image, o vision.DepthOptions) (*vision.DepthMap, error) {
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
		out := &vision.DepthMap{W: r.W, H: r.H, Depth: append([]float32(nil), r.Depth...), Metric: true, Mask: r.Mask,
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
	out := &vision.DepthMap{W: r.W, H: r.H, Depth: r.Depth, Sky: r.Sky, Conf: r.Conf}
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

func (d *depthEstimator) EstimateViews(ctx context.Context, imgs []image.Image, o vision.ViewOptions) ([]*vision.DepthMap, error) {
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
	out := make([]*vision.DepthMap, len(rs))
	for i, r := range rs {
		c := vision.Camera(cams[i])
		out[i] = &vision.DepthMap{W: r.W, H: r.H, Depth: r.Depth, Conf: r.Conf, Camera: &c, Took: took}
	}
	return out, nil
}

func (d *depthEstimator) Warm(ctx context.Context) error {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.da == nil {
		return nil
	}
	return d.da.Warm(chat.WithModel(ctx, d.model))
}

func (d *depthEstimator) Close() error {
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
