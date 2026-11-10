package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/vision/depth"
	"github.com/shibukawa/hwnpu/internal/vision/moge"
)

// DepthCmd is "hwnpu-image depth": the depth of pictures on AMD's NPU or Apple's Neural Engine with Depth Anything 3
// (depth-anything/DA3*, Apache-2.0; exports of tools/amd/iron_da3.py, internal/vision/depth) or
// MoGe-2 (Ruicheng/moge-2-*-normal, MIT; exports of tools/amd/iron_moge.py, internal/vision/moge).
//
// For each picture it writes NAME.depth.png next to it (or into --out): the inverse depth (near
// is bright) normalized to 8 bits, at the picture's size. The export holds DA3's or MoGe-2's
// official sizes and the picture is resized to the one nearest its aspect.
//
// The depth is relative (larger is farther, up to an unknown scale), except with DA3METRIC-LARGE
// given the camera's focal length (--focal, in the picture's pixels) or horizontal field of view
// (--fov, degrees), and with MoGe-2, which solves the focal length: then it is meters. An export
// of DA3-SMALL / BASE with --views N takes N pictures of one scene together: one scale for all,
// the cameras in cameras.json and with --ply the points in one frame.
type DepthCmd struct {
	Pictures []string `arg:"" type:"existingfile" help:"The pictures (with a multi-view export, exactly its views)."`
	Model    string   `short:"m" placeholder:"NAME|EXPORT" default:"depth" help:"The model: a name, a family or a use of models.yaml (depth: DA3-SMALL, DA3-BASE or DA3MONO-LARGE by the memory, as da3; da3, da3:small, da3metric, moge, moge:vitl; a family alone picks for the machine's memory, or for the memory a tag names: moge:8g), run from ~/models/NAME.amd-npu2, or an export's directory (tools/amd/iron_da3.py, iron_moge.py; a multi-view one of iron_da3.py --views)."`
	Out      string   `short:"o" type:"path" placeholder:"DIR" help:"The directory to write into (default: next to each picture)."`
	Raw      bool     `help:"Also write the depth as raw float32 (NAME.depth.f32, at the export's size; MoGe-2 also NAME.points.f32)."`
	Sky      bool     `help:"Also write the sky mask (NAME.sky.png); DA3-SMALL / BASE: the confidence (NAME.conf.png); MoGe-2: the mask (NAME.mask.png)."`
	Normal   bool     `help:"MoGe-2: also write the normals (NAME.normal.png, (n + 1) / 2 as RGB)."`
	Keep     bool     `name:"export-size" help:"Write the pictures at the export's size instead of the input's."`
	Focal    float64  `placeholder:"PIXELS" help:"DA3METRIC-LARGE: the focal length in the picture's pixels (fx = fy), for meters."`
	Fov      float64  `placeholder:"DEGREES" help:"The horizontal field of view in degrees, in place of --focal (MoGe-2: given, not solved)."`
	Ref      int      `default:"-1" help:"A multi-view export: the reference view's index (default: chosen as DA3 does)."`
	Ply      string   `type:"path" placeholder:"FILE" help:"A multi-view export: also write the views' points in one frame as a PLY file."`
}

// Run estimates the depth of each picture.
func (c *DepthCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	t0 := time.Now()
	model := expandHome(c.Model)
	apple := false
	if format, err := llm.ProgramFormat(ctx, ""); err == nil && format == llm.BackendCoreML {
		apple = true
	}
	if !isPath(c.Model) {
		m, err := resolve(c.Model, chat.KindDepth)
		if err != nil {
			return err
		}
		switch {
		case apple && m.Family == "moge":
			if model, err = moge.Prepare(ctx, moge.Setup{Dir: chat.ModelsDir(), Repo: m.Repo, Name: m.Name, Log: lw}); err != nil {
				return err
			}
		case apple:
			if model, err = depth.Prepare(ctx, depth.Setup{Dir: chat.ModelsDir(), Repo: m.Repo, Name: m.Name, Log: lw}); err != nil {
				return err
			}
		default:
			tool := "tools/amd/iron_da3.py"
			if m.Family == "moge" {
				tool = "tools/amd/iron_moge.py"
			}
			if model, err = amdExport(m, tool); err != nil {
				return err
			}
		}
	}
	var dev *hwnpu.Device
	if apple {
		d, _, err := llm.OpenDevice(ctx, llm.BackendCoreML, cacheDirOf(""), hwnpu.WithPlacement(hwnpu.PlacementStrict))
		if err != nil {
			return err
		}
		dev = d
	} else {
		d, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
		if err != nil {
			return err
		}
		dev = d
		iron.ShareExport(model, logf)
	}
	defer dev.Close()
	o := opts{dir: c.Out, raw: c.Raw, sky: c.Sky, keep: c.Keep, focal: c.Focal, fov: c.Fov, normal: c.Normal}
	if depth.IsMultiExport(model) {
		m, err := depth.OpenMulti(ctx, dev, model)
		if err != nil {
			return err
		}
		defer m.Close()
		m.Logf = logf
		m.Ref = c.Ref
		return runMulti(ctx, m, c.Pictures, o, c.Ply, logf)
	}
	if moge.IsExport(model) {
		m, err := moge.Open(ctx, dev, model)
		if err != nil {
			return err
		}
		defer m.Close()
		m.Logf = logf
		m.FovX = c.Fov
		logf("%s: MoGe-2, metric depth, sizes %v", filepath.Base(model), m.Sizes)
		for _, path := range c.Pictures {
			if err := runMoge(ctx, m, path, o, logf); err != nil {
				return fmt.Errorf("%s: %w", path, err)
			}
		}
		return nil
	}
	e, err := depth.Open(ctx, dev, model)
	if err != nil {
		return err
	}
	defer e.Close()
	e.Logf = logf
	kind := "relative depth"
	if e.Metric {
		kind = "metric depth"
		if c.Focal == 0 && c.Fov == 0 {
			kind += " (relative without --focal or --fov)"
		}
	}
	logf("%s: %s, sizes %v (%s)", filepath.Base(model), kind, e.Sizes, console.Duration(time.Since(t0)))
	for _, path := range c.Pictures {
		if err := run(ctx, e, path, o, logf); err != nil {
			return fmt.Errorf("%s: %w", path, err)
		}
	}
	return nil
}

type opts struct {
	dir                    string
	raw, sky, keep, normal bool
	focal, fov             float64
}

// runMoge writes MoGe-2's depth (meters) of the picture at path: NAME.depth.png (the inverse
// depth, the masked pixels black), --raw NAME.depth.f32 (+Inf where masked) and NAME.points.f32
// (x, y, z a pixel), --sky NAME.mask.png, --normal NAME.normal.png.
func runMoge(ctx context.Context, m *moge.Estimator, path string, o opts, logf func(string, ...any)) error {
	img, err := decode(path)
	if err != nil {
		return err
	}
	t0 := time.Now()
	r, err := m.Estimate(ctx, img)
	if err != nil {
		return err
	}
	base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	dir := o.dir
	if dir == "" {
		dir = filepath.Dir(path)
	}
	inv := make([]float32, len(r.Depth))
	lo, hi := float32(math.Inf(1)), float32(math.Inf(-1))
	var near float32 = float32(math.Inf(1))
	for i, d := range r.Depth {
		if r.Mask[i] {
			inv[i] = 1 / d
			lo, hi = min(lo, inv[i]), max(hi, inv[i])
			near = min(near, d)
		}
	}
	for i := range inv {
		if !r.Mask[i] {
			inv[i] = lo // black
		}
	}
	out := filepath.Join(dir, base+".depth.png")
	if err := writeGray(out, inv, r.W, r.H, lo, hi); err != nil {
		return err
	}
	if o.sky {
		mk := make([]float32, len(r.Mask))
		for i, v := range r.Mask {
			if v {
				mk[i] = 1
			}
		}
		if err := writeGray(filepath.Join(dir, base+".mask.png"), mk, r.W, r.H, 0, 1); err != nil {
			return err
		}
	}
	if o.normal {
		if err := writeRGB(filepath.Join(dir, base+".normal.png"), r.Normal, r.W, r.H); err != nil {
			return err
		}
	}
	if o.raw {
		if err := writeF32(filepath.Join(dir, base+".depth.f32"), r.Depth); err != nil {
			return err
		}
		if err := writeF32(filepath.Join(dir, base+".points.f32"), r.Points); err != nil {
			return err
		}
	}
	c := r.Depth[(r.H/2)*r.W+r.W/2]
	logf("%s: %dx%d in %s, %.2f m at the centre, %.2f m nearest, horizontal fov %.1f°", out, r.W, r.H,
		console.Duration(time.Since(t0)), c, near, r.FovX)
	return nil
}

// runMulti runs a multi-view export (iron_da3.py --views) on the pictures together: each
// one's NAME.depth.png (one scale for all; --raw, --sky the confidence), cameras.json beside the
// first's (or in --out: each picture's world-to-camera [R | t] and intrinsics at the processed
// size), and --ply the points of all the views in the reference view's frame.
func runMulti(ctx context.Context, m *depth.Multi, paths []string, o opts, ply string, logf func(string, ...any)) error {
	if len(paths) != m.Views {
		return fmt.Errorf("the export takes %d pictures together, got %d", m.Views, len(paths))
	}
	imgs := make([]image.Image, len(paths))
	for i, p := range paths {
		img, err := decode(p)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		imgs[i] = img
	}
	t0 := time.Now()
	rs, cams, err := m.Estimate(ctx, imgs)
	if err != nil {
		return err
	}
	lo, hi := float32(math.Inf(1)), float32(math.Inf(-1))
	for _, r := range rs {
		for _, d := range r.Depth {
			lo, hi = min(lo, 1/d), max(hi, 1/d)
		}
	}
	type camJSON struct {
		Image      string        `json:"image"`
		W2C        [3][4]float64 `json:"world_to_camera"`
		Intrinsics [3][3]float64 `json:"intrinsics"`
		Width      int           `json:"width"`
		Height     int           `json:"height"`
	}
	var cj []camJSON
	for i, r := range rs {
		base := strings.TrimSuffix(filepath.Base(paths[i]), filepath.Ext(paths[i]))
		dir := o.dir
		if dir == "" {
			dir = filepath.Dir(paths[i])
		}
		ow, oh := imgs[i].Bounds().Dx(), imgs[i].Bounds().Dy()
		inv := make([]float32, len(r.Depth))
		for p, d := range r.Depth {
			inv[p] = 1 / d
		}
		if err := writeGray(filepath.Join(dir, base+".depth.png"), Bilinear(inv, r.W, r.H, ow, oh), ow, oh, lo, hi); err != nil {
			return err
		}
		if o.sky {
			cl, ch := slices.Min(r.Conf), slices.Max(r.Conf)
			if err := writeGray(filepath.Join(dir, base+".conf.png"), Bilinear(r.Conf, r.W, r.H, ow, oh), ow, oh, cl, ch); err != nil {
				return err
			}
		}
		if o.raw {
			if err := writeF32(filepath.Join(dir, base+".depth.f32"), r.Depth); err != nil {
				return err
			}
		}
		c := cams[i]
		var j camJSON
		j.Image, j.Width, j.Height = paths[i], r.W, r.H
		for a := 0; a < 3; a++ {
			copy(j.W2C[a][:], c.W2C[a*4:a*4+4])
		}
		j.Intrinsics = [3][3]float64{{c.Fx, 0, c.Cx}, {0, c.Fy, c.Cy}, {0, 0, 1}}
		cj = append(cj, j)
		logf("%s: camera at (%.3f, %.3f, %.3f) of the reference's frame, fx %.1f", paths[i],
			-(c.W2C[0]*c.W2C[3] + c.W2C[4]*c.W2C[7] + c.W2C[8]*c.W2C[11]),
			-(c.W2C[1]*c.W2C[3] + c.W2C[5]*c.W2C[7] + c.W2C[9]*c.W2C[11]),
			-(c.W2C[2]*c.W2C[3] + c.W2C[6]*c.W2C[7] + c.W2C[10]*c.W2C[11]), c.Fx)
	}
	dir := o.dir
	if dir == "" {
		dir = filepath.Dir(paths[0])
	}
	b, _ := json.MarshalIndent(cj, "", " ")
	if err := os.WriteFile(filepath.Join(dir, "cameras.json"), append(b, '\n'), 0o644); err != nil {
		return err
	}
	if ply != "" {
		if err := writePLY(ply, imgs, rs, cams); err != nil {
			return err
		}
	}
	logf("%d views in %s; cameras.json in %s", len(paths), console.Duration(time.Since(t0)), dir)
	return nil
}

// writePLY writes every view's pixels as points in the reference view's frame (world = R^T
// (camera - t) of each view's world-to-camera [R | t]), colored by the picture.
func writePLY(path string, imgs []image.Image, rs []*depth.Result, cams []depth.Camera) error {
	var body []byte
	n := 0
	for i, r := range rs {
		c := cams[i]
		px := lfm3vl.Resize(imgs[i], r.W, r.H, "bilinear")
		for y := 0; y < r.H; y++ {
			for x := 0; x < r.W; x++ {
				d := float64(r.Depth[y*r.W+x])
				pc := [3]float64{(float64(x) + 0.5 - c.Cx) / c.Fx * d, (float64(y) + 0.5 - c.Cy) / c.Fy * d, d}
				var pw [3]float32
				for a := 0; a < 3; a++ {
					var s float64
					for k := 0; k < 3; k++ {
						s += c.W2C[k*4+a] * (pc[k] - c.W2C[k*4+3])
					}
					pw[a] = float32(s)
				}
				for _, v := range pw {
					body = binary.LittleEndian.AppendUint32(body, math.Float32bits(v))
				}
				o := 3 * (y*r.W + x)
				body = append(body, px[o], px[o+1], px[o+2])
				n++
			}
		}
	}
	head := fmt.Sprintf("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n", n)
	return os.WriteFile(path, append([]byte(head), body...), 0o644)
}

func decode(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func writeRGB(path string, n []float32, w, h int) error {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	c := func(v float32) uint8 { return uint8(math.Round(float64(min(1, max(0, (v+1)/2))) * 255)) }
	for i := 0; i < w*h; i++ {
		img.SetRGBA(i%w, i/w, color.RGBA{c(n[3*i]), c(n[3*i+1]), c(n[3*i+2]), 255})
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

func run(ctx context.Context, e *depth.Estimator, path string, o opts, logf func(string, ...any)) error {
	outDir, raw, sky, keep := o.dir, o.raw, o.sky, o.keep
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	img, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return err
	}
	t0 := time.Now()
	r, err := e.Estimate(ctx, img)
	if err != nil {
		return err
	}
	base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	dir := outDir
	if dir == "" {
		dir = filepath.Dir(path)
	}
	ow, oh := img.Bounds().Dx(), img.Bounds().Dy()
	metric := ""
	if r.Metric && (o.focal > 0 || o.fov > 0) {
		f := o.focal
		if f == 0 {
			f = float64(ow) / 2 / math.Tan(o.fov*math.Pi/360)
		}
		r.Meters(f, f, ow, oh)
		c := r.Depth[(r.H/2)*r.W+r.W/2]
		near := slices.Min(r.Depth)
		metric = fmt.Sprintf(", %.2f m at the centre, %.2f m nearest", c, near)
	}
	if keep {
		ow, oh = r.W, r.H
	}
	inv := make([]float32, len(r.Depth))
	lo, hi := float32(math.Inf(1)), float32(math.Inf(-1))
	for i, d := range r.Depth {
		inv[i] = 1 / d
		lo, hi = min(lo, inv[i]), max(hi, inv[i])
	}
	g := Bilinear(inv, r.W, r.H, ow, oh)
	out := filepath.Join(dir, base+".depth.png")
	if err := writeGray(out, g, ow, oh, lo, hi); err != nil {
		return err
	}
	if sky && r.Sky != nil {
		if err := writeGray(filepath.Join(dir, base+".sky.png"), Bilinear(r.Sky, r.W, r.H, ow, oh), ow, oh, 0, 1); err != nil {
			return err
		}
	}
	if sky && r.Conf != nil { // the any-view models' confidence in place of the sky
		lo, hi := slices.Min(r.Conf), slices.Max(r.Conf)
		if err := writeGray(filepath.Join(dir, base+".conf.png"), Bilinear(r.Conf, r.W, r.H, ow, oh), ow, oh, lo, hi); err != nil {
			return err
		}
	}
	if raw {
		b := make([]byte, 4*len(r.Depth))
		for i, d := range r.Depth {
			binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(d))
		}
		if err := os.WriteFile(filepath.Join(dir, base+".depth.f32"), b, 0o644); err != nil {
			return err
		}
	}
	logf("%s: %dx%d (ran at %dx%d) in %s%s", out, ow, oh, r.W, r.H, console.Duration(time.Since(t0)), metric)
	return nil
}

// Bilinear resizes the map xs (w x h, row major) to ow x oh, pixel centres aligned
// (align_corners=False).
func Bilinear(xs []float32, w, h, ow, oh int) []float32 {
	if w == ow && h == oh {
		return xs
	}
	out := make([]float32, ow*oh)
	for y := 0; y < oh; y++ {
		sy := math.Max(0, (float64(y)+0.5)*float64(h)/float64(oh)-0.5)
		y0 := min(int(sy), h-1)
		y1 := min(y0+1, h-1)
		fy := float32(sy - float64(y0))
		for x := 0; x < ow; x++ {
			sx := math.Max(0, (float64(x)+0.5)*float64(w)/float64(ow)-0.5)
			x0 := min(int(sx), w-1)
			x1 := min(x0+1, w-1)
			fx := float32(sx - float64(x0))
			a := xs[y0*w+x0] + fx*(xs[y0*w+x1]-xs[y0*w+x0])
			b := xs[y1*w+x0] + fx*(xs[y1*w+x1]-xs[y1*w+x0])
			out[y*ow+x] = a + fy*(b-a)
		}
	}
	return out
}

func writeGray(path string, xs []float32, w, h int, lo, hi float32) error {
	img := image.NewGray(image.Rect(0, 0, w, h))
	scale := float32(255)
	if hi > lo {
		scale /= hi - lo
	}
	for i, v := range xs {
		img.SetGray(i%w, i/w, color.Gray{Y: uint8(math.Round(float64(min(255, max(0, (v-lo)*scale)))))})
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
