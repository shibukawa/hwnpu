// hwnpu-depth estimates the depth of pictures on AMD's NPU with Depth Anything 3's monocular
// model (depth-anything/DA3MONO-LARGE, Apache-2.0), an export of tools/amd/iron_da3.py
// (internal/vision/depth):
//
//	hwnpu-depth -model ~/models/da3mono-large.amd-npu2 photo.jpg
//
// MoGe-2 (Ruicheng/moge-2-*-normal, MIT), an export of tools/amd/iron_moge.py, gives metric
// depth (and a point map, normals and a mask) without the camera's focal length: it solves it.
//
// For each picture it writes NAME.depth.png next to it (or into -out): the inverse depth
// (near is bright) normalized to 8 bits, at the picture's size. -raw also writes the depth as
// raw float32 (NAME.depth.f32, row major, at the export's size it ran at, printed) and -sky the
// sky mask. The export holds several sizes (the longest side 504) and the picture is stretched
// to the one nearest its aspect.
//
// The depth is relative (larger is farther, up to an unknown scale), except with an export of
// DA3METRIC-LARGE given the camera's focal length (-focal, in the picture's pixels) or
// horizontal field of view (-fov, degrees): then it is meters, and -raw writes meters.
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/vision/depth"
	"github.com/shibukawa/hwnpu/internal/vision/moge"
)

func defaultModel() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "models", "da3mono-large.amd-npu2")
}

func main() {
	model := flag.String("model", defaultModel(), "an export of tools/amd/iron_da3.py or tools/amd/iron_moge.py")
	outDir := flag.String("out", "", "the directory to write into (default: next to each picture)")
	raw := flag.Bool("raw", false, "also write the depth as raw float32 (NAME.depth.f32, the export's size)")
	sky := flag.Bool("sky", false, "also write the sky mask (NAME.sky.png), or with DA3-SMALL / BASE the confidence (NAME.conf.png)")
	keep := flag.Bool("export-size", false, "write the picture at the export's size instead of the input's")
	focal := flag.Float64("focal", 0, "a metric export: the focal length in the picture's pixels (fx = fy)")
	fov := flag.Float64("fov", 0, "a metric export: the horizontal field of view in degrees, in place of -focal")
	normal := flag.Bool("normal", false, "MoGe-2: also write the normals (NAME.normal.png, (n + 1) / 2 as RGB)")
	flag.Parse()
	if flag.NArg() == 0 {
		log.Fatal("hwnpu-depth: give one or more pictures")
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	t0 := time.Now()
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		log.Fatal(err)
	}
	defer dev.Close()
	iron.ShareExport(*model, logf)
	o := opts{dir: *outDir, raw: *raw, sky: *sky, keep: *keep, focal: *focal, fov: *fov, normal: *normal}
	if moge.IsExport(*model) {
		m, err := moge.Open(ctx, dev, *model)
		if err != nil {
			log.Fatal(err)
		}
		defer m.Close()
		m.Logf = logf
		logf("%s: MoGe-2, metric depth, sizes %v", filepath.Base(*model), m.Sizes)
		for _, path := range flag.Args() {
			if err := runMoge(ctx, m, path, o, logf); err != nil {
				log.Fatalf("%s: %v", path, err)
			}
		}
		return
	}
	e, err := depth.Open(ctx, dev, *model)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()
	e.Logf = logf
	kind := "relative depth"
	if e.Metric {
		kind = "metric depth"
		if *focal == 0 && *fov == 0 {
			kind += " (relative without -focal or -fov)"
		}
	}
	logf("%s: %s, sizes %v (%s)", filepath.Base(*model), kind, e.Sizes, console.Duration(time.Since(t0)))
	for _, path := range flag.Args() {
		if err := run(ctx, e, path, o, logf); err != nil {
			log.Fatalf("%s: %v", path, err)
		}
	}
}

type opts struct {
	dir                    string
	raw, sky, keep, normal bool
	focal, fov             float64
}

// runMoge writes MoGe-2's depth (meters) of the picture at path: NAME.depth.png (the inverse
// depth, the masked pixels black), -raw NAME.depth.f32 (+Inf where masked) and NAME.points.f32
// (x, y, z a pixel), -sky NAME.mask.png, -normal NAME.normal.png.
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

func decode(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	return img, err
}

func writeF32(path string, xs []float32) error {
	b := make([]byte, 4*len(xs))
	for i, d := range xs {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(d))
	}
	return os.WriteFile(path, b, 0o644)
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
