// hwnpu-depth estimates the depth of pictures on AMD's NPU with Depth Anything 3's monocular
// model (depth-anything/DA3MONO-LARGE, Apache-2.0), an export of tools/amd/iron_da3.py
// (internal/vision/depth):
//
//	hwnpu-depth -model ~/models/da3mono-large-504.amd-npu2 photo.jpg
//
// For each picture it writes NAME.depth.png next to it (or into -out): the inverse depth
// (near is bright) normalized to 8 bits, at the picture's size. -raw also writes the depth as
// raw float32 (NAME.depth.f32, the export's size, row major) and -sky the sky mask. The depth
// is relative: larger is farther, up to an unknown scale.
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
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/vision/depth"
)

func defaultModel() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, "models", "da3mono-large-504.amd-npu2")
}

func main() {
	model := flag.String("model", defaultModel(), "an export of tools/amd/iron_da3.py")
	outDir := flag.String("out", "", "the directory to write into (default: next to each picture)")
	raw := flag.Bool("raw", false, "also write the depth as raw float32 (NAME.depth.f32, the export's size)")
	sky := flag.Bool("sky", false, "also write the sky mask (NAME.sky.png)")
	keep := flag.Bool("export-size", false, "write the picture at the export's size instead of the input's")
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
	e, err := depth.Open(ctx, dev, *model)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()
	e.Logf = logf
	W, H := e.Size()
	logf("loaded %s (%dx%d) in %s", filepath.Base(*model), W, H, console.Duration(time.Since(t0)))
	for _, path := range flag.Args() {
		if err := run(ctx, e, path, *outDir, *raw, *sky, *keep, logf); err != nil {
			log.Fatalf("%s: %v", path, err)
		}
	}
}

func run(ctx context.Context, e *depth.Estimator, path, outDir string, raw, sky, keep bool, logf func(string, ...any)) error {
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
	if sky {
		if err := writeGray(filepath.Join(dir, base+".sky.png"), Bilinear(r.Sky, r.W, r.H, ow, oh), ow, oh, 0, 1); err != nil {
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
	logf("%s: %dx%d in %s", out, ow, oh, console.Duration(time.Since(t0)))
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
