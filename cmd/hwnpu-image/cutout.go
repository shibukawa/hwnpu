package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/vision/cutout"
)

// CutoutCmd is "hwnpu-image cutout": the people of pictures apart from what is behind them, with
// MediaPipe's Selfie Segmentation (Google, Apache-2.0; internal/vision/cutout) on the CPU (the
// ONNX file with ONNX Runtime). The model is 0.46 MB of weights and takes a few milliseconds a
// picture on one thread, so no NPU runs it.
//
// For each picture it writes NAME.cutout.png (the picture with what is behind the people made
// clear) or, with --matte, NAME.matte.png (how much of each pixel is a person, white a person),
// and prints one line of JSON: the file and the share of the picture that is a person.
type CutoutCmd struct {
	Pictures []string `arg:"" type:"existingfile" help:"The pictures."`
	Model    string   `short:"m" placeholder:"NAME|FILE" default:"cutout" help:"The model: a name, a family or a use of models.yaml (cutout: selfie-segmentation at 256 x 256; selfie-segmentation-landscape at 256 x 144), or an ONNX file."`
	Device   string   `enum:",npu,cpu" default:"" help:"Where to run: the CPU only (npu is an error: no NPU has a program of it)."`
	Threads  int      `default:"1" help:"The CPU's threads (1: the least CPU time and energy)."`
	Matte    bool     `help:"Write the matte (NAME.matte.png: white a person, black what is behind) in place of the cutout."`
	Out      string   `short:"o" type:"path" placeholder:"DIR" help:"The directory to write into (default: next to each picture)."`
	Repeat   int      `default:"1" help:"Run each picture this many times and report the mean time (a measure of a camera's frames)."`
}

// Run cuts the people out of each picture.
func (c *CutoutCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	m, err := c.open(ctx, logf)
	if err != nil {
		return err
	}
	defer m.Close()
	console.Running()
	enc := json.NewEncoder(os.Stdout)
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		if c.Repeat <= 1 {
			m.Logf = logf
		}
		var a []float32
		t0 := time.Now()
		for i := 0; i < max(1, c.Repeat); i++ {
			if a, err = m.Alpha(ctx, img); err != nil {
				return err
			}
			if i == 0 && c.Repeat > 1 {
				t0 = time.Now() // the first warms the session
			}
		}
		if c.Repeat > 1 {
			logf("%s: %.2f ms a picture over %d runs", filepath.Base(p), time.Since(t0).Seconds()*1000/float64(c.Repeat-1), c.Repeat-1)
		}
		dir := c.Out
		if dir == "" {
			dir = filepath.Dir(p)
		}
		stem := strings.TrimSuffix(filepath.Base(p), filepath.Ext(p))
		var (
			out string
			pic image.Image
		)
		if c.Matte {
			b := img.Bounds()
			g := image.NewGray(image.Rect(0, 0, b.Dx(), b.Dy()))
			for i, v := range cutout.Stretch(a, m.W, m.H, b.Dx(), b.Dy()) {
				g.Pix[i] = uint8(min(max(v, 0), 1)*255 + 0.5)
			}
			out, pic = filepath.Join(dir, stem+".matte.png"), g
		} else {
			out, pic = filepath.Join(dir, stem+".cutout.png"), cutout.Cut(img, a, m.W, m.H)
		}
		if err := writePNG(out, pic); err != nil {
			return err
		}
		logf("wrote %s", out)
		if err := enc.Encode(struct {
			Picture string  `json:"picture"`
			Out     string  `json:"out"`
			Person  float64 `json:"person"` // the share of the picture that is a person
		}{p, out, float64(int(cutout.Share(a)*1000+0.5)) / 1000}); err != nil {
			return err
		}
	}
	return nil
}

// open loads the model on the CPU: an ONNX file, or the one of a model of the definitions
// (fetched when missing; models.yaml pins its sha256).
func (c *CutoutCmd) open(ctx context.Context, logf func(string, ...any)) (*cutout.Model, error) {
	if c.Device == "npu" {
		return nil, fmt.Errorf("the cutout model runs on the CPU only: no NPU has a program of it")
	}
	if isPath(c.Model) {
		return cutout.OpenCPU(expandHome(c.Model), c.Threads)
	}
	m, err := resolve(c.Model, chat.KindCutout)
	if err != nil {
		return nil, err
	}
	logf("%s on the cpu (%d thread(s))", m.Name, max(1, c.Threads))
	path, err := chat.Fetch(ctx, m.Source(), "onnx/model.onnx", os.Stderr)
	if err != nil {
		return nil, err
	}
	return cutout.OpenCPU(path, c.Threads)
}

func writePNG(path string, img image.Image) error {
	w, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(w, img); err != nil {
		w.Close()
		return err
	}
	return w.Close()
}
