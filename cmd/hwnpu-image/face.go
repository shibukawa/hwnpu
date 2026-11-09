package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/vision/face"
)

// FaceDetectCmd is "hwnpu-image face-detect": the faces in pictures on AMD's NPU with YuNet
// (OpenCV Zoo's face_detection_yunet_2023mar, MIT; exports of tools/amd/iron_yunet.py,
// internal/vision/face).
//
// For each picture it prints one line of JSON: the faces' boxes (x, y, w, h), scores and five
// landmarks (the right eye, the left eye, the nose tip, the mouth's right and left corners) in
// the picture's pixels. With --draw it also writes NAME.faces.png with them drawn.
type FaceDetectCmd struct {
	Pictures []string `arg:"" type:"existingfile" help:"The pictures."`
	Model    string   `short:"m" placeholder:"NAME|EXPORT" default:"face-detect" help:"The model: a name, a family or a use of models.yaml (face-detect: YuNet), run from ~/models/NAME.amd-npu2, or an export's directory (tools/amd/iron_yunet.py)."`
	Size     string   `placeholder:"WxH" help:"The export's size to run at (default: the one the picture is scaled down least to fit)."`
	Score    float32  `default:"0.9" help:"The least score a face is kept with."`
	NMS      float32  `name:"nms" default:"0.3" help:"The overlap (intersection over union) above which the lower of two faces goes."`
	Draw     bool     `help:"Also write NAME.faces.png: the picture with the boxes and landmarks."`
	Out      string   `short:"o" type:"path" placeholder:"DIR" help:"The directory --draw writes into (default: next to each picture)."`
	Repeat   int      `default:"1" help:"Run each picture this many times and report the mean time (a measure of a camera's frames)."`
}

// Run finds the faces of each picture.
func (c *FaceDetectCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		return err
	}
	defer dev.Close()
	model := expandHome(c.Model)
	if !isPath(c.Model) {
		m, err := resolve(c.Model, chat.KindFace)
		if err != nil {
			return err
		}
		if model, err = amdExport(m, "tools/amd/iron_yunet.py"); err != nil {
			return err
		}
	}
	d, err := face.Open(ctx, dev, model)
	if err != nil {
		return err
	}
	defer d.Close()
	d.Score, d.NMS = c.Score, c.NMS
	if c.Size != "" {
		if _, err := fmt.Sscanf(c.Size, "%dx%d", &d.Size[0], &d.Size[1]); err != nil {
			return fmt.Errorf("--size %q: not WxH", c.Size)
		}
	}
	enc := json.NewEncoder(os.Stdout)
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		if c.Repeat <= 1 {
			d.Logf = logf
		}
		var faces []face.Face
		t0 := time.Now()
		for i := 0; i < max(1, c.Repeat); i++ {
			if faces, err = d.Detect(ctx, img); err != nil {
				return err
			}
			if i == 0 && c.Repeat > 1 {
				t0 = time.Now() // the first loads the graph
			}
		}
		if c.Repeat > 1 {
			logf("%s: %.2f ms a picture over %d runs", filepath.Base(p), time.Since(t0).Seconds()*1000/float64(c.Repeat-1), c.Repeat-1)
		}
		if err := enc.Encode(faceJSON(p, faces)); err != nil {
			return err
		}
		if c.Draw {
			dir := c.Out
			if dir == "" {
				dir = filepath.Dir(p)
			}
			out := filepath.Join(dir, strings.TrimSuffix(filepath.Base(p), filepath.Ext(p))+".faces.png")
			if err := writeFaces(out, img, faces); err != nil {
				return err
			}
			logf("wrote %s", out)
		}
	}
	return nil
}

func readPicture(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return img, nil
}

type faceOut struct {
	Box       [4]float32    `json:"box"` // x, y, w, h
	Score     float32       `json:"score"`
	Landmarks [5][2]float32 `json:"landmarks"`
}

func faceJSON(path string, faces []face.Face) any {
	out := struct {
		Picture string    `json:"picture"`
		Faces   []faceOut `json:"faces"`
	}{Picture: path, Faces: []faceOut{}}
	r := func(x float32) float32 { return float32(int(x*10+0.5)) / 10 }
	for _, f := range faces {
		o := faceOut{Box: [4]float32{r(f.X), r(f.Y), r(f.W), r(f.H)}, Score: float32(int(f.Score*1000+0.5)) / 1000}
		for i, l := range f.Landmarks {
			o.Landmarks[i] = [2]float32{r(l[0]), r(l[1])}
		}
		out.Faces = append(out.Faces, o)
	}
	return out
}

// writeFaces writes img with each face's box (green) and landmarks (OpenCV's demo colors:
// the eyes blue and red, the nose green, the mouth's corners magenta and yellow).
func writeFaces(path string, img image.Image, faces []face.Face) error {
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), img, b.Min, draw.Src)
	t := max(1, min(b.Dx(), b.Dy())/300)
	rect := func(x0, y0, x1, y1 int, c color.Color) {
		draw.Draw(dst, image.Rect(x0, y0, x1, y1).Intersect(dst.Bounds()), image.NewUniform(c), image.Point{}, draw.Src)
	}
	green := color.RGBA{0, 255, 0, 255}
	marks := []color.RGBA{{0, 0, 255, 255}, {255, 0, 0, 255}, {0, 255, 0, 255}, {255, 0, 255, 255}, {255, 255, 0, 255}}
	for _, f := range faces {
		x0, y0, x1, y1 := int(f.X), int(f.Y), int(f.X+f.W), int(f.Y+f.H)
		rect(x0, y0, x1, y0+t, green)
		rect(x0, y1-t, x1, y1, green)
		rect(x0, y0, x0+t, y1, green)
		rect(x1-t, y0, x1, y1, green)
		for i, l := range f.Landmarks {
			x, y := int(l[0]), int(l[1])
			rect(x-2*t, y-2*t, x+2*t, y+2*t, marks[i])
		}
	}
	w, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(w, dst); err != nil {
		w.Close()
		return err
	}
	return w.Close()
}
