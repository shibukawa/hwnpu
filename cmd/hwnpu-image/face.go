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

	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
)

// FaceDetectCmd is "hwnpu-image face-detect": the faces in pictures with YuNet (OpenCV Zoo's
// face_detection_yunet_2023mar, MIT; ai/vision's Faces), on AMD's NPU (exports of
// tools/amd/iron_yunet.py), Apple's and Intel's NPU (the ONNX graph converted) or on the CPU
// (the ONNX file with ONNX Runtime). models.yaml's device says which for the machine's NPU
// (YuNet: the CPU on AMD, where the NPU took 16 times the energy); --device and HWNPU_DEVICE
// override it.
//
// For each picture it prints one line of JSON: the faces' boxes (x, y, w, h), scores and five
// landmarks (the right eye, the left eye, the nose tip, the mouth's right and left corners) in
// the picture's pixels. With --draw it also writes NAME.faces.png with them drawn.
type FaceDetectCmd struct {
	Pictures    []string `arg:"" type:"existingfile" help:"The pictures."`
	Model       string   `short:"m" placeholder:"NAME|EXPORT" default:"face-detect" help:"The model: a name, a family or a use of models.yaml (face-detect: YuNet), an export's directory (tools/amd/iron_yunet.py, the NPU) or an ONNX file (the CPU)."`
	Device      string   `enum:",npu,cpu" default:"" help:"Where to run: npu or cpu (default: models.yaml's device for this machine's NPU, or HWNPU_DEVICE)."`
	Threads     int      `default:"1" help:"The CPU's threads (1: the least CPU time and energy)."`
	Size        string   `placeholder:"WxH" help:"The export's size to run at (default: the one the picture is scaled down least to fit)."`
	Score       float32  `default:"0.9" help:"The least score a face is kept with."`
	NMS         float32  `name:"nms" default:"0.3" help:"The overlap (intersection over union) above which the lower of two faces goes."`
	Draw        bool     `help:"Also write NAME.faces.png: the picture with the boxes and landmarks."`
	Out         string   `short:"o" type:"path" placeholder:"DIR" help:"The directory --draw writes into (default: next to each picture)."`
	Repeat      int      `default:"1" help:"Run each picture this many times and report the mean time (a measure of a camera's frames)."`
	backendFlag `embed:""`
}

// Run finds the faces of each picture.
func (c *FaceDetectCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	o := vision.FaceOptions{Score: orAll(c.Score), NMS: c.NMS, Log: c.Repeat <= 1}
	if c.Size != "" {
		if _, err := fmt.Sscanf(c.Size, "%dx%d", &o.Size[0], &o.Size[1]); err != nil {
			return fmt.Errorf("--size %q: not WxH", c.Size)
		}
	}
	useBackend(c.Backend)
	how := made{device: c.Device}
	how.threads(c.Threads)
	name := c.Model
	if isPath(c.Model) {
		// an export's directory (the NPU) or an ONNX file (the CPU): a model like YuNet
		path := expandHome(c.Model)
		if strings.EqualFold(filepath.Ext(path), ".onnx") {
			how.program("onnx", path)
		} else {
			how.program("export", path)
		}
		name = "face-detect"
	}
	d, rt, _, err := load(ctx, name, vision.Faces, how, lw)
	if err != nil {
		return err
	}
	defer rt.Close()
	console.Running()
	enc := json.NewEncoder(os.Stdout)
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		var faces []vision.Face
		t0 := time.Now()
		for i := 0; i < max(1, c.Repeat); i++ {
			if faces, err = d.DetectWith(ctx, img, o); err != nil {
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

// npuVendor is the vendor of this machine's NPU ("amd", "apple", "intel"), "" without one
// (chat.HereVendor: HWNPU_BACKEND=openvino is an Intel machine).
func npuVendor(context.Context) string { return chat.HereVendor() }

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

func faceJSON(path string, faces []vision.Face) any {
	out := struct {
		Picture string    `json:"picture"`
		Faces   []faceOut `json:"faces"`
	}{Picture: path, Faces: []faceOut{}}
	r := func(x float64) float32 { return float32(int(float32(x)*10+0.5)) / 10 }
	for _, f := range faces {
		o := faceOut{Box: [4]float32{r(f.Box.Min.X), r(f.Box.Min.Y), r(f.Box.Dx()), r(f.Box.Dy())}, Score: float32(int(float32(f.Score)*1000+0.5)) / 1000}
		for i, l := range landmarks(f) {
			o.Landmarks[i] = [2]float32{r(l.X), r(l.Y)}
		}
		out.Faces = append(out.Faces, o)
	}
	return out
}

// landmarks are a face's five landmarks in YuNet's order: the right eye, the left eye, the
// nose, the mouth's right and left corners.
func landmarks(f vision.Face) [5]vision.Point {
	return [5]vision.Point{f.RightEye, f.LeftEye, f.Nose, f.MouthRight, f.MouthLeft}
}

// writeFaces writes img with each face's box (green) and landmarks (OpenCV's demo colors:
// the eyes blue and red, the nose green, the mouth's corners magenta and yellow).
func writeFaces(path string, img image.Image, faces []vision.Face) error {
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
		x0, y0, x1, y1 := int(f.Box.Min.X), int(f.Box.Min.Y), int(float32(f.Box.Max.X)), int(float32(f.Box.Max.Y))
		rect(x0, y0, x1, y0+t, green)
		rect(x0, y1-t, x1, y1, green)
		rect(x0, y0, x0+t, y1, green)
		rect(x1-t, y0, x1, y1, green)
		for i, l := range landmarks(f) {
			x, y := int(l.X), int(l.Y)
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
