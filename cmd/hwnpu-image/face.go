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
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/vision/face"
)

// FaceDetectCmd is "hwnpu-image face-detect": the faces in pictures with YuNet (OpenCV Zoo's
// face_detection_yunet_2023mar, MIT; internal/vision/face), on AMD's NPU (exports of
// tools/amd/iron_yunet.py) or on the CPU (the ONNX file with ONNX Runtime). models.yaml's
// device says which for the machine's NPU (YuNet: the CPU on AMD, where the NPU took 16 times
// the energy); --device and HWNPU_DEVICE override it.
//
// For each picture it prints one line of JSON: the faces' boxes (x, y, w, h), scores and five
// landmarks (the right eye, the left eye, the nose tip, the mouth's right and left corners) in
// the picture's pixels. With --draw it also writes NAME.faces.png with them drawn.
type FaceDetectCmd struct {
	Pictures []string `arg:"" type:"existingfile" help:"The pictures."`
	Model    string   `short:"m" placeholder:"NAME|EXPORT" default:"face-detect" help:"The model: a name, a family or a use of models.yaml (face-detect: YuNet), an export's directory (tools/amd/iron_yunet.py, the NPU) or an ONNX file (the CPU)."`
	Device   string   `enum:",npu,cpu" default:"" help:"Where to run: npu or cpu (default: models.yaml's device for this machine's NPU, or HWNPU_DEVICE)."`
	Threads  int      `default:"1" help:"The CPU's threads (1: the least CPU time and energy)."`
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
	d, err := c.open(ctx, logf)
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
	console.Running()
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

// open loads the detector where it runs: the NPU (an export, AMD's) or the CPU (the ONNX file).
func (c *FaceDetectCmd) open(ctx context.Context, logf func(string, ...any)) (*face.Detector, error) {
	model := expandHome(c.Model)
	vendor := npuVendor(ctx)
	if isPath(c.Model) {
		device := c.Device
		if device == "" {
			device = "npu"
			if strings.EqualFold(filepath.Ext(model), ".onnx") {
				device = "cpu"
			}
		}
		if device == "cpu" {
			return face.OpenCPU(model, c.Threads)
		}
		return openFaceNPU(ctx, model)
	}
	m, err := resolve(c.Model, chat.KindFace)
	if err != nil {
		return nil, err
	}
	device, why, err := chat.ChooseDevice(m, vendor, c.Device)
	if err != nil {
		return nil, err
	}
	if device == "npu" && vendor != "amd" && vendor != "apple" && vendor != "intel" {
		// AMD's NPU has YuNet's graph (iron_yunet.py), Apple's and Intel's the ONNX graph as
		// Core ML and OpenVINO IR
		if c.Device == "npu" {
			return nil, fmt.Errorf("%s runs on AMD's, Apple's and Intel's NPU (this machine's NPU: %s); --device cpu runs it on the CPU", m.Name, orNone(vendor))
		}
		device, why = "cpu", fmt.Sprintf("no %s graph for this NPU (%s)", m.Name, orNone(vendor))
	}
	logf("%s on the %s (%s)", m.Name, device, why)
	if device == "npu" && (vendor == "apple" || vendor == "intel") {
		path, err := faceONNX(ctx, m)
		if err != nil {
			return nil, err
		}
		var opts []hwnpu.Option
		if vendor == "apple" {
			opts = append(opts, hwnpu.WithPlacement(hwnpu.PlacementStrict))
		}
		dev, backend, err := llm.OpenDevice(ctx, "", cacheDirOf(""), opts...)
		if err != nil {
			return nil, err
		}
		d, err := face.OpenProgram(ctx, dev, backend, path, filepath.Dir(path))
		if err != nil {
			dev.Close()
			return nil, err
		}
		d.OnClose = func() { dev.Close() }
		return d, nil
	}
	if device == "cpu" {
		path, err := faceONNX(ctx, m)
		if err != nil {
			return nil, err
		}
		return face.OpenCPU(path, c.Threads)
	}
	dir, err := amdExport(m, "tools/amd/iron_yunet.py")
	if err != nil {
		return nil, err
	}
	return openFaceNPU(ctx, dir)
}

// openFaceNPU loads the export in dir onto AMD's NPU; the device closes with the detector.
func openFaceNPU(ctx context.Context, dir string) (*face.Detector, error) {
	dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
	if err != nil {
		return nil, err
	}
	d, err := face.Open(ctx, dev, dir)
	if err != nil {
		dev.Close()
		return nil, err
	}
	d.OnClose = func() { dev.Close() }
	return d, nil
}

// faceONNX is the model's ONNX file: in ~/models/NAME (a download for iron_yunet.py) when
// there, else fetched into hwnpu's models directory (models.yaml pins its sha256).
func faceONNX(ctx context.Context, m chat.Model) (string, error) {
	const file = "face_detection_yunet_2023mar.onnx"
	home, _ := os.UserHomeDir()
	if p := filepath.Join(home, "models", filepath.Base(m.Repo), file); fileExists(p) {
		return p, nil
	}
	p := filepath.Join(chat.ModelsDir(), m.Name, file)
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		return "", err
	}
	if err := chat.Fetch(ctx, m.Source(), file, p, os.Stderr); err != nil {
		return "", err
	}
	return p, nil
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.Mode().IsRegular()
}

// npuVendor is the vendor of this machine's NPU ("amd", "apple", "intel"), "" without one
// (chat.HereVendor: HWNPU_BACKEND=openvino is an Intel machine).
func npuVendor(context.Context) string { return chat.HereVendor() }

func orNone(v string) string {
	if v == "" {
		return "none"
	}
	return v
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
