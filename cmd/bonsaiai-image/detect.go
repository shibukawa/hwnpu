package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/shibukawa/bonsaiai/ai/vision"
	"github.com/shibukawa/bonsaiai/internal/console"
	"github.com/shibukawa/bonsaiai/internal/imagedev"
)

// DetectCmd is "bonsaiai-image detect": the objects in pictures (COCO's 80 classes) with RTMDet
// (OpenMMLab MMDetection, Apache-2.0; ai/vision's Detect), on Apple's Neural Engine or Intel's
// NPU, converted the first time, on AMD's NPU from an export of tools/amd/iron_onnxcnn.py (made
// with IRON on the first run) where models.yaml says so, and elsewhere or with --device cpu on
// the CPU, the ONNX file with ONNX Runtime (--cpu: the converted program on the CPU
// interpreter, for checking a conversion). The network up to its head runs on the NPU;
// decoding the boxes and suppressing the overlaps on the host.
//
// For each picture it prints one line of JSON: each object's box (x1, y1, x2, y2 in the
// picture's pixels), label, class and score. With --draw it also writes NAME.objects.png.
type DetectCmd struct {
	Pictures    []string `arg:"" type:"existingfile" help:"The pictures (JPEG or PNG)."`
	Model       string   `short:"m" default:"detect" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (detect, rtmdet, rtmdet:tiny)."`
	Score       float32  `default:"0.3" help:"The least score of an object to report."`
	NMS         float32  `name:"nms" default:"0.65" help:"The overlap (intersection over union) above which the lower of two objects of a class goes."`
	Class       []string `placeholder:"LABEL" help:"Only these classes (person,car,...; bonsaiai-image detect --class help lists them)."`
	Draw        bool     `help:"Also write NAME.objects.png: the picture with the boxes."`
	Out         string   `short:"o" type:"path" placeholder:"DIR" help:"The directory --draw writes into (default: next to each picture)."`
	Repeat      int      `default:"1" help:"Run each picture this many times and report the mean time."`
	CPU         bool     `name:"cpu" help:"Run the converted program on the CPU interpreter instead of the NPU (slow; for checking a conversion)."`
	Device      string   `enum:",npu,cpu" default:"" help:"Where to run: npu or cpu (ONNX Runtime; default: models.yaml's device for this machine's NPU, or BONSAIAI_DEVICE)."`
	Threads     int      `default:"1" help:"The CPU's threads with --device cpu (1: the least CPU time and energy)."`
	backendFlag `embed:""`
}

// Run finds the objects of each picture.
func (c *DetectCmd) Run(ctx context.Context) error {
	if err := outDir(c.Out); err != nil {
		return err
	}
	if slices.Contains(c.Class, "help") {
		fmt.Println(strings.Join(vision.COCO, ", "))
		return nil
	}
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	useBackend(c.Backend)
	how := made{device: c.Device}
	how.threads(c.Threads)
	how.interpreter(c.CPU)
	d, rt, loaded, err := load(ctx, c.Model, vision.Detect, how, lw)
	if err != nil {
		return err
	}
	defer rt.Close()
	defer imagedev.NPUHeld(logf)()
	logf("%s, %s", loaded, d.Where())
	o := vision.DetectOptions{Score: orAll(c.Score), NMS: c.NMS}
	for _, l := range c.Class {
		for _, k := range strings.Split(l, ",") {
			k = strings.TrimSpace(k)
			if !slices.Contains(d.Classes(), k) {
				return fmt.Errorf("--class %q is not one of the model's classes (--class help lists them)", k)
			}
			o.Classes = append(o.Classes, k)
		}
	}
	console.Running()
	enc := json.NewEncoder(os.Stdout)
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		var found *vision.Objects
		start := time.Now()
		for i := range max(1, c.Repeat) {
			if i == 1 {
				start = time.Now()
			}
			if found, err = d.Detect(ctx, img, o); err != nil {
				return fmt.Errorf("%s: %w", p, err)
			}
		}
		runs := max(1, c.Repeat-1)
		if c.Repeat <= 1 {
			runs = 1
		}
		logf("%s: %d objects in %.1f ms (resize %.1f, network %.1f, boxes on the host %.1f)%s", filepath.Base(p), len(found.Objects),
			ms(time.Since(start))/float64(runs), ms(found.Resize), ms(found.Network), ms(found.Host), repeated(c.Repeat))
		objs := []objectJSON{}
		for _, ob := range found.Objects {
			objs = append(objs, objectJSON{Box: [4]float32{float32(ob.Box.Min.X), float32(ob.Box.Min.Y), float32(ob.Box.Max.X), float32(ob.Box.Max.Y)},
				Class: ob.Class, Label: ob.Label, Score: float32(ob.Score)})
		}
		if err := enc.Encode(struct {
			File    string       `json:"file"`
			Width   int          `json:"width"`
			Height  int          `json:"height"`
			Objects []objectJSON `json:"objects"`
		}{p, img.Bounds().Dx(), img.Bounds().Dy(), objs}); err != nil {
			return err
		}
		if c.Draw {
			out := drawPath(c.Out, p, ".objects.png")
			var boxes []box
			for _, o := range objs {
				boxes = append(boxes, box{corners: rectCorners(o.Box), color: classColor(o.Class)})
			}
			if err := drawBoxes(out, img, boxes); err != nil {
				return err
			}
			logf("wrote %s", out)
		}
	}
	return nil
}

// objectJSON is an object as detect prints it: its box (x1, y1, x2, y2), class, label and score.
type objectJSON struct {
	Box   [4]float32 `json:"box"`
	Class int        `json:"class"`
	Label string     `json:"label"`
	Score float32    `json:"score"`
}

// orAll is a least score of a flag as ai/vision's options take it: 0 keeps everything (a
// negative score there; 0 is the default).
func orAll(score float32) float32 {
	if score == 0 {
		return -1
	}
	return score
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// repeated says how many runs a time is the mean of.
func repeated(n int) string {
	if n <= 1 {
		return ""
	}
	return fmt.Sprintf(", the mean of %d runs after the first", n-1)
}

// drawPath is where --draw writes a picture's drawing: dir (or the picture's directory) and the
// picture's name with suffix.
// outDir makes the directory -o names (nothing for none), so that a command does not fail at its
// first file after the models ran.
func outDir(dir string) error {
	if dir == "" {
		return nil
	}
	return os.MkdirAll(dir, 0o755)
}

func drawPath(dir, picture, suffix string) string {
	if dir == "" {
		dir = filepath.Dir(picture)
	}
	return filepath.Join(dir, strings.TrimSuffix(filepath.Base(picture), filepath.Ext(picture))+suffix)
}

type box struct {
	corners [4][2]float32
	color   color.Color
}

func rectCorners(b [4]float32) [4][2]float32 {
	return [4][2]float32{{b[0], b[1]}, {b[2], b[1]}, {b[2], b[3]}, {b[0], b[3]}}
}

// classColor is a color for a class: hues spread by the golden angle.
func classColor(c int) color.Color {
	h := math.Mod(float64(c)*137.508, 360) / 60
	x := 1 - math.Abs(math.Mod(h, 2)-1)
	var r, g, b float64
	switch int(h) {
	case 0:
		r, g = 1, x
	case 1:
		r, g = x, 1
	case 2:
		g, b = 1, x
	case 3:
		g, b = x, 1
	case 4:
		r, b = x, 1
	default:
		r, b = 1, x
	}
	return color.RGBA{uint8(40 + 215*r), uint8(40 + 215*g), uint8(40 + 215*b), 255}
}

// drawBoxes writes img with the boxes' outlines drawn to a PNG at path.
func drawBoxes(path string, img image.Image, boxes []box) error {
	b := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(dst, dst.Bounds(), img, b.Min, draw.Src)
	line := func(x0, y0, x1, y1 float64, c color.Color) {
		n := int(math.Max(math.Abs(x1-x0), math.Abs(y1-y0))) + 1
		for i := 0; i <= n; i++ {
			t := float64(i) / float64(n)
			x, y := int(x0+(x1-x0)*t), int(y0+(y1-y0)*t)
			for dx := -1; dx <= 1; dx++ {
				for dy := -1; dy <= 1; dy++ {
					dst.Set(x+dx, y+dy, c)
				}
			}
		}
	}
	for _, bx := range boxes {
		for k := range 4 {
			a, c := bx.corners[k], bx.corners[(k+1)%4]
			line(float64(a[0]), float64(a[1]), float64(c[0]), float64(c[1]), bx.color)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := png.Encode(f, dst); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
