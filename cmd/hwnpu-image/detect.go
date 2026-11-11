package main

import (
	"github.com/shibukawa/hwnpu/internal/milcpu"
	"github.com/shibukawa/hwnpu/internal/onnxrt"

	"context"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"maps"
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
	"github.com/shibukawa/hwnpu/internal/mem"
	"github.com/shibukawa/hwnpu/internal/vision/detect"
)

// DetectCmd is "hwnpu-image detect": the objects in pictures (COCO's 80 classes) with RTMDet
// (OpenMMLab MMDetection, Apache-2.0; internal/vision/detect), on Apple's Neural Engine or
// Intel's NPU, converted the first time, and elsewhere (AMD's NPU has no form of it yet) or with
// --device cpu on the CPU, the ONNX file with ONNX Runtime (--cpu: the converted program on the
// CPU interpreter, for checking a conversion). The network up to its head runs on the NPU;
// decoding the boxes and suppressing the overlaps on the host.
//
// For each picture it prints one line of JSON: each object's box (x1, y1, x2, y2 in the
// picture's pixels), label, class and score. With --draw it also writes NAME.objects.png.
type DetectCmd struct {
	Pictures []string `arg:"" type:"existingfile" help:"The pictures (JPEG or PNG)."`
	Model    string   `short:"m" default:"detect" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (detect, rtmdet, rtmdet:tiny)."`
	Score    float32  `default:"0.3" help:"The least score of an object to report."`
	NMS      float32  `name:"nms" default:"0.65" help:"The overlap (intersection over union) above which the lower of two objects of a class goes."`
	Class    []string `placeholder:"LABEL" help:"Only these classes (person,car,...; hwnpu-image detect --class help lists them)."`
	Draw     bool     `help:"Also write NAME.objects.png: the picture with the boxes."`
	Out      string   `short:"o" type:"path" placeholder:"DIR" help:"The directory --draw writes into (default: next to each picture)."`
	Repeat   int      `default:"1" help:"Run each picture this many times and report the mean time."`
	Dir      string   `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's --dir)."`
	Cache    string   `placeholder:"DIR" help:"Compile cache (default: hwnpu in the user cache directory)."`
	CPU      bool     `name:"cpu" help:"Run the converted program on the CPU interpreter instead of the NPU (slow; for checking a conversion)."`
	Device   string   `enum:",npu,cpu" default:"" help:"Where to run: npu or cpu (ONNX Runtime; default: models.yaml's device for this machine's NPU, or HWNPU_DEVICE)."`
	Threads  int      `default:"1" help:"The CPU's threads with --device cpu (1: the least CPU time and energy)."`
}

// Run finds the objects of each picture.
func (c *DetectCmd) Run(ctx context.Context) error {
	if slices.Contains(c.Class, "help") {
		fmt.Println(strings.Join(detect.COCO, ", "))
		return nil
	}
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	m, err := resolve(c.Model, chat.KindDetect)
	if err != nil {
		return err
	}
	file, err := onlyFile(m, ".onnx")
	if err != nil {
		return err
	}
	dir := c.Dir
	if dir == "" {
		dir = chat.ModelsDir()
	}
	setup := detect.Setup{Dir: filepath.Join(dir, m.DirName()), Repo: m.Source(), File: file, SHA256: m.SHA256[file], Log: lw}
	var d *detect.Detector
	t0 := time.Now()
	ort, builtin := cpuRoute(ctx, m, c.Device, c.CPU, c.Threads, logf)
	c.CPU = c.CPU || builtin
	if ort {
		model, err := detect.Fetch(ctx, setup)
		if err != nil {
			return err
		}
		if d, err = detect.OpenCPU(model, c.Threads); err != nil {
			return err
		}
		logf("loaded %s in %.1f s, on the CPU (ONNX Runtime, %d threads)", m.Name, time.Since(t0).Seconds(), max(1, c.Threads))
	} else {
		dev, backend, closeDev, err := openProgramDevice(ctx, c.CPU, c.Cache, "RTMDet")
		if err != nil {
			return err
		}
		defer closeDev()
		defer npuHeld(logf)()
		setup.Backend = backend
		_, program, err := detect.Prepare(ctx, setup)
		if err != nil {
			return err
		}
		t0 = time.Now()
		if d, err = detect.Open(ctx, dev, program); err != nil {
			return err
		}
		logf("loaded %s in %.1f s, %s", m.Name, time.Since(t0).Seconds(), placement(c.CPU, d.OnNPU()))
	}
	defer d.Close()
	d.Score, d.NMS = c.Score, c.NMS
	keep := map[string]bool{}
	for _, l := range c.Class {
		for _, k := range strings.Split(l, ",") {
			k = strings.TrimSpace(k)
			if !slices.Contains(d.Classes, k) {
				return fmt.Errorf("--class %q is not one of the model's classes (--class help lists them)", k)
			}
			keep[k] = true
		}
	}
	console.Running()
	enc := json.NewEncoder(os.Stdout)
	for _, p := range c.Pictures {
		img, err := readPicture(p)
		if err != nil {
			return err
		}
		var objs []detect.Object
		start := time.Now()
		for i := range max(1, c.Repeat) {
			if i == 1 {
				start = time.Now()
			}
			if objs, err = d.Detect(ctx, img); err != nil {
				return fmt.Errorf("%s: %w", p, err)
			}
		}
		runs := max(1, c.Repeat-1)
		if c.Repeat <= 1 {
			runs = 1
		}
		if len(keep) > 0 {
			objs = slices.DeleteFunc(objs, func(o detect.Object) bool { return !keep[o.Label] })
		}
		logf("%s: %d objects in %.1f ms (resize %.1f, network %.1f, boxes on the host %.1f)%s", filepath.Base(p), len(objs),
			ms(time.Since(start))/float64(runs), ms(d.Times[0]), ms(d.Times[1]), ms(d.Times[2]), repeated(c.Repeat))
		if objs == nil {
			objs = []detect.Object{}
		}
		if err := enc.Encode(struct {
			File    string          `json:"file"`
			Width   int             `json:"width"`
			Height  int             `json:"height"`
			Objects []detect.Object `json:"objects"`
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

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// repeated says how many runs a time is the mean of.
func repeated(n int) string {
	if n <= 1 {
		return ""
	}
	return fmt.Sprintf(", the mean of %d runs after the first", n-1)
}

// cpuRoute is where an ONNX model m runs (chat.ConvertedDevice) and logs it: ort, ONNX
// Runtime on the CPU when its library is here; builtin, hwnpu's own CPU executor on the
// converted program (internal/milcpu, on threads goroutines) when it is not, or with --cpu
// (interp: all the cores); neither, the NPU.
func cpuRoute(ctx context.Context, m chat.Model, flag string, interp bool, threads int, logf func(string, ...any)) (ort, builtin bool) {
	if interp {
		milcpu.Threads = 0
		return false, true
	}
	device, why, err := chat.ConvertedDevice(m, chat.HereVendor(), flag)
	if err != nil {
		logf("%v; the NPU", err)
		return false, false
	}
	if device != "cpu" {
		logf("%s on the %s (%s)", m.Name, device, why)
		return false, false
	}
	if onnxrt.Available() {
		logf("%s on the CPU with ONNX Runtime (%s)", m.Name, why)
		return true, false
	}
	milcpu.Threads = max(1, threads)
	logf("%s on the CPU with hwnpu's own executor (%s; ONNX Runtime, faster, is not here: HWNPU_ONNXRUNTIME)", m.Name, why)
	return false, true
}

// placement is where a loaded program runs, for the log.
func placement(cpu, onNPU bool) string {
	switch {
	case cpu:
		return "on the CPU interpreter"
	case onNPU:
		return "all on the NPU"
	}
	return "NOT all on the NPU"
}

// npuHeld logs, when called after the pictures, how much more memory the Neural Engine holds
// than before (mem.NPUWired: the programs' weights and buffers, outside the process's
// footprint).
func npuHeld(logf func(string, ...any)) func() {
	before, ok := mem.NPUWired()
	return func() {
		if after, ok2 := mem.NPUWired(); ok && ok2 && after > before {
			logf("the Neural Engine holds %.0f MB more", float64(after-before)/1e6)
		}
	}
}

// onlyFile is the one file with the extension that models.yaml's sha256 of m names: the file of
// a repository with several models (bukuroo/RTMDet-ONNX).
func onlyFile(m chat.Model, ext string) (string, error) {
	var files []string
	for f := range maps.Keys(m.SHA256) {
		if strings.EqualFold(filepath.Ext(f), ext) && !strings.Contains(f, "/") {
			files = append(files, f)
		}
	}
	if len(files) != 1 {
		return "", fmt.Errorf("%s: models.yaml's sha256 must name its one %s file (it names %d)", m.Name, ext, len(files))
	}
	return files[0], nil
}

// openProgramDevice opens the NPU for converted programs (Core ML or OpenVINO IR), or none for
// the CPU interpreter. what names the model for the error on an NPU that runs neither.
func openProgramDevice(ctx context.Context, cpu bool, cache, what string) (*hwnpu.Device, string, func(), error) {
	if cpu {
		return nil, llm.BackendCoreML, func() {}, nil
	}
	if cache == "" {
		if d, err := os.UserCacheDir(); err == nil {
			cache = filepath.Join(d, "hwnpu")
		}
	}
	dev, device, err := llm.OpenDevice(ctx, "", cache)
	if err != nil {
		return nil, "", nil, err
	}
	backend := llm.ProgramBackend(device)
	if backend == "" {
		dev.Close()
		return nil, "", nil, fmt.Errorf("%s runs neither Core ML packages nor OpenVINO IR (--device cpu runs %s on the CPU)", device, what)
	}
	return dev, backend, func() { dev.Close() }, nil
}

// drawPath is where --draw writes a picture's drawing: dir (or the picture's directory) and the
// picture's name with suffix.
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
