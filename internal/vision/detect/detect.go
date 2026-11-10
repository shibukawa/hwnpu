// Package detect finds objects in a picture with RTMDet (OpenMMLab MMDetection's real-time
// detector, Apache-2.0): each object's box, class (COCO's 80) and score, from the ONNX export of
// MMDeploy (bukuroo/RTMDet-ONNX: rtmdet-tiny-coco.onnx, rtmdet-s-coco.onnx, ...).
//
// The network runs on the NPU up to its head's convolutions: for each of the three strides (8,
// 16, 32) the class logits and the box's four distances (left, top, right, bottom, in pixels)
// at every cell, channels first, as one output (internal/onnx's Flat). The graph's tail lays them
// out anchor by anchor, takes the sigmoid, decodes every box and keeps the top 5000 before a
// class-wise non-maximum suppression; the host does this for the cells whose logit passes the
// least score only (the sigmoid is monotonic), which is the same for the objects kept and
// leaves the Neural Engine no transposes.
//
// The picture is scaled to fit 640 x 640 (its aspect kept), placed at the top left and the rest
// filled with gray 114, as MMDetection's test pipeline pads it.
package detect

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
	"slices"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/onnx"
)

// Size is the picture's size the model takes.
const Size = 640

// Arch names the program in its metadata.
const Arch = "rtmdet"

// Revision counts the changes to the program that call for converting it again.
const Revision = 1 // with onnx.Program.ExpSigmoid

// Strides are the head's strides, in the order of its outputs.
var Strides = []int{8, 16, 32}

// COCO are the names of COCO's 80 classes, in the model's order.
var COCO = []string{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
	"cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
	"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
	"orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
	"dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}

// Mean and Std normalize the RGB pixels (0 to 255), as MMDetection's data preprocessor does.
var (
	Mean = [3]float32{123.675, 116.28, 103.53}
	Std  = [3]float32{58.395, 57.12, 57.375}
)

// Setup says what Prepare downloads and converts.
type Setup struct {
	Dir, Repo string
	// File is the ONNX file in the repository (rtmdet-s-coco.onnx).
	File string
	// SHA256 is the file's, when the definitions pin it.
	SHA256  string
	Backend string
	Log     io.Writer
}

// Prepare downloads the ONNX file into s.Dir unless it is there and converts the program
// unless it is. It returns the model's and the program's paths.
func Prepare(ctx context.Context, s Setup) (model, program string, err error) {
	log := cmp.Or[io.Writer](s.Log, io.Discard)
	model = filepath.Join(s.Dir, filepath.FromSlash(s.File))
	if err := os.MkdirAll(filepath.Dir(model), 0o755); err != nil {
		return "", "", err
	}
	if err := chat.Fetch(ctx, s.Repo, s.File, model, log); err != nil {
		return "", "", err
	}
	if s.SHA256 != "" {
		if err := CheckSHA(model, s.SHA256); err != nil {
			return "", "", err
		}
	}
	ext := ".mlpackage"
	if s.Backend == llm.BackendIntel {
		ext = ".xml"
	}
	stem := filepath.Base(s.File)
	stem = stem[:len(stem)-len(filepath.Ext(stem))]
	program = filepath.Join(s.Dir, fmt.Sprintf("%s-%s-r%d-fp16-%d%s", Arch, stem, Revision, Size, ext))
	if _, err := os.Stat(program + ".done"); err == nil {
		return model, program, nil
	}
	g, err := onnx.Load(model)
	if err != nil {
		return "", "", err
	}
	p, err := NewProgram(g)
	if err != nil {
		return "", "", err
	}
	bar := console.NewBar(log, "converting "+filepath.Base(program))
	os.RemoveAll(program)
	_, err = llm.WritePart(program, p, nil, llm.Options{Format: "fp16", Backend: s.Backend}, func(pkg *mlpackage.Package) {
		pkg.Description = fmt.Sprintf("RTMDet from %s up to its head's convolutions, %dx%d, fp16 (hwnpu)", s.Repo, Size, Size)
		pkg.Metadata[llm.MetaArch] = Arch
	})
	if err != nil {
		bar.Done("failed")
		return "", "", err
	}
	bar.Done(console.Duration(bar.Elapsed()))
	return model, program, os.WriteFile(program+".done", nil, 0o644)
}

// CheckSHA checks a file's sha256.
func CheckSHA(path, want string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("%s has sha256 %s, not %s", path, got, want)
	}
	return nil
}

// Heads are the graph's head outputs, (1, C, H, W) each, per stride: the class logits and the
// box distances (already times the stride). MMDeploy's export lays each out anchor by anchor
// (Transpose to NHWC, Reshape to (1, -1, C)) and concatenates the strides: the inputs of the two
// such Concats are the heads, the one of 4 columns the boxes.
func Heads(g *onnx.Graph) (cls, box []string, err error) {
	prod := map[string]*onnx.Node{}
	for _, n := range g.Nodes {
		for _, o := range n.Out {
			prod[o] = n
		}
	}
	for _, n := range g.Nodes {
		if n.Op != "Concat" || len(n.In) != len(Strides) {
			continue
		}
		var heads []string
		cols := -1
		for _, in := range n.In {
			r := prod[in]
			if r == nil || r.Op != "Reshape" || g.Init[r.In[1]] == nil {
				break
			}
			t := prod[r.In[0]]
			if t == nil || t.Op != "Transpose" || !slices.Equal(t.IntList("perm"), []int{0, 2, 3, 1}) {
				break
			}
			shape := g.Init[r.In[1]].Ints()
			cols = shape[len(shape)-1]
			heads = append(heads, t.In[0])
		}
		if len(heads) != len(Strides) {
			continue
		}
		if cols == 4 {
			box = heads
		} else {
			cls = heads
		}
	}
	if cls == nil || box == nil {
		return nil, nil, fmt.Errorf("detect: no RTMDet head (the class and box outputs laid out per anchor) in the graph")
	}
	return cls, box, nil
}

// NewProgram is the program of the graph from the picture up to its heads: for each stride the
// class logits, then the boxes.
func NewProgram(g *onnx.Graph) (*onnx.Program, error) {
	if len(g.Inputs) != 1 {
		return nil, fmt.Errorf("detect: the graph has %d inputs, not 1", len(g.Inputs))
	}
	cls, box, err := Heads(g)
	if err != nil {
		return nil, err
	}
	var outs []string
	for i := range Strides {
		outs = append(outs, cls[i], box[i])
	}
	return &onnx.Program{G: g, Inputs: []onnx.ValueInfo{{Name: g.Inputs[0].Name, Shape: []int{1, 3, Size, Size}}}, Outputs: outs, Flat: true, ExpSigmoid: true}, nil
}

// Object is an object found: its box (x1, y1, x2, y2 in the picture's pixels), class and score.
type Object struct {
	Box   [4]float32 `json:"box"`
	Class int        `json:"class"`
	Label string     `json:"label"`
	Score float32    `json:"score"`
}

// Detector runs RTMDet.
type Detector struct {
	// Score is the least score an object is kept with, NMS the overlap (intersection over union)
	// above which the lower of two objects of a class goes (MMDetection's RTMDet: 0.65), Max the
	// most objects kept (300).
	Score, NMS float32
	Max        int
	// Classes are the class names (COCO).
	Classes []string
	// Times are the last picture's: resizing, the program, decoding and suppressing.
	Times [3]time.Duration
	prog  runner.Program
}

// cells is the head's cells over every stride.
func cells() int {
	n := 0
	for _, s := range Strides {
		n += (Size / s) * (Size / s)
	}
	return n
}

// HeadShapes are the program's outputs, in order (NewProgram's): for each stride the class
// logits (1, classes, H, W) and the boxes (1, 4, H, W).
func HeadShapes(classes int) [][]int {
	var out [][]int
	for _, s := range Strides {
		out = append(out, []int{1, classes, Size / s, Size / s}, []int{1, 4, Size / s, Size / s})
	}
	return out
}

// Open loads the program on dev (nil: the CPU interpreter). Its classes are what its output
// holds besides the boxes: (classes + 4) values a cell.
func Open(ctx context.Context, dev *hwnpu.Device, program string) (*Detector, error) {
	prog, err := runner.Open(ctx, dev, program)
	if err != nil {
		return nil, err
	}
	y := prog.Output()
	total := onnx.Len(y)
	if len(y) != 3 || total%cells() != 0 || total/cells() <= 4 {
		prog.Close()
		return nil, fmt.Errorf("detect: %s gives %v, not RTMDet's heads", program, y)
	}
	classes := COCO
	if c := total/cells() - 4; c != len(COCO) {
		classes = make([]string, c)
		for i := range classes {
			classes[i] = fmt.Sprint(i)
		}
	}
	return &Detector{Score: 0.3, NMS: 0.65, Max: 300, Classes: classes, prog: prog}, nil
}

// OnNPU reports whether the program runs all on the NPU.
func (d *Detector) OnNPU() bool { return d.prog.OnNPU() }

func (d *Detector) Close() error { return d.prog.Close() }

// Pixels are a picture as the model reads it: scaled by scale to fit Size x Size (PIL's
// bilinear filter), at the top left, gray 114 past it, normalized by Mean and Std, channels
// first.
func Pixels(img image.Image) (x []float32, scale float64) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	scale = min(float64(Size)/float64(w), float64(Size)/float64(h))
	nw, nh := max(1, min(Size, int(float64(w)*scale+0.5))), max(1, min(Size, int(float64(h)*scale+0.5)))
	rgb := lfm3vl.ResizeRGB(lfm3vl.RGB(img), w, h, nw, nh, "bilinear")
	x = make([]float32, 3*Size*Size)
	var pad [3]float32
	for c := range 3 {
		pad[c] = (114 - Mean[c]) / Std[c]
	}
	for c := range 3 {
		plane := x[c*Size*Size : (c+1)*Size*Size]
		inv := 1 / Std[c]
		for y := range Size {
			row := plane[y*Size : (y+1)*Size]
			if y >= nh {
				for i := range row {
					row[i] = pad[c]
				}
				continue
			}
			src := rgb[y*nw*3:]
			for i := range nw {
				row[i] = (float32(src[3*i+c]) - Mean[c]) * inv
			}
			for i := nw; i < Size; i++ {
				row[i] = pad[c]
			}
		}
	}
	return x, scale
}

// Detect finds the objects in a picture.
func (d *Detector) Detect(ctx context.Context, img image.Image) ([]Object, error) {
	t0 := time.Now()
	x, scale := Pixels(img)
	t1 := time.Now()
	y, err := d.prog.Run(ctx, map[string][]float32{"x": x})
	if err != nil {
		return nil, err
	}
	t2 := time.Now()
	objs := d.Decode(onnx.SplitFlat(y, HeadShapes(len(d.Classes))))
	b := img.Bounds()
	for i := range objs {
		for j := range 4 {
			v := float64(objs[i].Box[j]) / scale
			lim := float64(b.Dx())
			if j%2 == 1 {
				lim = float64(b.Dy())
			}
			objs[i].Box[j] = float32(math.Max(0, math.Min(lim, v)))
		}
	}
	d.Times = [3]time.Duration{t1.Sub(t0), t2.Sub(t1), time.Since(t2)}
	return objs, nil
}

// Decode turns the heads' outputs (HeadShapes', SplitFlat's) into the objects of at least the
// least score in the model's pixels, suppressed.
func (d *Detector) Decode(heads [][]float32) []Object {
	// sigmoid(v) >= Score where v >= logit(Score)
	least := float32(math.Log(float64(d.Score) / (1 - float64(d.Score))))
	var cands []Object
	for i, s := range Strides {
		cls, box := heads[2*i], heads[2*i+1]
		HW := (Size / s) * (Size / s)
		W := Size / s
		for c := range len(cls) / HW {
			for cell, v := range cls[c*HW : (c+1)*HW] {
				if v < least {
					continue
				}
				px, py := float32(cell%W*s), float32(cell/W*s)
				cands = append(cands, Object{
					Box:   [4]float32{px - box[cell], py - box[HW+cell], px + box[2*HW+cell], py + box[3*HW+cell]},
					Class: c, Label: d.Classes[c], Score: float32(1 / (1 + math.Exp(-float64(v)))),
				})
			}
		}
	}
	return Suppress(cands, d.NMS, d.Max)
}

// Suppress keeps, class by class and from the highest score, the objects that overlap no kept
// one of their class by more than iou, then the max of the highest scores (all when max is 0).
func Suppress(objs []Object, iou float32, max int) []Object {
	slices.SortStableFunc(objs, func(a, b Object) int { return cmp.Compare(b.Score, a.Score) })
	var kept []Object
	for _, o := range objs {
		ok := true
		for _, k := range kept {
			if k.Class == o.Class && IoU(k.Box, o.Box) > iou {
				ok = false
				break
			}
		}
		if ok {
			kept = append(kept, o)
			if max > 0 && len(kept) == max {
				break
			}
		}
	}
	return kept
}

// IoU is the intersection over union of two boxes (x1, y1, x2, y2).
func IoU(a, b [4]float32) float32 {
	iw := min(a[2], b[2]) - max(a[0], b[0])
	ih := min(a[3], b[3]) - max(a[1], b[1])
	if iw <= 0 || ih <= 0 {
		return 0
	}
	in := iw * ih
	u := (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - in
	if u <= 0 {
		return 0
	}
	return in / u
}
