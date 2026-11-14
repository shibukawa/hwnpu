// Package pose finds people and their 17 COCO keypoints in an image with RTMO (OpenMMLab
// MMPose's one-stage pose estimator, Apache-2.0), from the ONNX export that Transformers.js
// reads (Xenova/RTMO-s, -m, -l).
//
// The graph runs in two pieces (internal/onnx): from the image to the head's outputs for every
// anchor (the scores, the boxes, the pose features and the keypoints' visibility) is a program
// for the NPU; the rest (top-k, non-maximum suppression, and the dynamic coordinate classifier
// that turns the pose features of each person kept into keypoints) depends on how many people
// there are, and runs on the host, evaluated from the graph itself.
package pose

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
	"github.com/shibukawa/hwnpu/internal/onnxrt"
	"github.com/shibukawa/hwnpu/internal/vision/ironcnn"
)

// Size is the image's size the model takes: images are stretched to it.
const Size = 640

// ModelFile is the ONNX export in its repository.
const ModelFile = "onnx/model.onnx"

// Arch names the program in its metadata.
const Arch = "rtmo"

// Revision counts the changes to the program that call for converting it again.
const Revision = 2 // 2: pools of 5 for the larger windows, y in rows (onnx.Program.Flat)

// Names of the COCO keypoints, in the model's order.
var Names = []string{"nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
	"left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee",
	"left_ankle", "right_ankle"}

// dynamic are the ops that split the graph: what reads their outputs depends on how many people
// there are, and runs on the host (onnx.Graph.Cut).
var dynamic = map[string]bool{"TopK": true, "NonMaxSuppression": true}

// outputs are the graph's outputs: the people's boxes and scores, and their keypoints.
var outputs = []string{"dets", "keypoints"}

// Setup says what Prepare downloads and converts.
type Setup struct {
	Dir, Repo string
	// SHA256 is the model file's, when the definitions pin it.
	SHA256  string
	Backend string
	Log     io.Writer
}

// Prepare downloads the ONNX export into s.Dir unless it is there and converts the program
// unless it is. It returns the model's and the program's paths.
func Prepare(ctx context.Context, s Setup) (model, program string, err error) {
	log := cmp.Or[io.Writer](s.Log, io.Discard)
	if model, err = Fetch(ctx, s); err != nil {
		return "", "", err
	}
	ext := llm.ProgramExt(s.Backend) // .aimodel, .mlpackage or .xml
	program = filepath.Join(s.Dir, fmt.Sprintf("%s-r%d-fp16-%d%s", Arch, Revision, Size, ext))
	done := program + ".done"
	if _, err := os.Stat(done); err == nil {
		return model, program, nil
	}
	g, err := onnx.Load(model)
	if err != nil {
		return "", "", err
	}
	bar := console.NewBar(log, "converting "+filepath.Base(program))
	os.RemoveAll(program)
	p := newProgram(g)
	_, err = llm.WritePart(program, p, nil, llm.Options{Format: "fp16", Backend: s.Backend}, func(pkg *mlpackage.Package) {
		pkg.Description = fmt.Sprintf("RTMO from %s up to its head's outputs, %dx%d, fp16 (hwnpu)", s.Repo, Size, Size)
		pkg.Metadata[llm.MetaArch] = Arch
	})
	if err != nil {
		bar.Done("failed")
		return "", "", err
	}
	bar.Done(console.Duration(bar.Elapsed()))
	return model, program, os.WriteFile(done, nil, 0o644)
}

// newProgram is the program of the graph up to its dynamic part: for every anchor, the person
// score, the decoded box, the pose feature and the keypoints' visibility, as the graph lays them
// out for its top-k, flattened and concatenated into y.
func newProgram(g *onnx.Graph) *onnx.Program {
	if len(g.Inputs) != 1 {
		return &onnx.Program{G: g}
	}
	in := g.Inputs[0].Name
	return &onnx.Program{G: g, Inputs: []onnx.ValueInfo{{Name: in, Shape: []int{1, 3, Size, Size}}}, Outputs: g.Cut(in, dynamic, outputs), Flat: true}
}

// Fetch downloads the ONNX export into s.Dir unless it is there, checks its sha256 and returns
// its path: all the CPU (OpenCPU) needs.
func Fetch(ctx context.Context, s Setup) (string, error) {
	model := filepath.Join(s.Dir, filepath.FromSlash(ModelFile))
	if err := os.MkdirAll(filepath.Dir(model), 0o755); err != nil {
		return "", err
	}
	if err := chat.Fetch(ctx, s.Repo, ModelFile, model, cmp.Or[io.Writer](s.Log, io.Discard)); err != nil {
		return "", err
	}
	if s.SHA256 != "" {
		if err := checkSHA(model, s.SHA256); err != nil {
			return "", err
		}
	}
	return model, nil
}

func checkSHA(path, want string) error {
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
		return fmt.Errorf("pose: %s has sha256 %s, not %s", path, got, want)
	}
	return nil
}

// Estimator runs RTMO.
type Estimator struct {
	// Times are the last image's resize, program (with the outputs taken apart) and host part.
	Times [3]time.Duration
	g     *onnx.Graph
	p     *onnx.Program
	prog  runner.Program
	npu   *ironcnn.Graph // an export of tools/amd/iron_onnxcnn.py on AMD's NPU, in place of prog
}

// ExportVersion is the version of ExportNPU's exports (a key of the shared layers: chat.PinnedRecipe),
// bumped by hand when what it passes iron_onnxcnn.py (npuOptions, the outputs) changes.
const ExportVersion = 1

// npuOptions are iron_onnxcnn.py's options for RTMO: its pixels as they are (0 to 255), SiLU as
// an operator of its own (the GEMM epilogue's vtanh is a few ulps off near 0) and bf16's
// products (HIGH) in place of bfp16's: at MEDIUM the keypoints' scores moved by up to 0.13 from
// ONNX Runtime's and a doubtful person's box by 97 px, at HIGH by 0.03 and 6 px, the network
// 0.15 s -> 0.22 s (docs/amd-npu-models.md 22)
var npuOptions = []string{"--mean", "0", "0", "0", "--std", "1", "1", "1", "--silu", "op", "--precision", "HIGH", "--dw-precision", "HIGH"}

// ExportNPU makes the export of tools/amd/iron_onnxcnn.py of the ONNX file model in out with
// IRON's python (chat.IRONPython): the network from the image to the head's convolutions
// (ironcnn.LastConvs), its transformer layer included.
func ExportNPU(ctx context.Context, python []string, name, model, out string, log io.Writer) error {
	g, err := onnx.Load(model)
	if err != nil {
		return err
	}
	return ironcnn.Make(ctx, python, name, model, out, Size, ironcnn.LastConvs(g), npuOptions, log)
}

// OpenNPU loads an export of ExportNPU (dir) onto dev, and the ONNX file model for the host's
// part: from the head's convolutions to the people.
func OpenNPU(ctx context.Context, dev *hwnpu.Device, dir, model string) (*Estimator, error) {
	g, err := onnx.Load(model)
	if err != nil {
		return nil, err
	}
	x, err := ironcnn.Read(dir)
	if err != nil {
		return nil, err
	}
	if x.Width != Size || x.Height != Size {
		return nil, fmt.Errorf("pose: %s is for %dx%d, not %dx%d", dir, x.Width, x.Height, Size, Size)
	}
	want := ironcnn.LastConvs(g)
	if len(want) != len(x.Outputs) {
		return nil, fmt.Errorf("pose: %s gives %d values, not the %d of RTMO's head", dir, len(x.Outputs), len(want))
	}
	for _, o := range x.Outputs {
		if !slices.Contains(want, o.Name) {
			return nil, fmt.Errorf("pose: %s gives %s, not one of RTMO's head %v", dir, o.Name, want)
		}
	}
	npu, err := ironcnn.Open(ctx, dev, dir)
	if err != nil {
		return nil, err
	}
	return &Estimator{g: g, npu: npu}, nil
}

// Open loads the graph and the program on dev (nil: the CPU interpreter).
func Open(ctx context.Context, dev *hwnpu.Device, model, program string) (*Estimator, error) {
	g, err := onnx.Load(model)
	if err != nil {
		return nil, err
	}
	// build the program once more without writing it: the values it folds on the host and its
	// shapes are what the host's part of the graph reads besides the program's output
	p := newProgram(g)
	b := mlpackage.NewBuilder("CoreML8", nil)
	b.ImmediateLimit = math.MaxInt // no weight file: the constants are dropped with the builder
	if _, _, err := p.Build(b, nil, llm.Options{Format: "fp16"}); err != nil {
		return nil, err
	}
	prog, err := runner.Open(ctx, dev, program)
	if err != nil {
		return nil, err
	}
	return newEstimator(g, p, prog, program)
}

// OpenCPU loads the ONNX file (Prepare's model) into ONNX Runtime on threads threads of the CPU
// (0 or 1: the calling thread), the graph up to the program's outputs; the host's part is the
// same.
func OpenCPU(model string, threads int) (*Estimator, error) {
	g, err := onnx.Load(model)
	if err != nil {
		return nil, err
	}
	p := newProgram(g)
	b := mlpackage.NewBuilder("CoreML8", nil)
	b.ImmediateLimit = math.MaxInt
	if _, _, err := p.Build(b, nil, llm.Options{Format: "fp16"}); err != nil {
		return nil, err
	}
	raw, err := onnxrt.ReadModel(model)
	if err != nil {
		return nil, err
	}
	prog, err := onnxrt.OpenProgram(raw, p, threads)
	if err != nil {
		return nil, err
	}
	return newEstimator(g, p, prog, model)
}

func newEstimator(g *onnx.Graph, p *onnx.Program, prog runner.Program, program string) (*Estimator, error) {
	y := prog.Output()
	total := 0
	for _, w := range p.Widths() {
		total += w
	}
	if len(y) != 3 || y[2] != total {
		prog.Close()
		return nil, fmt.Errorf("pose: %s gives %v, not the %d values of RTMO's head", program, y, total)
	}
	return &Estimator{g: g, p: p, prog: prog}, nil
}

// OnNPU reports whether the program runs all on the NPU.
func (e *Estimator) OnNPU() bool { return e.npu != nil || e.prog.OnNPU() }

func (e *Estimator) Close() error {
	if e.npu != nil {
		return e.npu.Close()
	}
	return e.prog.Close()
}

// Keypoint is a keypoint of a person, in the image's pixels, and its score (visible from about 0.3).
type Keypoint struct {
	Name  string  `json:"name"`
	X     float64 `json:"x"`
	Y     float64 `json:"y"`
	Score float64 `json:"score"`
}

// Person is a person found: the box (x1, y1, x2, y2 in the image's pixels), its score and the
// keypoints.
type Person struct {
	Box       [4]float64 `json:"box"`
	Score     float64    `json:"score"`
	Keypoints []Keypoint `json:"keypoints"`
}

// Pixels are an image stretched to Size x Size as the model reads it: RGB, 0 to 255, channels
// first (PIL's bilinear resize, as Transformers.js's processor does).
func Pixels(img image.Image) []float32 {
	rgb := lfm3vl.Resize(img, Size, Size, "bilinear")
	out := make([]float32, 3*Size*Size)
	for i := range Size * Size {
		for c := range 3 {
			out[c*Size*Size+i] = float32(rgb[3*i+c])
		}
	}
	return out
}

// network runs the network on x: the values the host's part of the graph is given, and the
// static shapes it may read.
func (e *Estimator) network(ctx context.Context, x []float32) (map[string]*onnx.Tensor, map[string][]int, error) {
	if e.npu != nil {
		planes, err := e.npu.Run(ctx, x)
		if err != nil {
			return nil, nil, err
		}
		given := map[string]*onnx.Tensor{}
		for j, o := range e.npu.X.Outputs {
			t := &onnx.Tensor{Shape: []int{1, o.C, o.H, o.W}, F: make([]float64, len(planes[j])), Type: onnx.Float}
			for i, v := range planes[j] {
				t.F[i] = float64(v)
			}
			given[o.Name] = t
		}
		return given, nil, nil
	}
	y, err := e.prog.Run(ctx, map[string][]float32{"x": x})
	if err != nil {
		return nil, nil, err
	}
	given := e.p.Split(y)
	for k, v := range e.p.Folded() {
		given[k] = v
	}
	return given, e.p.Shapes(), nil
}

// Estimate finds the people in an image, with a score of at least minScore.
func (e *Estimator) Estimate(ctx context.Context, img image.Image, minScore float64) ([]Person, error) {
	b := img.Bounds()
	t0 := time.Now()
	x := Pixels(img)
	t1 := time.Now()
	given, shapes, err := e.network(ctx, x)
	if err != nil {
		return nil, err
	}
	t2 := time.Now()
	out, err := e.g.EvalShapes(given, shapes, outputs)
	if err != nil {
		return nil, err
	}
	e.Times = [3]time.Duration{t1.Sub(t0), t2.Sub(t1), time.Since(t2)}
	dets, kpts := out["dets"], out["keypoints"]
	sx, sy := float64(b.Dx())/Size, float64(b.Dy())/Size
	var people []Person
	n, k := dets.Shape[1], kpts.Shape[2]
	for i := range n {
		d := dets.F[i*5:]
		if d[4] < minScore {
			continue
		}
		p := Person{Box: [4]float64{d[0] * sx, d[1] * sy, d[2] * sx, d[3] * sy}, Score: d[4]}
		for j := range k {
			q := kpts.F[(i*k+j)*3:]
			name := fmt.Sprint(j)
			if j < len(Names) {
				name = Names[j]
			}
			p.Keypoints = append(p.Keypoints, Keypoint{Name: name, X: q[0] * sx, Y: q[1] * sy, Score: q[2]})
		}
		people = append(people, p)
	}
	return people, nil
}
