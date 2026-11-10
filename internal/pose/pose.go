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
	model = filepath.Join(s.Dir, filepath.FromSlash(ModelFile))
	if err := os.MkdirAll(filepath.Dir(model), 0o755); err != nil {
		return "", "", err
	}
	if err := chat.Fetch(ctx, s.Repo, ModelFile, model, log); err != nil {
		return "", "", err
	}
	if s.SHA256 != "" {
		if err := checkSHA(model, s.SHA256); err != nil {
			return "", "", err
		}
	}
	ext := ".mlpackage"
	if s.Backend == llm.BackendIntel {
		ext = ".xml"
	}
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
func (e *Estimator) OnNPU() bool { return e.prog.OnNPU() }

func (e *Estimator) Close() error { return e.prog.Close() }

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

// Estimate finds the people in an image, with a score of at least minScore.
func (e *Estimator) Estimate(ctx context.Context, img image.Image, minScore float64) ([]Person, error) {
	b := img.Bounds()
	t0 := time.Now()
	x := Pixels(img)
	t1 := time.Now()
	y, err := e.prog.Run(ctx, map[string][]float32{"x": x})
	if err != nil {
		return nil, err
	}
	given := e.p.Split(y)
	for k, v := range e.p.Folded() {
		given[k] = v
	}
	t2 := time.Now()
	out, err := e.g.EvalShapes(given, e.p.Shapes(), outputs)
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
