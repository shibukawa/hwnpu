package face

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	llmrunner "github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/onnx"
)

// ProgramRevision counts the changes to the programs OpenProgram writes that call for
// converting them again.
const ProgramRevision = 1

// OpenProgram runs the ONNX file (face_detection_yunet_2023mar.onnx) as programs of backend
// (llm.Options.Backend: Core ML packages for Apple's Neural Engine, OpenVINO IR for Intel's NPU)
// on dev (nil: the CPU interpreter, Core ML's): the whole graph, its 12 outputs (each stride's
// cls, obj, bbox and kps) in one program per size, converted into dir when a picture first
// takes the size. The sizes are CPUSizes'.
func OpenProgram(ctx context.Context, dev *hwnpu.Device, backend, path, dir string) (*Detector, error) {
	g, err := onnx.Load(path)
	if err != nil {
		return nil, err
	}
	if len(g.Inputs) != 1 {
		return nil, fmt.Errorf("face: %s has %d inputs", path, len(g.Inputs))
	}
	x := Export{Name: "YuNet", Kind: "face-detect", Sizes: CPUSizes, Strides: []int{8, 16, 32}}
	if backend == "" {
		backend = llm.BackendCoreML
	}
	return newDetector(x, "npu", &programRunner{g: g, dev: dev, backend: backend, dir: dir, strides: x.Strides}), nil
}

// programRunner runs a size's program, loaded for the first picture that takes it.
type programRunner struct {
	g       *onnx.Graph
	dev     *hwnpu.Device
	backend string
	dir     string
	strides []int
	w, h    int
	p       *onnx.Program
	prog    llmrunner.Program
	// x is the input of the last picture, kept for the next (a picture of 320 x 256 is 1 MB of
	// floats: on hwnpu's CPU executor, made anew each time, the garbage collector ran every
	// few pictures); xw and xh its picture's size, outside which it is zeros
	x      []float32
	xw, xh int
}

func (r *programRunner) outputs() []string {
	var out []string
	for _, st := range r.strides {
		for _, k := range []string{"cls", "obj", "bbox", "kps"} {
			out = append(out, fmt.Sprintf("%s_%d", k, st))
		}
	}
	return out
}

func (r *programRunner) program(W, H int) *onnx.Program {
	return &onnx.Program{G: r.g, Inputs: []onnx.ValueInfo{{Name: r.g.Inputs[0].Name, Shape: []int{1, 3, H, W}}}, Outputs: r.outputs(), Flat: true}
}

// ConvertPrograms converts the programs of OpenProgram for each of sizes into dir ahead (or
// takes them from the shared layers), as the first picture of each size would: a detector then
// needs no network nor conversion for them.
func ConvertPrograms(ctx context.Context, backend, path, dir string, sizes [][2]int) error {
	g, err := onnx.Load(path)
	if err != nil {
		return err
	}
	if backend == "" {
		backend = llm.BackendCoreML
	}
	r := &programRunner{g: g, backend: backend, dir: dir, strides: []int{8, 16, 32}}
	for _, sz := range sizes {
		if _, err := r.convert(ctx, sz[0], sz[1]); err != nil {
			return err
		}
	}
	return nil
}

// Warm loads the program of each size once (OpenProgram's detector), converting what is
// missing: the first load of each compiles it, so that a picture of any size does not wait.
func (d *Detector) Warm(ctx context.Context) error {
	r, ok := d.run.(*programRunner)
	if !ok {
		return nil
	}
	for _, sz := range d.Sizes {
		if err := r.use(ctx, sz[0], sz[1]); err != nil {
			return err
		}
	}
	return nil
}

// convert writes the program of W x H into r.dir when it is missing (or takes it from the
// shared layers) and returns its path.
func (r *programRunner) convert(ctx context.Context, W, H int) (string, error) {
	// Core AI: NAME.coreai/model.aimodel; Core ML and Intel: NAME.mlpackage, NAME.xml
	format, err := chat.ConvertFormat(ctx, filepath.Join(r.dir, "yunet"), r.backend, "fp16")
	if err != nil {
		return "", err
	}
	name := fmt.Sprintf("yunet-r%d-%s-%dx%d", ProgramRevision, format, W, H)
	path, done, export := llm.ProgramPath(r.dir, name, r.backend)
	if llm.ProgramReady(ctx, r.dir, name, done, export) {
		return path, nil
	}
	if err := os.MkdirAll(r.dir, 0o755); err != nil {
		return "", err
	}
	if err := llm.StartProgram(ctx, path, export); err != nil {
		return "", err
	}
	_, err = llm.WritePart(path, r.program(W, H), nil, llm.Options{Format: format, Backend: r.backend}, func(pkg *mlpackage.Package) {
		pkg.Description = fmt.Sprintf("YuNet (face_detection_yunet_2023mar), %dx%d, %s (hwnpu)", W, H, format)
	})
	if err != nil {
		return "", fmt.Errorf("face: %w", err)
	}
	return path, llm.FinishProgram(ctx, r.dir, name, done, export)
}

// use loads the program of W x H, converting it first when it is missing.
func (r *programRunner) use(ctx context.Context, W, H int) error {
	if r.prog != nil && r.w == W && r.h == H {
		return nil
	}
	defer console.Loading(fmt.Sprintf("YuNet %dx%d in %s", W, H, r.dir))()
	if err := r.close(); err != nil {
		return err
	}
	p := r.program(W, H)
	path, err := r.convert(ctx, W, H)
	if err != nil {
		return err
	}
	{
		// built again without writing it: Split needs the outputs' widths
		b := mlpackage.NewBuilder("CoreML8", nil)
		b.ImmediateLimit = math.MaxInt
		if _, _, err := p.Build(b, nil, llm.Options{Format: "fp16"}); err != nil {
			return fmt.Errorf("face: %w", err)
		}
	}
	prog, err := llmrunner.Open(ctx, r.dev, path)
	if err != nil {
		return fmt.Errorf("face: %w", err)
	}
	// OpenVINO's CPU plugin (the Intel path checked without Intel's NPU) is not the NPU
	if r.dev != nil && !prog.OnNPU() && r.dev.Info().Backend != "openvino" {
		prog.Close()
		return fmt.Errorf("face: %s does not run on the NPU alone", filepath.Base(path))
	}
	r.p, r.prog, r.w, r.h = p, prog, W, H
	return nil
}

func (r *programRunner) close() error {
	if r.prog == nil {
		return nil
	}
	err := r.prog.Close()
	r.prog = nil
	return err
}

func (r *programRunner) run(ctx context.Context, px []uint8, pw, ph, W, H int) ([]cells, error) {
	if err := r.use(ctx, W, H); err != nil {
		return nil, err
	}
	if len(r.x) != 3*W*H || r.xw != pw || r.xh != ph {
		r.x = make([]float32, 3*W*H) // B, G, R planes in 0..255, as cpuRunner's
		r.xw, r.xh = pw, ph
	}
	x := r.x
	for y := 0; y < ph; y++ {
		for i := 0; i < pw; i++ {
			p := (y*pw + i) * 3
			o := y*W + i
			x[o], x[W*H+o], x[2*W*H+o] = float32(px[p+2]), float32(px[p+1]), float32(px[p])
		}
	}
	y, err := r.prog.Run(ctx, map[string][]float32{"x": x})
	if err != nil {
		return nil, fmt.Errorf("face: %w", err)
	}
	// the outputs in the order of r.outputs(), float32 as the cells take them
	parts := onnx.SplitFlat(y, r.p.OutShapes())
	if len(parts) != 4*len(r.strides) {
		return nil, fmt.Errorf("face: the program gave %d outputs, not %d", len(parts), 4*len(r.strides))
	}
	var cs []cells
	for i, st := range r.strides {
		cs = append(cs, cells{stride: st, cls: parts[4*i], obj: parts[4*i+1], bbox: parts[4*i+2], kps: parts[4*i+3]})
	}
	return cs, nil
}
