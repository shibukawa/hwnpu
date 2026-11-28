package face

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

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
// takes the size. The sizes are CPUSizes'. path "" (the ONNX file not downloaded: the Core ML
// packages taken from the shared layers, PreparePrograms) runs the programs in dir alone, their
// outputs' shapes from their metadata; a size with no program there is then an error.
func OpenProgram(ctx context.Context, dev *hwnpu.Device, backend, path, dir string) (*Detector, error) {
	r := &programRunner{dev: dev, backend: backend, dir: dir, strides: []int{8, 16, 32}}
	if r.backend == "" {
		r.backend = llm.BackendCoreML
	}
	if path != "" {
		if err := r.load(path); err != nil {
			return nil, err
		}
	}
	x := Export{Name: "YuNet", Kind: "face-detect", Sizes: CPUSizes, Strides: r.strides}
	return newDetector(x, "npu", r), nil
}

// load reads the ONNX file the programs are converted from.
func (r *programRunner) load(path string) error {
	g, err := onnx.Load(path)
	if err != nil {
		return err
	}
	if len(g.Inputs) != 1 {
		return fmt.Errorf("face: %s has %d inputs", path, len(g.Inputs))
	}
	r.g = g
	return nil
}

// programRunner runs a size's program, loaded for the first picture that takes it.
type programRunner struct {
	g       *onnx.Graph
	dev     *hwnpu.Device
	backend string
	dir     string
	strides []int
	w, h    int
	// shapes are the outputs' shapes of the program loaded, in the order of outputs()
	shapes [][]int
	prog   llmrunner.Program
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
	_, err := PreparePrograms(ctx, backend, func() (string, error) { return path, nil }, dir, sizes)
	return err
}

// PreparePrograms is ConvertPrograms that looks for the programs here and in the shared layers
// first and downloads the ONNX file (onnxFile) only for one it converts. It returns the ONNX
// file's path, "" when every program was here or fetched (OpenProgram runs them without it).
func PreparePrograms(ctx context.Context, backend string, onnxFile func() (string, error), dir string, sizes [][2]int) (string, error) {
	if backend == "" {
		backend = llm.BackendCoreML
	}
	r := &programRunner{backend: backend, dir: dir, strides: []int{8, 16, 32}}
	path := ""
	for _, sz := range sizes {
		if r.ready(ctx, sz[0], sz[1]) {
			continue
		}
		if r.g == nil {
			if err := llm.MayConvert(ctx, filepath.Join(dir, r.name(sz[0], sz[1]))); err != nil {
				return "", err // nothing converted now: the ONNX file is not downloaded either
			}
			var err error
			if path, err = onnxFile(); err != nil {
				return "", err
			}
			if err := r.load(path); err != nil {
				return "", err
			}
		}
		if _, err := r.convert(ctx, sz[0], sz[1]); err != nil {
			return "", err
		}
	}
	return path, nil
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
	format, err := chat.ConvertFormat(ctx, filepath.Join(r.dir, "yunet"), r.backend, "fp16")
	if err != nil {
		return "", err
	}
	name := fmt.Sprintf("yunet-r%d-%s-%dx%d", ProgramRevision, format, W, H)
	path, done, export := llm.ProgramPath(r.dir, name, r.backend)
	if llm.ProgramReady(ctx, r.dir, name, done, export) {
		return path, nil
	}
	if r.g == nil {
		return "", fmt.Errorf("face: no program of %dx%d here nor in the shared layers, and no ONNX file to convert it from", W, H)
	}
	shapes, err := r.outShapes(W, H)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(r.dir, 0o755); err != nil {
		return "", err
	}
	if err := llm.StartProgram(ctx, path, export); err != nil {
		return "", err
	}
	_, err = llm.WritePart(path, r.program(W, H), nil, llm.Options{Format: format, Backend: r.backend}, func(pkg *mlpackage.Package) {
		pkg.Description = fmt.Sprintf("YuNet (face_detection_yunet_2023mar), %dx%d, %s (hwnpu)", W, H, format)
		if r.backend == llm.BackendCoreML {
			// the CPU's package, which a machine may run without the ONNX file (the Core AI
			// export and Intel's IR stay as they were)
			pkg.Metadata[metaOutShapes] = formatShapes(shapes)
		}
	})
	if err != nil {
		return "", fmt.Errorf("face: %w", err)
	}
	return path, llm.FinishProgram(ctx, r.dir, name, done, export)
}

// name is the program of W x H's name in r.dir (ProgramPath's NAME), without the backend's
// extension: for MayConvert's message.
func (r *programRunner) name(W, H int) string {
	return fmt.Sprintf("yunet-r%d-%dx%d", ProgramRevision, W, H)
}

// ready reports whether the program of W x H is here or in the shared layers (fetched then).
func (r *programRunner) ready(ctx context.Context, W, H int) bool {
	format, err := chat.ConvertFormat(ctx, filepath.Join(r.dir, "yunet"), r.backend, "fp16")
	if err != nil {
		return false
	}
	name := fmt.Sprintf("yunet-r%d-%s-%dx%d", ProgramRevision, format, W, H)
	path, done, export := llm.ProgramPath(r.dir, name, r.backend)
	stale := func() bool {
		if r.backend != llm.BackendCoreML {
			return false
		}
		_, err := readOutShapes(path)
		return err != nil
	}
	if llm.ExportDone(done) && stale() {
		// a Core ML package of before its outputs' shapes were written: made again (or taken
		// from the shared layers), as a run without the ONNX file needs them
		os.RemoveAll(path)
		os.Remove(done)
	}
	return llm.ProgramReady(ctx, r.dir, name, done, export) && !stale()
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
	path, err := r.convert(ctx, W, H)
	if err != nil {
		return err
	}
	// Split needs the outputs' widths: the ONNX graph's, else the program's metadata
	var shapes [][]int
	if r.g != nil {
		if shapes, err = r.outShapes(W, H); err != nil {
			return err
		}
	} else if shapes, err = readOutShapes(path); err != nil {
		return err
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
	r.shapes, r.prog, r.w, r.h = shapes, prog, W, H
	return nil
}

// metaOutShapes is the program's metadata that holds its outputs' shapes (formatShapes), which a
// run without the ONNX file splits the program's output by.
const metaOutShapes = "hwnpu.outshapes"

// outShapes are the outputs' shapes of the program of W x H, built again without writing it.
func (r *programRunner) outShapes(W, H int) ([][]int, error) {
	p := r.program(W, H)
	b := mlpackage.NewBuilder("CoreML8", nil)
	b.ImmediateLimit = math.MaxInt
	if _, _, err := p.Build(b, nil, llm.Options{Format: "fp16"}); err != nil {
		return nil, fmt.Errorf("face: %w", err)
	}
	return p.OutShapes(), nil
}

// formatShapes writes shapes as 1x1600x1;1x1600x4;... (metaOutShapes).
func formatShapes(shapes [][]int) string {
	var parts []string
	for _, s := range shapes {
		var dims []string
		for _, d := range s {
			dims = append(dims, strconv.Itoa(d))
		}
		parts = append(parts, strings.Join(dims, "x"))
	}
	return strings.Join(parts, ";")
}

// readOutShapes reads the outputs' shapes from the metadata of the Core ML package at path.
func readOutShapes(path string) ([][]int, error) {
	m, err := mlpackage.ReadModel(path)
	if err != nil {
		return nil, fmt.Errorf("face: %w (the outputs' shapes of a program without its ONNX file)", err)
	}
	v := m.GetDescription().GetMetadata().GetUserDefined()[metaOutShapes]
	if v == "" {
		return nil, fmt.Errorf("face: %s has no %s: converted before it was written, its ONNX file is needed", filepath.Base(path), metaOutShapes)
	}
	var out [][]int
	for _, part := range strings.Split(v, ";") {
		var s []int
		for _, d := range strings.Split(part, "x") {
			n, err := strconv.Atoi(d)
			if err != nil {
				return nil, fmt.Errorf("face: %s: %s %q: %w", filepath.Base(path), metaOutShapes, v, err)
			}
			s = append(s, n)
		}
		out = append(out, s)
	}
	return out, nil
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
	parts := onnx.SplitFlat(y, r.shapes)
	if len(parts) != 4*len(r.strides) {
		return nil, fmt.Errorf("face: the program gave %d outputs, not %d", len(parts), 4*len(r.strides))
	}
	var cs []cells
	for i, st := range r.strides {
		cs = append(cs, cells{stride: st, cls: parts[4*i], obj: parts[4*i+1], bbox: parts[4*i+2], kps: parts[4*i+3]})
	}
	return cs, nil
}
