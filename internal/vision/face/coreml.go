package face

import (
	"context"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	llmrunner "github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/onnx"
)

// ProgramRevision counts the changes to the Core ML programs OpenProgram writes that call for
// converting them again.
const ProgramRevision = 1

// OpenProgram runs the ONNX file (face_detection_yunet_2023mar.onnx) as Core ML programs on dev
// (Apple's Neural Engine; nil: the CPU interpreter): the whole graph, its 12 outputs (each
// stride's cls, obj, bbox and kps) in one program per size, converted into dir when a picture
// first takes the size. The sizes are CPUSizes'.
func OpenProgram(ctx context.Context, dev *hwnpu.Device, path, dir string) (*Detector, error) {
	g, err := onnx.Load(path)
	if err != nil {
		return nil, err
	}
	if len(g.Inputs) != 1 {
		return nil, fmt.Errorf("face: %s has %d inputs", path, len(g.Inputs))
	}
	x := Export{Name: "YuNet", Kind: "face-detect", Sizes: CPUSizes, Strides: []int{8, 16, 32}}
	return newDetector(x, "npu", &programRunner{g: g, dev: dev, dir: dir, strides: x.Strides}), nil
}

// programRunner runs a size's program, loaded for the first picture that takes it.
type programRunner struct {
	g       *onnx.Graph
	dev     *hwnpu.Device
	dir     string
	strides []int
	w, h    int
	p       *onnx.Program
	prog    llmrunner.Program
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

// use loads the program of W x H, converting it first when it is missing.
func (r *programRunner) use(ctx context.Context, W, H int) error {
	if r.prog != nil && r.w == W && r.h == H {
		return nil
	}
	defer console.Loading(fmt.Sprintf("YuNet %dx%d in %s", W, H, r.dir))()
	if err := r.close(); err != nil {
		return err
	}
	path := filepath.Join(r.dir, fmt.Sprintf("yunet-r%d-fp16-%dx%d.mlpackage", ProgramRevision, W, H))
	p := r.program(W, H)
	if _, err := os.Stat(path + ".done"); err != nil {
		os.RemoveAll(path)
		if err := os.MkdirAll(r.dir, 0o755); err != nil {
			return err
		}
		_, err := llm.WritePart(path, p, nil, llm.Options{Format: "fp16", Backend: llm.BackendCoreML}, func(pkg *mlpackage.Package) {
			pkg.Description = fmt.Sprintf("YuNet (face_detection_yunet_2023mar), %dx%d, fp16 (hwnpu)", W, H)
		})
		if err != nil {
			return fmt.Errorf("face: %w", err)
		}
		if err := os.WriteFile(path+".done", nil, 0o644); err != nil {
			return err
		}
	} else {
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
	if r.dev != nil && !prog.OnNPU() {
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
	x := make([]float32, 3*W*H) // B, G, R planes in 0..255, as cpuRunner's
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
	out := r.p.Split(y)
	var cs []cells
	for _, st := range r.strides {
		c := cells{stride: st}
		for _, f := range []struct {
			k string
			p *[]float32
		}{{"cls", &c.cls}, {"obj", &c.obj}, {"bbox", &c.bbox}, {"kps", &c.kps}} {
			t, ok := out[fmt.Sprintf("%s_%d", f.k, st)]
			if !ok {
				return nil, fmt.Errorf("face: the program gave no %s_%d", f.k, st)
			}
			v := make([]float32, len(t.F))
			for i, e := range t.F {
				v[i] = float32(e)
			}
			*f.p = v
		}
		cs = append(cs, c)
	}
	return cs, nil
}
