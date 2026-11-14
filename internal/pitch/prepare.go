package pitch

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
)

// ModelFile is the model's file in its repository.
const ModelFile = "model.onnx"

// Setup says what Prepare downloads and converts.
type Setup struct {
	// Dir is where the model and its program are kept; Repo is where the model is downloaded
	// from when it is not there (chat.Fetch's OWNER/NAME@REVISION).
	Dir, Repo string
	// Backend is the program's format (llm.Options.Backend): coreai, coreml (what the CPU
	// interpreter runs) or intel.
	Backend string
	Log     io.Writer
}

// Prepare downloads model.onnx into s.Dir unless it is there, checks that it is swift-f0
// 0.3.0's (ModelSHA256), and converts the CNN's program unless it is: a call of ProgramFrames
// frames (30 s and the context around), in fp16 (the 14K weights are nothing to compress).
// It returns the model's and the program's paths.
func Prepare(ctx context.Context, s Setup) (model, program string, err error) {
	log := cmp.Or[io.Writer](s.Log, io.Discard)
	if model, err = Fetch(ctx, s); err != nil {
		return "", "", err
	}
	ext := llm.ProgramExt(s.Backend) // .aimodel, .mlpackage or .xml
	program = filepath.Join(s.Dir, fmt.Sprintf("%s-r%d-fp16-t%d%s", Arch, Revision, ProgramFrames, ext))
	done := program + ".done"
	if _, err := os.Stat(done); err == nil {
		return model, program, nil
	}
	m, err := Load(model)
	if err != nil {
		return "", "", err
	}
	bar := console.NewBar(log, "converting "+filepath.Base(program))
	os.RemoveAll(program)
	_, err = llm.WritePart(program, &Program{Frames: ProgramFrames}, m.Weights(), llm.Options{Format: "fp16", Backend: s.Backend}, func(p *mlpackage.Package) {
		p.Description = fmt.Sprintf("SwiftF0's CNN (swift-f0 0.3.0), %d frames a call, fp16 (hwnpu)", ProgramFrames)
		p.Metadata[llm.MetaArch] = Arch
	})
	if err != nil {
		bar.Done("failed")
		return "", "", err
	}
	bar.Done(console.Duration(bar.Elapsed()))
	return model, program, os.WriteFile(done, nil, 0o644)
}

// Fetch downloads model.onnx into s.Dir unless it is there, checks that it is swift-f0 0.3.0's
// and returns its path: all the CPU (OpenCPU) needs.
func Fetch(ctx context.Context, s Setup) (string, error) {
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return "", err
	}
	model := filepath.Join(s.Dir, ModelFile)
	if err := chat.Fetch(ctx, s.Repo, ModelFile, model, cmp.Or[io.Writer](s.Log, io.Discard)); err != nil {
		return "", err
	}
	return model, checkSHA(model)
}

func checkSHA(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return err
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != ModelSHA256 {
		return fmt.Errorf("pitch: %s is not swift-f0 0.3.0's model (sha256 %s, want %s)", path, got, ModelSHA256)
	}
	return nil
}

// Detector runs SwiftF0 with its program on a device.
type Detector struct {
	*Model
	prog runner.Program
	cpu  *cpuRunner // OpenCPU's, in place of prog
}

// Open loads the model and its program on dev (nil: the CPU interpreter).
func Open(ctx context.Context, dev *hwnpu.Device, model, program string) (*Detector, error) {
	m, err := Load(model)
	if err != nil {
		return nil, err
	}
	p, err := runner.Open(ctx, dev, program)
	if err != nil {
		return nil, err
	}
	if x, y := p.Inputs()["x"], p.Output(); len(x) != 4 || x[3] != ProgramFrames || len(y) != 3 || y[1] != Bins+1 {
		p.Close()
		return nil, fmt.Errorf("pitch: %s takes %v and gives %v, not a program of %d frames", program, x, y, ProgramFrames)
	}
	return &Detector{Model: m, prog: p}, nil
}

// OnNPU reports whether the program runs all on the NPU.
func (d *Detector) OnNPU() bool { return d.prog != nil && d.prog.OnNPU() }

// Pitch estimates the pitch of 16 kHz samples (Model.Detect) with the program.
func (d *Detector) Pitch(ctx context.Context, x []float32, fmin, fmax float64) (Result, error) {
	if d.cpu != nil {
		return d.Detect(x, fmin, fmax, d.cpu.run)
	}
	return d.Detect(x, fmin, fmax, func(in, mask []float32) ([]float32, error) {
		return d.prog.Run(ctx, map[string][]float32{"x": in, "mask": mask})
	})
}

func (d *Detector) Close() error {
	if d.cpu != nil {
		return d.cpu.s.Close()
	}
	return d.prog.Close()
}
