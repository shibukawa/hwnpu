package segment

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/mlpackage"
	"github.com/shibukawa/hwnpu/internal/onnx"
)

// ProgramRevision counts the changes to the encoder's program that call for converting it again.
const ProgramRevision = 1

// EncoderProgram is the encoder's ONNX graph g as a program of a Size x Size picture (x, its RGB
// in [0, 1], channels first) to the embedding (1, 256, Grid, Grid): the graph's own resize of
// its input to Size x Size is then the input itself (internal/onnx lowers a Resize to its own
// size as nothing), so the host stretches the picture as the graph does (Resize).
func EncoderProgram(g *onnx.Graph) *onnx.Program {
	var outs []string
	for _, o := range g.Outputs {
		outs = append(outs, o.Name)
	}
	return &onnx.Program{G: g, Inputs: []onnx.ValueInfo{{Name: g.Inputs[0].Name, Shape: []int{1, 3, Size, Size}}}, Outputs: outs}
}

// OpenProgram runs the encoder (its ONNX file) as a program of backend (llm.Options.Backend: Core
// ML packages, or OpenVINO IR for Intel's NPU) on dev, converted into dir unless it is there
// (NAME.done), and loads the decoder's ONNX file for the host. The conversion's progress goes to
// log (nil for none).
func OpenProgram(ctx context.Context, dev *hwnpu.Device, backend, encoder, decoder, dir string, log io.Writer) (*Model, error) {
	dec, err := onnx.Load(decoder)
	if err != nil {
		return nil, err
	}
	stem := strings.TrimSuffix(filepath.Base(encoder), filepath.Ext(encoder))
	// Core AI: NAME.coreai/model.aimodel; Core ML and Intel: NAME.mlpackage, NAME.xml
	name := fmt.Sprintf("%s-r%d-fp16-%d", stem, ProgramRevision, Size)
	path, done, export := llm.ProgramPath(dir, name, backend)
	if !llm.ExportDone(done) {
		g, err := onnx.Load(encoder)
		if err != nil {
			return nil, err
		}
		if len(g.Inputs) != 1 || len(g.Outputs) != 1 {
			return nil, fmt.Errorf("segment: %s has %d inputs and %d outputs, not the encoder's 1 and 1", encoder, len(g.Inputs), len(g.Outputs))
		}
		bar := console.NewBar(cmp.Or[io.Writer](log, io.Discard), "converting "+name)
		if err := llm.StartProgram(path, export); err != nil {
			bar.Done("failed")
			return nil, fmt.Errorf("segment: %w", err)
		}
		if _, err := llm.WritePart(path, EncoderProgram(g), nil, llm.Options{Format: "fp16", Backend: backend}, func(pkg *mlpackage.Package) {
			pkg.Description = fmt.Sprintf("EfficientSAM's image encoder (%s), %dx%d, fp16 (hwnpu)", filepath.Base(encoder), Size, Size)
		}); err != nil {
			bar.Done("failed")
			return nil, fmt.Errorf("segment: %w", err)
		}
		bar.Done(console.Duration(bar.Elapsed()))
		if err := llm.FinishProgram(dir, name, done, export); err != nil {
			return nil, err
		}
	}
	p, err := runner.Open(ctx, dev, path)
	if err != nil {
		return nil, fmt.Errorf("segment: %w", err)
	}
	return &Model{dec: dec, prog: p, progName: filepath.Base(path)}, nil
}
