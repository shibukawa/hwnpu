package segment

import (
	"cmp"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
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
	path, err := ConvertEncoder(ctx, backend, encoder, decoder, dir, log)
	if err != nil {
		return nil, err
	}
	p, err := runner.Open(ctx, dev, path)
	if err != nil {
		return nil, fmt.Errorf("segment: %w", err)
	}
	return &Model{dec: dec, prog: p, progName: filepath.Base(path)}, nil
}

// ConvertEncoder converts the encoder (its ONNX file) into a program of backend in dir unless it
// is there (or a shared layer has it), and returns its path: what OpenProgram loads, made with no
// device (hwnpu-dev models package). A Core ML package gets the decoder's part file too
// (DecoderPart, from the ONNX file decoder; none when decoder is ""), with which the CPU runs it
// without the ONNX files (OpenCPUProgram).
func ConvertEncoder(ctx context.Context, backend, encoder, decoder, dir string, log io.Writer) (string, error) {
	name, format, err := encoderName(ctx, backend, filepath.Base(encoder), dir)
	if err != nil {
		return "", err
	}
	path, done, export := llm.ProgramPath(dir, name, backend)
	if decoder != "" {
		llm.DropStaleCoreML(path, done, DecoderPart(path))
	}
	if !llm.ProgramReady(ctx, dir, name, done, export) {
		g, err := onnx.Load(encoder)
		if err != nil {
			return "", err
		}
		if len(g.Inputs) != 1 || len(g.Outputs) != 1 {
			return "", fmt.Errorf("segment: %s has %d inputs and %d outputs, not the encoder's 1 and 1", encoder, len(g.Inputs), len(g.Outputs))
		}
		bar := console.NewBar(cmp.Or[io.Writer](log, io.Discard), "converting "+name)
		if err := llm.StartProgram(ctx, path, export); err != nil {
			bar.Done("failed")
			return "", fmt.Errorf("segment: %w", err)
		}
		if _, err := llm.WritePart(path, EncoderProgram(g), nil, llm.Options{Format: format, Backend: backend}, func(pkg *mlpackage.Package) {
			pkg.Description = fmt.Sprintf("EfficientSAM's image encoder (%s), %dx%d, %s (hwnpu)", filepath.Base(encoder), Size, Size, format)
		}); err != nil {
			bar.Done("failed")
			return "", fmt.Errorf("segment: %w", err)
		}
		if backend == llm.BackendCoreML && decoder != "" {
			if err := writeDecoderPart(path, decoder); err != nil {
				bar.Done("failed")
				return "", err
			}
		}
		bar.Done(console.Duration(bar.Elapsed()))
		if err := llm.FinishProgram(ctx, dir, name, done, export); err != nil {
			return "", err
		}
	}
	return path, nil
}

// encoderName is the name of the encoder's program for backend in dir (ProgramPath's NAME), of the
// encoder's file name in the repository, and its weights' format.
func encoderName(ctx context.Context, backend, encoderFile, dir string) (name, format string, err error) {
	stem := strings.TrimSuffix(filepath.Base(encoderFile), filepath.Ext(encoderFile))
	// Core AI: NAME.coreai/model.aimodel; Core ML and Intel: NAME.mlpackage, NAME.xml
	if format, err = chat.ConvertFormat(ctx, filepath.Join(dir, stem), backend, "fp16"); err != nil {
		return "", "", err
	}
	return fmt.Sprintf("%s-r%d-%s-%d", stem, ProgramRevision, format, Size), format, nil
}

// DecoderPart is the part file (onnx.PartExt) in the encoder's Core ML package: the decoder's
// whole graph, which the host runs on the encoder's embedding and the prompt.
func DecoderPart(program string) string { return filepath.Join(program, "decoder"+onnx.PartExt) }

func hasDecoderPart(program string) bool {
	_, err := os.Stat(DecoderPart(program))
	return err == nil
}

func writeDecoderPart(program, decoder string) error {
	dec, err := onnx.LoadLazy(decoder)
	if err != nil {
		return err
	}
	var in, out []string
	for _, v := range dec.Inputs {
		in = append(in, v.Name)
	}
	for _, v := range dec.Outputs {
		out = append(out, v.Name)
	}
	if err := onnx.WritePart(DecoderPart(program), onnx.Part{Graph: dec.Subgraph(in, out)}); err != nil {
		return fmt.Errorf("segment: %w", err)
	}
	return nil
}

// PrepareCPU makes the encoder's Core ML package with the decoder's part file ready in dir (what
// the CPU runs on hwnpu's own executor, OpenCPUProgram): here or from the shared layers, else
// converted, the ONNX files downloaded for it then. It returns the package's path.
func PrepareCPU(ctx context.Context, s Setup, dir string) (string, error) {
	name, _, err := encoderName(ctx, llm.BackendCoreML, s.Encoder, dir)
	if err != nil {
		return "", err
	}
	path, done, export := llm.ProgramPath(dir, name, llm.BackendCoreML)
	llm.DropStaleCoreML(path, done, DecoderPart(path))
	ready := llm.ProgramReady(ctx, dir, name, done, export)
	if ready && hasDecoderPart(path) {
		return path, nil
	}
	if !ready {
		if err := llm.MayConvert(ctx, path); err != nil {
			return "", err // nothing converted now: the ONNX files are not downloaded either
		}
	}
	enc, dec, err := Fetch(ctx, s)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	return ConvertEncoder(ctx, llm.BackendCoreML, enc, dec, dir, s.Log)
}

// OpenCPUProgram runs the encoder's Core ML package of PrepareCPU on hwnpu's own executor (the
// CPU) and the decoder from its part file, mapped: no ONNX file.
func OpenCPUProgram(ctx context.Context, program string) (*Model, error) {
	part, err := onnx.ReadPart(DecoderPart(program))
	if err != nil {
		return nil, fmt.Errorf("segment: %w", err)
	}
	if part.Graph == nil {
		return nil, fmt.Errorf("segment: %s holds no decoder", DecoderPart(program))
	}
	p, err := runner.Open(ctx, nil, program)
	if err != nil {
		return nil, fmt.Errorf("segment: %w", err)
	}
	return &Model{dec: part.Graph, prog: p, progName: filepath.Base(program), cpu: true}, nil
}
