// Package notes transcribes audio into notes (MIDI) with Basic Pitch (spotify/basic-pitch,
// Apache-2.0): polyphonic, any instrument, with pitch bends.
//
// Basic Pitch ships as one graph (nmp.onnx) that takes 2 second windows of 22050 Hz audio. Its
// front end, a constant-Q transform written as 1 D convolutions over the samples, and its log
// normalization run on the host, evaluated from the graph itself (the samples' axis, 43844, is
// over the Neural Engine's 16384 per dimension); the rest, the harmonic stacking and the CNN that
// gives the notes', the onsets' and the pitch contours' activations of 172 frames, is a program
// for the NPU (internal/onnx). The windows, the joining of their outputs and the making of notes
// follow basic_pitch's inference.py and note_creation.py.
package notes

import (
	"cmp"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"math"
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
	"github.com/shibukawa/hwnpu/internal/onnxrt"
)

// basic_pitch's constants.
const (
	SampleRate     = 22050
	Hop            = 256
	WindowSamples  = 43844 // 2 s less a hop
	WindowFrames   = 172
	OverlapFrames  = 30
	Pitches        = 88 // the piano's keys from A0
	ContourBins    = 3 * Pitches
	MIDIOffset     = 21
	cqtBins        = 309
	outputs        = 2*Pitches + ContourBins
	overlapSamples = OverlapFrames * Hop
	hopSamples     = WindowSamples - overlapSamples
)

// ModelFile is the model in its repository; Arch names the program.
const (
	ModelFile = "nmp.onnx"
	Arch      = "basic-pitch"
	Revision  = 2 // 2: the outputs in rows (onnx.Program.Flat)
)

// The graph's outputs: the notes', the onsets' and the contours' activations.
var heads = []string{"StatefulPartitionedCall:1", "StatefulPartitionedCall:2", "StatefulPartitionedCall:0"}

// cut is the value the program starts from: the normalized log CQT (normalized_log's
// div_no_nan) as the Reshape after it lays it out again, by the shape of a value before it.
func cut(g *onnx.Graph) (string, error) {
	where := ""
	for _, n := range g.Nodes {
		for _, o := range n.Out {
			if n.Op == "Where" && strings.Contains(o, "normalized_log") {
				where = o
			}
		}
		if where != "" && n.Op == "Reshape" && n.In[0] == where {
			return n.Out[0], nil
		}
	}
	return "", fmt.Errorf("notes: the graph has no normalized log CQT: not Basic Pitch's nmp.onnx")
}

func newProgram(g *onnx.Graph) (*onnx.Program, string, error) {
	c, err := cut(g)
	if err != nil {
		return nil, "", err
	}
	// Flat: the three outputs concatenated on their last axis, (1, 172, 440), came out wrong on
	// the Neural Engine of an M1 (the notes' activations 59% off; each alone or two of them
	// right); in rows of a common size they are right
	return &onnx.Program{G: g, Inputs: []onnx.ValueInfo{{Name: c, Shape: []int{1, WindowFrames, cqtBins}}}, Outputs: heads, Flat: true}, c, nil
}

// Setup says what Prepare downloads and converts.
type Setup struct {
	Dir, Repo, SHA256 string
	Backend           string
	Log               io.Writer
}

// Prepare downloads nmp.onnx unless it is there and converts the program unless it is. It
// returns the model's and the program's paths.
func Prepare(ctx context.Context, s Setup) (model, program string, err error) {
	log := cmp.Or[io.Writer](s.Log, io.Discard)
	if model, err = Fetch(ctx, s); err != nil {
		return "", "", err
	}
	ext := llm.ProgramExt(s.Backend) // .aimodel, .mlpackage or .xml
	program = filepath.Join(s.Dir, fmt.Sprintf("%s-r%d-fp16%s", Arch, Revision, ext))
	done := program + ".done"
	if _, err := os.Stat(done); err == nil {
		return model, program, nil
	}
	g, err := onnx.Load(model)
	if err != nil {
		return "", "", err
	}
	p, _, err := newProgram(g)
	if err != nil {
		return "", "", err
	}
	bar := console.NewBar(log, "converting "+filepath.Base(program))
	os.RemoveAll(program)
	_, err = llm.WritePart(program, p, nil, llm.Options{Format: "fp16", Backend: s.Backend}, func(pkg *mlpackage.Package) {
		pkg.Description = "Basic Pitch's harmonic stacking and CNN, from the normalized log CQT of a window (hwnpu)"
		pkg.Metadata[llm.MetaArch] = Arch
	})
	if err != nil {
		bar.Done("failed")
		return "", "", err
	}
	bar.Done(console.Duration(bar.Elapsed()))
	return model, program, os.WriteFile(done, nil, 0o644)
}

// Fetch downloads nmp.onnx unless it is there, checks its sha256 and returns its path: all the
// CPU (OpenCPU) needs.
func Fetch(ctx context.Context, s Setup) (string, error) {
	if err := os.MkdirAll(s.Dir, 0o755); err != nil {
		return "", err
	}
	model := filepath.Join(s.Dir, ModelFile)
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
		return fmt.Errorf("notes: %s has sha256 %s, not %s", path, got, want)
	}
	return nil
}

// Transcriber runs Basic Pitch.
type Transcriber struct {
	g     *onnx.Graph
	p     *onnx.Program
	input string // the graph's input
	cut   string
	prog  runner.Program
	// OnClose, when set, is called by Close (the device the program is on).
	OnClose func()
}

// Open loads the graph and the program on dev (nil: the CPU interpreter).
func Open(ctx context.Context, dev *hwnpu.Device, model, program string) (*Transcriber, error) {
	g, err := onnx.Load(model)
	if err != nil {
		return nil, err
	}
	if len(g.Inputs) != 1 {
		return nil, fmt.Errorf("notes: %s has %d inputs", model, len(g.Inputs))
	}
	// build the program once more without writing it, for its layout of the outputs
	op, c, err := newProgram(g)
	if err != nil {
		return nil, err
	}
	b := mlpackage.NewBuilder("CoreML8", nil)
	b.ImmediateLimit = math.MaxInt
	if _, _, err := op.Build(b, nil, llm.Options{Format: "fp16"}); err != nil {
		return nil, err
	}
	p, err := runner.Open(ctx, dev, program)
	if err != nil {
		return nil, err
	}
	return newTranscriber(g, op, c, p, program)
}

// OpenCPU loads nmp.onnx (Fetch's model) into ONNX Runtime on threads threads of the CPU (0 or
// 1: the calling thread), the graph from the CQT to the heads as the program has it; the host's
// CQT is the same.
func OpenCPU(model string, threads int) (*Transcriber, error) {
	g, err := onnx.Load(model)
	if err != nil {
		return nil, err
	}
	if len(g.Inputs) != 1 {
		return nil, fmt.Errorf("notes: %s has %d inputs", model, len(g.Inputs))
	}
	op, c, err := newProgram(g)
	if err != nil {
		return nil, err
	}
	b := mlpackage.NewBuilder("CoreML8", nil)
	b.ImmediateLimit = math.MaxInt
	if _, _, err := op.Build(b, nil, llm.Options{Format: "fp16"}); err != nil {
		return nil, err
	}
	raw, err := onnxrt.ReadModel(model)
	if err != nil {
		return nil, err
	}
	p, err := onnxrt.OpenProgram(raw, op, threads)
	if err != nil {
		return nil, err
	}
	return newTranscriber(g, op, c, p, model)
}

func newTranscriber(g *onnx.Graph, op *onnx.Program, c string, p runner.Program, program string) (*Transcriber, error) {
	total := 0
	for _, w := range op.Widths() {
		total += w
	}
	if y := p.Output(); len(y) != 3 || y[2] != total || y[1]*y[2] != WindowFrames*outputs {
		p.Close()
		return nil, fmt.Errorf("notes: %s gives %v, not the %d values of a window", program, y, WindowFrames*outputs)
	}
	return &Transcriber{g: g, p: op, input: g.Inputs[0].Name, cut: c, prog: p}, nil
}

// OnNPU reports whether the program runs all on the NPU.
func (t *Transcriber) OnNPU() bool { return t.prog.OnNPU() }

// Close releases the program, then calls OnClose.
func (t *Transcriber) Close() error {
	err := t.prog.Close()
	if t.OnClose != nil {
		t.OnClose()
	}
	return err
}

// Spectrum is the graph's front end on the host: the normalized log CQT (172, 309) of a window
// of WindowSamples.
func (t *Transcriber) Spectrum(window []float32) ([]float32, error) {
	x := &onnx.Tensor{Shape: []int{1, WindowSamples, 1}, F: make([]float64, WindowSamples), Type: onnx.Float}
	for i, v := range window {
		x.F[i] = float64(v)
	}
	out, err := t.g.Eval(map[string]*onnx.Tensor{t.input: x}, []string{t.cut})
	if err != nil {
		return nil, err
	}
	f := out[t.cut].F
	s := make([]float32, len(f))
	for i, v := range f {
		s[i] = float32(v)
	}
	return s, nil
}

// Activations are the model's outputs over a clip, frame by frame (11.6 ms): the notes', the
// onsets' (Pitches each) and the pitch contours' (ContourBins, a third of a semitone each).
type Activations struct {
	Frames               int
	Note, Onset, Contour []float64 // row major (frames, bins)
}

// Run gives the activations of 22050 Hz samples, as run_inference does: half the overlap of
// silence before them, windows of WindowSamples every WindowSamples - 30 frames of samples (the
// last zero padded), and of each window's 172 frames the middle ones, without 15 at each end,
// joined and cut to the clip's frames.
func (t *Transcriber) Run(ctx context.Context, x []float32) (Activations, error) {
	padded := append(make([]float32, overlapSamples/2), x...)
	keep := WindowFrames - OverlapFrames
	var a Activations
	window := make([]float32, WindowSamples)
	for start := 0; start < len(padded); start += hopSamples {
		clear(window)
		copy(window, padded[start:min(start+WindowSamples, len(padded))])
		spec, err := t.Spectrum(window)
		if err != nil {
			return a, err
		}
		y, err := t.window(ctx, spec)
		if err != nil {
			return a, err
		}
		for f := OverlapFrames / 2; f < OverlapFrames/2+keep; f++ {
			row := y[f*outputs:]
			for j := range Pitches {
				a.Note = append(a.Note, float64(row[j]))
				a.Onset = append(a.Onset, float64(row[Pitches+j]))
			}
			for j := range ContourBins {
				a.Contour = append(a.Contour, float64(row[2*Pitches+j]))
			}
		}
		a.Frames += keep
	}
	n := int(float64(len(x)) / float64(hopSamples) * float64(keep))
	n = min(n, a.Frames)
	a.Frames = n
	a.Note, a.Onset, a.Contour = a.Note[:n*Pitches], a.Onset[:n*Pitches], a.Contour[:n*ContourBins]
	return a, nil
}

// window runs the program on a window's spectrum and lays its outputs out by frame: (172, 440),
// the notes', the onsets' and the contours' activations of each.
func (t *Transcriber) window(ctx context.Context, spec []float32) ([]float32, error) {
	y, err := t.prog.Run(ctx, map[string][]float32{"x": spec})
	if err != nil {
		return nil, err
	}
	parts := t.p.Split(y)
	out := make([]float32, WindowFrames*outputs)
	at := 0
	for _, h := range heads {
		v := parts[h].F
		w := len(v) / WindowFrames
		for f := range WindowFrames {
			for j := range w {
				out[f*outputs+at+j] = float32(v[f*w+j])
			}
		}
		at += w
	}
	return out, nil
}
