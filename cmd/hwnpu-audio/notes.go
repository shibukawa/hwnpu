package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/notes"
)

// NotesCmd is "hwnpu-audio notes": the notes of music in WAV files with Basic Pitch
// (spotify/basic-pitch), as CSV or JSON on stdout and optionally a MIDI file. The model (230
// KB) is downloaded and converted the first time. The constant-Q transform runs on the host, the
// network on the NPU (Apple's, Intel's; elsewhere or with --device cpu the ONNX file's network
// with ONNX Runtime on the CPU), the making of notes on the host.
type NotesCmd struct {
	Files   []string `arg:"" type:"existingfile" name:"FILE.wav" help:"The WAV files (any rate; resampled to 22050 Hz, channels averaged)."`
	Model   string   `short:"m" default:"notes" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (notes, basic-pitch)."`
	MIDI    string   `name:"midi" type:"path" placeholder:"FILE.mid" help:"Also write a MIDI file, with pitch bends (one input)."`
	JSON    bool     `name:"json" help:"Write JSON (with the pitch bends) instead of CSV."`
	Onset   float64  `default:"${onset}" help:"The least onset activation of a note's start."`
	Frame   float64  `default:"${frame}" help:"The least note activation that keeps a note on."`
	MinNote float64  `name:"min-note" default:"${min_note}" placeholder:"MS" help:"The shortest note, in milliseconds."`
	FMin    float64  `name:"fmin" placeholder:"HZ" help:"Drop the notes below this frequency (0: none)."`
	FMax    float64  `name:"fmax" placeholder:"HZ" help:"Drop the notes above this frequency (0: none)."`
	Melodia bool     `default:"${melodia}" negatable:"" help:"Also take notes from the strong activations left without an onset (basic-pitch's melodia trick)."`
	Dir     string   `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's --dir)."`
	Cache   string   `placeholder:"DIR" help:"Compile cache (default: hwnpu in the user cache directory)."`
	CPU     bool     `name:"cpu" help:"Run the converted program on the CPU interpreter instead of the NPU (slow; for checking a conversion)."`
	Device  string   `enum:",npu,cpu" default:"" help:"Where to run: npu or cpu (ONNX Runtime; default: models.yaml's device for this machine's NPU, or HWNPU_DEVICE)."`
	Threads int      `default:"1" help:"The CPU's threads with --device cpu (1: the least CPU time and energy)."`
}

// Run writes the notes of each file.
func (c *NotesCmd) Run(ctx context.Context) error {
	if c.MIDI != "" && len(c.Files) != 1 {
		return fmt.Errorf("--midi takes one input, not %d", len(c.Files))
	}
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	m, err := resolveKind(c.Model, chat.KindNotes, "", lw)
	if err != nil {
		return err
	}
	setup := notes.Setup{Dir: filepath.Join(modelsDir(c.Dir), m.DirName()), Repo: m.Source(), SHA256: m.SHA256[notes.ModelFile], Log: lw}
	var tr *notes.Transcriber
	t0 := time.Now()
	ort, builtin := cpuRoute(m, c.Device, c.CPU, c.Threads, logf)
	c.CPU = c.CPU || builtin
	if ort {
		model, err := notes.Fetch(ctx, setup)
		if err != nil {
			return err
		}
		if tr, err = notes.OpenCPU(model, c.Threads); err != nil {
			return err
		}
		logf("loaded %s in %.1f s, on the CPU (ONNX Runtime, %d threads)", m.Name, time.Since(t0).Seconds(), max(1, c.Threads))
	} else if tr, err = c.openConverted(ctx, m, setup, logf); err != nil {
		return err
	}
	defer tr.Close()
	console.Running()
	return c.transcribe(ctx, tr, logf)
}

// openConverted loads Basic Pitch's converted program on the NPU (or the CPU interpreter).
func (c *NotesCmd) openConverted(ctx context.Context, m chat.Model, setup notes.Setup, logf func(string, ...any)) (*notes.Transcriber, error) {
	backend := llm.BackendCoreML
	var dev *hwnpu.Device
	if !c.CPU {
		d, device, err := llm.OpenDevice(ctx, "", cacheDir(c.Cache))
		if err != nil {
			return nil, err
		}
		if backend = llm.ProgramBackend(device); backend == "" {
			d.Close()
			return nil, fmt.Errorf("%s runs neither Core ML packages nor OpenVINO IR (--device cpu runs it on the CPU)", device)
		}
		dev = d
	}
	closeDev := func() {
		if dev != nil {
			dev.Close()
		}
	}
	setup.Backend = backend
	model, program, err := notes.Prepare(ctx, setup)
	if err != nil {
		closeDev()
		return nil, err
	}
	t0 := time.Now()
	tr, err := notes.Open(ctx, dev, model, program)
	if err != nil {
		closeDev()
		return nil, err
	}
	tr.OnClose = closeDev
	logf("loaded %s in %.1f s, %s", m.Name, time.Since(t0).Seconds(), placement(c.CPU, tr.OnNPU()))
	return tr, nil
}

// transcribe writes the notes of each file with tr.
func (c *NotesCmd) transcribe(ctx context.Context, tr *notes.Transcriber, logf func(string, ...any)) error {
	o := notes.DefaultOptions
	o.OnsetThreshold, o.FrameThreshold, o.MinNoteMS, o.MinHz, o.MaxHz, o.Melodia = c.Onset, c.Frame, c.MinNote, c.FMin, c.FMax, c.Melodia
	for _, p := range c.Files {
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		x, err := lfm3audio.ReadWAVAt(raw, notes.SampleRate) // mono
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		start := time.Now()
		a, err := tr.Run(ctx, x)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		ns := notes.Notes(a, o)
		secs := float64(len(x)) / notes.SampleRate
		logf("%s: %.1f s of audio, %d notes in %.2f s (%.0fx real time)", p, secs, len(ns), time.Since(start).Seconds(), secs/time.Since(start).Seconds())
		if c.JSON {
			if err := json.NewEncoder(os.Stdout).Encode(struct {
				File  string       `json:"file"`
				Notes []notes.Note `json:"notes"`
			}{p, ns}); err != nil {
				return err
			}
		} else {
			if len(c.Files) > 1 {
				fmt.Printf("# %s\n", p)
			}
			fmt.Println("start,end,pitch,velocity")
			for _, n := range ns {
				fmt.Printf("%.3f,%.3f,%d,%d\n", n.Start, n.End, n.Pitch, int(math.RoundToEven(127*n.Amplitude)))
			}
		}
		if c.MIDI != "" {
			f, err := os.Create(c.MIDI)
			if err != nil {
				return err
			}
			if err := notes.WriteMIDI(f, ns); err != nil {
				f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return err
			}
			logf("wrote %s", c.MIDI)
		}
	}
	return nil
}
