package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/shibukawa/bonsaiai/ai"
	"github.com/shibukawa/bonsaiai/ai/sound"
	"github.com/shibukawa/bonsaiai/ai/speech"
	"github.com/shibukawa/bonsaiai/internal/console"
)

// NotesCmd is "bonsaiai-audio notes": the notes of music in WAV files with Basic Pitch
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
	CPU     bool     `name:"cpu" help:"Run the converted program on the CPU interpreter instead of the NPU (slow; for checking a conversion)."`
	Device  string   `enum:",npu,cpu" default:"" help:"Where to run: npu or cpu (ONNX Runtime; default: models.yaml's device for this machine's NPU, or BONSAIAI_DEVICE)."`
	Threads int      `default:"1" help:"The CPU's threads with --device cpu (1: the least CPU time and energy)."`
}

// Run writes the notes of each file.
func (c *NotesCmd) Run(ctx context.Context) error {
	if c.MIDI != "" && len(c.Files) != 1 {
		return fmt.Errorf("--midi takes one input, not %d", len(c.Files))
	}
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	st := store("")
	use, defs, err := useFor(st, sound.Notes(c.Model))
	if err != nil {
		return err
	}
	use, defs = defined(use, defs, convertedDefinition(c.Device, c.CPU, c.Threads), sound.Notes)
	st.Definitions = defs
	app := ai.NewApp("bonsaiai-audio", use)
	app.Store = st
	rt, err := open(ctx, app, lw)
	if err != nil {
		return err
	}
	defer rt.Close()
	tr, err := use.Get(ctx, rt)
	if err != nil {
		return err
	}
	model, took := loaded(rt)
	logf("loaded %s in %.1f s, %s", model, took.Seconds(), tr.Where())
	console.Running()
	o := sound.NotesOptions{Onset: c.Onset, Frame: c.Frame, MinNoteMS: c.MinNote, MinHz: c.FMin, MaxHz: c.FMax, NoMelodia: !c.Melodia}
	for _, p := range c.Files {
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		x, err := speech.ReadWAV(raw, sound.NotesRate) // mono
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		l, err := tr.Transcribe(ctx, x, o)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		ns := l.Notes
		logf("%s: %.1f s of audio, %d notes in %.2f s (%.0fx real time)", p, l.Seconds, len(ns), l.Took.Seconds(), l.Seconds/l.Took.Seconds())
		if c.JSON {
			if err := json.NewEncoder(os.Stdout).Encode(struct {
				File  string       `json:"file"`
				Notes []sound.Note `json:"notes"`
			}{p, ns}); err != nil {
				return err
			}
		} else {
			if len(c.Files) > 1 {
				fmt.Printf("# %s\n", p)
			}
			fmt.Println("start,end,pitch,velocity")
			for _, n := range ns {
				fmt.Printf("%.3f,%.3f,%d,%d\n", n.Start, n.End, n.Pitch, n.Velocity())
			}
		}
		if c.MIDI != "" {
			f, err := os.Create(c.MIDI)
			if err != nil {
				return err
			}
			if err := sound.WriteMIDI(f, ns); err != nil {
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
