package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strconv"

	"github.com/shibukawa/bonsaiai/ai"
	"github.com/shibukawa/bonsaiai/ai/sound"
	"github.com/shibukawa/bonsaiai/ai/speech"
	"github.com/shibukawa/bonsaiai/internal/console"
)

// PitchCmd is "bonsaiai-audio pitch": the pitch (F0) of a voice or an instrument in WAV files with
// SwiftF0 (lars76/swift-f0), a pitch and a voicing confidence every 16 ms, as CSV or JSON on
// stdout. The model (140 KB) is downloaded and its program converted the first time. The front
// end (the three spectrograms) and the decoding of the logits run on the host; the CNN on the NPU
// (Apple's, Intel's; elsewhere or with --device cpu the ONNX file's CNN with ONNX Runtime on the
// CPU).
type PitchCmd struct {
	Files     []string `arg:"" type:"existingfile" name:"FILE.wav" help:"The WAV files (any rate; resampled to 16 kHz, channels averaged)."`
	Model     string   `short:"m" default:"pitch" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (pitch, swiftf0)."`
	JSON      bool     `name:"json" help:"Write JSON instead of CSV."`
	FMin      float64  `name:"fmin" placeholder:"HZ" help:"The lowest pitch to consider (0: 46.875)."`
	FMax      float64  `name:"fmax" placeholder:"HZ" help:"The highest pitch to consider (0: 2093.75)."`
	Threshold float64  `default:"0.5" help:"The confidence at and over which a frame is voiced."`
	CPU       bool     `name:"cpu" help:"Run the converted program on the CPU interpreter instead of the NPU (slow; for checking a conversion)."`
	Device    string   `enum:",npu,cpu" default:"" help:"Where to run: npu or cpu (ONNX Runtime; default: models.yaml's device for this machine's NPU, or BONSAIAI_DEVICE)."`
	Threads   int      `default:"1" help:"The CPU's threads with --device cpu (1: the least CPU time and energy)."`
}

// Run writes the pitch of each file.
func (c *PitchCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	st := store("")
	use, defs, err := useFor(st, sound.Pitch(c.Model))
	if err != nil {
		return err
	}
	use, defs = defined(use, defs, convertedDefinition(c.Device, c.CPU, c.Threads), sound.Pitch)
	st.Definitions = defs
	app := ai.NewApp("bonsaiai-audio", use)
	app.Store = st
	rt, err := open(ctx, app, lw)
	if err != nil {
		return err
	}
	defer rt.Close()
	d, err := use.Get(ctx, rt)
	if err != nil {
		return err
	}
	model, took := loaded(rt)
	logf("loaded %s in %.1f s, %s", model, took.Seconds(), d.Where())
	console.Running()
	for _, p := range c.Files {
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		x, err := speech.ReadWAV(raw, sound.PitchRate) // mono at 16 kHz
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		t, err := d.Detect(ctx, x, sound.PitchOptions{FMin: c.FMin, FMax: c.FMax})
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		if err := writePitch(p, t, c.Threshold, c.JSON, len(c.Files) > 1); err != nil {
			return err
		}
		logf("%s: %.1f s of audio, %d frames in %.3f s (%.0fx real time)", p, t.Seconds, len(t.Frames), t.Took.Seconds(), t.Seconds/t.Took.Seconds())
	}
	return nil
}

// convertedDefinition is what --device, --cpu and --threads make of a converted model (pitch,
// notes): a device of its own, bonsaiai's CPU executor on all the cores for --cpu, the CPU's
// threads.
func convertedDefinition(device string, cpu bool, threads int) ai.Definition {
	var d ai.Definition
	switch {
	case cpu:
		d.Device, d.Settings = "cpu", map[string]string{"cpu": "bonsaiai"}
	default:
		d.Device = device
		if threads != 1 {
			d.Settings = map[string]string{"threads": strconv.Itoa(threads)}
		}
	}
	return d
}

func writePitch(path string, t sound.PitchTrack, threshold float64, asJSON, named bool) error {
	if asJSON {
		type frame struct {
			Time       float64 `json:"timestamp"`
			Pitch      float64 `json:"pitch_hz"`
			Confidence float64 `json:"confidence"`
			Voiced     bool    `json:"voiced"`
			Loudness   float64 `json:"loudness_db"`
		}
		out := struct {
			File   string  `json:"file,omitempty"`
			Frames []frame `json:"frames"`
		}{}
		if named {
			out.File = path
		}
		for _, f := range t.Frames {
			out.Frames = append(out.Frames, frame{f.Time, f.Hz, f.Confidence, f.Confidence >= threshold, f.LoudnessDB})
		}
		return json.NewEncoder(os.Stdout).Encode(out)
	}
	if named {
		fmt.Printf("# %s\n", path)
	}
	fmt.Println("timestamp,pitch_hz,confidence,voiced,loudness_db")
	for _, f := range t.Frames {
		fmt.Printf("%.4f,%.2f,%.4f,%v,%.1f\n", f.Time, f.Hz, f.Confidence, f.Confidence >= threshold, f.LoudnessDB)
	}
	return nil
}
