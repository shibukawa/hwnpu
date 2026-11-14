package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/pitch"
)

// PitchCmd is "hwnpu-audio pitch": the pitch (F0) of a voice or an instrument in WAV files with
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
	Dir       string   `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's --dir)."`
	Cache     string   `placeholder:"DIR" help:"Compile cache (default: hwnpu in the user cache directory)."`
	CPU       bool     `name:"cpu" help:"Run the converted program on the CPU interpreter instead of the NPU (slow; for checking a conversion)."`
	Device    string   `enum:",npu,cpu" default:"" help:"Where to run: npu or cpu (ONNX Runtime; default: models.yaml's device for this machine's NPU, or HWNPU_DEVICE)."`
	Threads   int      `default:"1" help:"The CPU's threads with --device cpu (1: the least CPU time and energy)."`
}

// Run writes the pitch of each file.
func (c *PitchCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	m, err := resolveKind(c.Model, chat.KindPitch, "", lw)
	if err != nil {
		return err
	}
	setup := pitch.Setup{Dir: filepath.Join(modelsDir(c.Dir), m.DirName()), Repo: m.Source(), Log: lw}
	var d *pitch.Detector
	t0 := time.Now()
	ort, builtin := cpuRoute(m, c.Device, c.CPU, c.Threads, logf)
	c.CPU = c.CPU || builtin
	if ort {
		model, err := pitch.Fetch(ctx, setup)
		if err != nil {
			return err
		}
		if d, err = pitch.OpenCPU(model, c.Threads); err != nil {
			return err
		}
		logf("loaded %s in %.1f s, on the CPU (ONNX Runtime, %d threads)", m.Name, time.Since(t0).Seconds(), max(1, c.Threads))
	} else {
		backend := llm.BackendCoreML // what the CPU interpreter runs
		var dev *hwnpu.Device
		if !c.CPU {
			if dev, backend, err = openAudioDevice(ctx, cacheDir(c.Cache), "--device cpu runs it on the CPU"); err != nil {
				return err
			}
			defer dev.Close()
		}
		setup.Backend = backend
		model, program, err := pitch.Prepare(ctx, setup)
		if err != nil {
			return err
		}
		t0 = time.Now()
		if d, err = pitch.Open(ctx, dev, model, program); err != nil {
			return err
		}
		logf("loaded %s in %.1f s, %s", m.Name, time.Since(t0).Seconds(), placement(c.CPU, d.OnNPU()))
	}
	defer d.Close()
	console.Running()
	for _, p := range c.Files {
		raw, err := os.ReadFile(p)
		if err != nil {
			return err
		}
		x, err := lfm3audio.ReadWAV(raw) // mono at 16 kHz, SwiftF0's rate too
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		start := time.Now()
		r, err := d.Pitch(ctx, x, c.FMin, c.FMax)
		if err != nil {
			return fmt.Errorf("%s: %w", p, err)
		}
		el := time.Since(start)
		if err := writePitch(p, r, c.Threshold, c.JSON, len(c.Files) > 1); err != nil {
			return err
		}
		secs := float64(len(x)) / pitch.SampleRate
		logf("%s: %.1f s of audio, %d frames in %.3f s (%.0fx real time)", p, secs, len(r.PitchHz), el.Seconds(), secs/el.Seconds())
	}
	return nil
}

func writePitch(path string, r pitch.Result, threshold float64, asJSON, named bool) error {
	period := float64(pitch.Hop) / pitch.SampleRate
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
		for i := range r.PitchHz {
			out.Frames = append(out.Frames, frame{float64(i) * period, r.PitchHz[i], r.Confidence[i], r.Confidence[i] >= threshold, r.LoudnessDB[i]})
		}
		return json.NewEncoder(os.Stdout).Encode(out)
	}
	if named {
		fmt.Printf("# %s\n", path)
	}
	fmt.Println("timestamp,pitch_hz,confidence,voiced,loudness_db")
	for i := range r.PitchHz {
		fmt.Printf("%.4f,%.2f,%.4f,%v,%.1f\n", float64(i)*period, r.PitchHz[i], r.Confidence[i], r.Confidence[i] >= threshold, r.LoudnessDB[i])
	}
	return nil
}
