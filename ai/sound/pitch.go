package sound

import (
	"context"
	"io"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/pitch"
)

// The pitch's audio and frames: 16 kHz mono in, a frame every 16 ms out.
const (
	PitchRate = pitch.SampleRate
	// PitchPeriod is the time from a frame to the next (256 samples, 16 ms).
	PitchPeriod = time.Duration(pitch.Hop) * time.Second / pitch.SampleRate
	// Voiced is the confidence at and over which a frame is voiced, as SwiftF0 counts it.
	Voiced = 0.5
)

// Pitch is the use of a model that tells the pitch of one voice or instrument (SwiftF0) for an
// app: a model, a family or a use of the definitions ("pitch", "swiftf0").
func Pitch(name string) ai.Use[*PitchDetector] {
	return ai.Define[*PitchDetector](pitchRole, name)
}

// pitchRole's models run as the converted models do (converted.go): the front end (the three
// spectrograms) and the decoding of the logits on the host, the CNN on the NPU or the CPU.
var pitchRole = &airole.Role{
	Name:  "pitch",
	Kinds: []chat.Kind{chat.KindPitch},
	Prepare: func(e airole.PrepareEnv) (airole.Files, error) {
		return convertedPrepare(e, func(backend string) (string, string, error) {
			return pitch.Prepare(e.Ctx, pitch.Setup{Dir: e.Dir, Repo: e.Model.Source(), Backend: backend, Log: e.Log})
		})
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		r, err := convertedRoute(e.Def, e.Model, e.NPU)
		if err != nil {
			return nil, err
		}
		c, dev, err := convertedLoad(ctx, f, e, r, convertedOpener{
			cpu: func(model string, threads int) (io.Closer, error) { return pitch.OpenCPU(model, threads) },
			npu: func(ctx context.Context, dev *hwnpu.Device, model, program string) (io.Closer, error) {
				return pitch.Open(ctx, dev, model, program)
			}})
		if err != nil {
			return nil, err
		}
		return &PitchDetector{d: c.(*pitch.Detector), dev: dev, r: r, model: e.Model}, nil
	},
}

// PitchDetector tells the pitch (F0) of a voice or an instrument (SwiftF0), and how sure it is
// that there is one, every 16 ms. It runs one request at a time; others wait.
type PitchDetector struct {
	mu    sync.Mutex
	d     *pitch.Detector
	dev   *hwnpu.Device
	r     route
	model string
}

// Model is the name of the model in the definitions.
func (d *PitchDetector) Model() string { return d.model }

// Device is where the model runs: npu or cpu.
func (d *PitchDetector) Device() string { return d.r.device }

// Where says where the model runs, for a log: "all on the NPU", "on the CPU (ONNX Runtime, 1
// threads)", "on the CPU interpreter", ...
func (d *PitchDetector) Where() string {
	if w := d.r.Where(); w != "" {
		return w
	}
	return onNPUWhere(d.d.OnNPU())
}

// onNPUWhere is how a log says where a model of the NPU runs.
func onNPUWhere(onNPU bool) string {
	if onNPU {
		return "all on the NPU"
	}
	return "NOT all on the NPU"
}

// PitchOptions are how the pitch is told.
type PitchOptions struct {
	// FMin and FMax are the lowest and the highest pitch considered (0: 46.875 and 2093.75 Hz).
	FMin, FMax float64
}

// PitchFrame is a frame's pitch.
type PitchFrame struct {
	// Time is where the frame is, in seconds; Hz the pitch; Confidence how sure the model is
	// that there is one (voiced at Voiced and over); LoudnessDB the level of the 32 ms centered
	// on the frame.
	Time, Hz, Confidence, LoudnessDB float64
}

// PitchTrack is the pitch of some audio.
type PitchTrack struct {
	Frames []PitchFrame
	// Seconds is the audio's length and Took how long telling its pitch took.
	Seconds float64
	Took    time.Duration
}

// Detect tells the pitch of audio (mono, at PitchRate, in [-1, 1]).
func (d *PitchDetector) Detect(ctx context.Context, pcm []float32, o PitchOptions) (PitchTrack, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	t0 := time.Now()
	r, err := d.d.Pitch(ctx, pcm, o.FMin, o.FMax)
	if err != nil {
		return PitchTrack{}, err
	}
	t := PitchTrack{Seconds: float64(len(pcm)) / pitch.SampleRate, Took: time.Since(t0)}
	period := float64(pitch.Hop) / pitch.SampleRate
	for i := range r.PitchHz {
		t.Frames = append(t.Frames, PitchFrame{Time: float64(i) * period, Hz: r.PitchHz[i], Confidence: r.Confidence[i], LoudnessDB: r.LoudnessDB[i]})
	}
	return t, nil
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (d *PitchDetector) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	err := d.d.Close()
	if d.dev != nil {
		d.dev.Close()
		d.dev = nil
	}
	return err
}
