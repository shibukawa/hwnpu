package sound

import (
	"context"
	"io"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/sound/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/soundwire"
)

// The pitch's audio and frames: 16 kHz mono in, a frame every 16 ms out.
const (
	PitchRate = soundwire.PitchRate
	// PitchPeriod is the time from a frame to the next (256 samples, 16 ms).
	PitchPeriod = time.Duration(soundwire.PitchHop) * time.Second / soundwire.PitchRate
	// Voiced is the confidence at and over which a frame is voiced, as SwiftF0 counts it.
	Voiced = 0.5
)

// Pitch is the use of a model that tells the pitch of one voice or instrument (SwiftF0) for an
// app: a model, a family or a use of the definitions ("pitch", "swiftf0").
func Pitch(name string) ai.Use[*PitchDetector] {
	return ai.Define[*PitchDetector](pitchRole, name)
}

// pitchRole's Prepare and Load are the engine's (ai/sound/engine): the front end (the three
// spectrograms) and the decoding of the logits on the host, the CNN on the NPU or the CPU.
var pitchRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &PitchDetector{remote: h} },
	Name:   "pitch",
	Kinds:  []catalog.Kind{catalog.KindPitch},
}

// PitchDetector tells the pitch (F0) of a voice or an instrument (SwiftF0), and how sure it is
// that there is one, every 16 ms. It runs one request at a time; others wait.
type PitchDetector struct {
	remote *airpc.Handle     // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
	local  pitchDetectorImpl // the engine's (ai/sound/engine), when it is here
}

// pitchDetectorImpl is a pitch model of the engine in this process.
type pitchDetectorImpl interface {
	Model() string
	Device() string
	Where() string
	Detect(ctx context.Context, pcm []float32, o PitchOptions) (PitchTrack, error)
	Close() error
}

// Model is the name of the model in the definitions.
func (d *PitchDetector) Model() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Model")
	}
	return d.local.Model()
}

// Device is where the model runs: npu or cpu.
func (d *PitchDetector) Device() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Device")
	}
	return d.local.Device()
}

// Where says where the model runs, for a log: "all on the NPU", "on the CPU (ONNX Runtime, 1
// threads)", "on the CPU interpreter", ...
func (d *PitchDetector) Where() string {
	if d.remote != nil {
		return airpc.Value[string](d.remote, "Where")
	}
	return d.local.Where()
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
	if d.remote != nil {
		return airpc.Call1[PitchTrack](d.remote, ctx, "Detect", pcm, o)
	}
	return d.local.Detect(ctx, pcm, o)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (d *PitchDetector) Close() error {
	if d.remote != nil {
		return d.remote.Release() // the Runtime in hwnpud unloads it
	}
	return d.local.Close()
}

func init() {
	airole.Register(pitchRole)
	bridge.Roles[pitchRole.Name] = pitchRole
	bridge.Wrap[pitchRole.Name] = func(l any) io.Closer { return &PitchDetector{local: l.(pitchDetectorImpl)} }
}
