package sound

import (
	"context"
	"io"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/sound/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/soundwire"
)

// The music's audio: 48 kHz stereo, its samples interleaved (left, right), 25 frames of the
// model a second.
const (
	MusicRate      = soundwire.MusicRate
	MusicFrameRate = soundwire.MusicFrameRate
	// MusicGain is what magenta-rt multiplies the samples by when it writes them as 16 bit
	// integers: hwnpu-audio writes and plays the music at it.
	MusicGain = 0.5
	// StyleCodes are the codes of a style (MusicCoCa's 12).
	StyleCodes = soundwire.StyleCodes
)

// Music is the use of a model that makes music (Magenta RealTime 2) for an app: a model, a
// family or a use of the definitions ("music", "magenta-realtime-2").
func Music(name string) ai.Use[*MusicGenerator] {
	return ai.Define[*MusicGenerator](musicRole, name)
}

// The settings a music model's definition may give (ai.Definition):
//
//	settings: format   the language model's weight format: lut8 (the default), lut6 or fp16
//
// A model of device cpu runs on hwnpu's CPU interpreter (slow). On AMD's NPU SpectroStream's
// decoder runs as the graph of tools/amd/iron_magenta.py (exported on the first preparation),
// the language model and MusicCoCa on the CPU interpreter.
//
// Every part is prepared: MusicCoCa's text and audio sides (styles from texts and from audio)
// and SpectroStream's encoder (continuing a clip), loaded at their first use. The role's
// Prepare and Load are the engine's (ai/sound/engine).
var musicRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &MusicGenerator{remote: h} },
	Name:   "music",
	Kinds:  []catalog.Kind{catalog.KindMusic},
}

// MusicGenerator makes music (Magenta RealTime 2): styles from texts and audio (Style), then
// the music of a style, as it is made (Generate). It runs one request at a time; others wait.
type MusicGenerator struct {
	remote *airpc.Handle      // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
	local  musicGeneratorImpl // the engine's (ai/sound/engine), when it is here
}

// musicGeneratorImpl is a music model of the engine in this process.
type musicGeneratorImpl interface {
	Model() string
	OnCPU() bool
	Where() string
	Style(ctx context.Context, parts ...Style) (StyleResult, error)
	Generate(ctx context.Context, o MusicOptions) (flow.Stream[[]float32], *MusicStats)
	Close() error
}

// Model is the name of the model in the definitions.
func (m *MusicGenerator) Model() string {
	if m.remote != nil {
		return airpc.Value[string](m.remote, "Model")
	}
	return m.local.Model()
}

// OnCPU reports whether the model runs on hwnpu's CPU interpreter (its definition's device cpu).
func (m *MusicGenerator) OnCPU() bool {
	if m.remote != nil {
		return airpc.Value[bool](m.remote, "OnCPU")
	}
	return m.local.OnCPU()
}

// Where says where the programs run, for a log: "all on the NPU", "SpectroStream's decoder on
// the NPU, the language model on the CPU" (AMD's NPU), "on the CPU interpreter", ...
func (m *MusicGenerator) Where() string {
	if m.remote != nil {
		return airpc.Value[string](m.remote, "Where")
	}
	return m.local.Where()
}

// Clip is audio of any rate and channels, as a WAV file holds it (DecodeWAV).
type Clip struct {
	// Channels are the samples of each channel, in [-1, 1].
	Channels [][]float32
	Rate     int
}

// Seconds is how long the clip lasts.
func (c Clip) Seconds() float64 {
	if len(c.Channels) == 0 || c.Rate <= 0 {
		return 0
	}
	return float64(len(c.Channels[0])) / float64(c.Rate)
}

// DecodeWAV decodes a WAV file (PCM of 8, 16, 24 or 32 bits, or 32 bit float) as it is.
func DecodeWAV(data []byte) (Clip, error) {
	ch, rate, err := soundwire.DecodeWAV(data)
	return Clip{Channels: ch, Rate: rate}, err
}

// Style is a part of a style: a text ("disco funk") or audio in that style, and its weight
// among the parts.
type Style struct {
	Text  string
	Audio Clip
	// Weight is the part's share of the mix (0: 1).
	Weight float64
}

// StyleResult is a style made: its codes, which MusicOptions.Codes takes.
type StyleResult struct {
	Codes []int
	// Took is how long making it took, the first time with loading MusicCoCa.
	Took time.Duration
}

// Style is the style of the parts mixed by their weights (MusicCoCa's embeddings, averaged,
// then its 12 codes). MusicCoCa is loaded at the first call and kept.
func (m *MusicGenerator) Style(ctx context.Context, parts ...Style) (StyleResult, error) {
	if m.remote != nil {
		return airpc.Call1[StyleResult](m.remote, ctx, "Style", parts)
	}
	return m.local.Style(ctx, parts...)
}

// Drums says whether the music has drums.
type Drums int

const (
	DrumsAuto Drums = iota // the model decides
	DrumsOn
	DrumsOff
)

// Guidance are the classifier-free guidance scales (-1 to 7): how strongly the style, the
// notes and the drums setting guide the music.
type Guidance struct{ Style, Notes, Drums float64 }

// DefaultGuidance is magenta-rt's: 3, 5 and 1.
var DefaultGuidance = Guidance{Style: soundwire.GuidanceStyle, Notes: soundwire.GuidanceNotes, Drums: soundwire.GuidanceDrums}

// MusicOptions are a piece of music's settings.
type MusicOptions struct {
	// Codes are the style's (StyleResult.Codes, StyleCodes of them from 0 to 1023); nil for none,
	// when the clip alone leads (Continue).
	Codes []int
	// Duration is how long the music made lasts; 0 is the notes' length and half a second with
	// Notes, else 10 seconds.
	Duration time.Duration
	// Notes are notes the music plays (ReadMIDI; the drums' channel 10 is left out). Onsets
	// tells their starts (otherwise the model decides where a held note starts again);
	// NoteWidth says that the pitches within that many semitones of a sounding note are silent
	// (127: all).
	Notes     []MIDINote
	Onsets    bool
	NoteWidth int
	// Continue is a clip the music goes on from (its last 20 seconds count): the stream gives
	// the clip, then what follows it; with NewOnly, what follows alone.
	Continue Clip
	NewOnly  bool
	Drums    Drums
	// Temperature (0: 1.3) and TopK (0: 40) are how the tokens are sampled.
	Temperature float64
	TopK        int
	// Guidance is nil for DefaultGuidance.
	Guidance *Guidance
	// Fade fades the last of the music out (0: it stops where the generation does).
	Fade time.Duration
	// Seed makes the music repeatable (0: a new one, told in MusicStats.Seed).
	Seed uint64
	// Progress, when not nil, is told of the clip being read and of each second generated, in
	// the goroutine that reads the stream.
	Progress func(MusicProgress)
}

// MusicStage is what a piece's generation is doing.
type MusicStage int

const (
	// MusicReadingClip: SpectroStream's encoder reads the clip to continue (ClipSeconds of it,
	// the last 20 seconds at most), and the language model goes over its frames.
	MusicReadingClip MusicStage = iota
	// MusicClipRead: the clip is read (MusicStats' ClipFrames and Clip).
	MusicClipRead
	// MusicGenerating: Frames of the Of frames to generate are made, after each second.
	MusicGenerating
)

// MusicProgress is how far a piece's generation is.
type MusicProgress struct {
	Stage       MusicStage
	ClipSeconds float64
	Frames, Of  int
	// Elapsed is the time since the frames began.
	Elapsed time.Duration
}

// MusicStats are what making a piece took. Seed is set when Generate returns; the rest as the
// stream goes, whole when it ends.
type MusicStats struct {
	// Seed is the seed the tokens were sampled with (the one drawn when MusicOptions had none:
	// given again, it makes the same music).
	Seed uint64
	// Frames are the frames generated (25 a second, one more than the duration's: the decoder's
	// lookahead holds the last one back); Samples the stereo samples the stream gave.
	Frames, Samples int
	// ClipFrames are the clip's frames read and Clip how long reading them took.
	ClipFrames int
	Clip       time.Duration
	// Took is the time from the first frame to the end of the music; Decode the decoder's
	// part of it (it runs beside the language model).
	Took, Decode time.Duration
}

// Seconds is how long the music generated lasts (the clip left out).
func (s MusicStats) Seconds() float64 { return float64(s.Frames) / MusicFrameRate }

// RealTime is how many times real time the music was made.
func (s MusicStats) RealTime() float64 { return s.Seconds() / s.Took.Seconds() }

// Generate makes a piece of music and gives it as it is made: 48 kHz stereo (interleaved) a
// second at a time, the stream's clock in samples, the last chunk final. It is faster than real
// time on the NPU; a player paces it (flow.Pace) and plays it at MusicGain. The stats are those
// of the piece, whole when the stream ends. The stream generates once, when it is read.
func (m *MusicGenerator) Generate(ctx context.Context, o MusicOptions) (flow.Stream[[]float32], *MusicStats) {
	if m.remote != nil {
		// the stats are whole when the stream ends, as here (airpc sends them with its end)
		s, st, err := airpc.Call2[flow.Stream[[]float32], *MusicStats](m.remote, ctx, "Generate", o)
		if err != nil {
			return func(yield func(flow.Chunk[[]float32], error) bool) { yield(flow.Chunk[[]float32]{}, err) }, &MusicStats{Seed: o.Seed}
		}
		return s, st
	}
	return m.local.Generate(ctx, o)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (m *MusicGenerator) Close() error {
	if m.remote != nil {
		return m.remote.Release() // the Runtime in hwnpud unloads it
	}
	return m.local.Close()
}

func init() {
	airole.Register(musicRole)
	bridge.Roles[musicRole.Name] = musicRole
	bridge.Wrap[musicRole.Name] = func(l any) io.Closer { return &MusicGenerator{local: l.(musicGeneratorImpl)} }
}
