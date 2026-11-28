package speech

import (
	"context"
	"errors"
	"io"
	"math/rand/v2"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/speech/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
)

// Speak is the use of a model of text to speech (LFM3-Audio, Supertonic 3) for an app: a model,
// a family or a use of the definitions ("speech", "supertonic").
func Speak(name string) ai.Use[*Speaker] {
	return ai.Define[*Speaker](speakRole, name)
}

// The settings and programs a speech model's definition may give (ai.Definition):
//
//	settings: format     the text model's weight format (default: the model's, 8 bit tables)
//	programs: iron       on AMD's NPU, an export of tools/amd/iron_lfm3speech.py (its directory)
//	          text       the text model converted by hand (-cache and -hidden-out 1), its packages
//	                     of different -seq; or iron-text, on AMD's NPU its export of
//	                     iron_export_lm.py lfm3tts
//	          depth      the depthformer (-arch lfm3-depthformer)
//	          codec      the codec decoder (-arch lfm3-codec)
//	          checkpoint the checkpoint's directory of those (config.json, model.safetensors,
//	                     tokenizer.json, audio_tokenizer/config.json)
//
// A model of device cpu runs on hwnpu's CPU interpreter (slow).
// speakRole's Prepare and Load are the engine's (ai/speech/engine).
var speakRole = &airole.Role{
	Name:   "speak",
	Remote: func(h *airpc.Handle) io.Closer { return &Speaker{p: airpc.RemoteProxy[speakerImpl](h)} },
	Kinds:  []catalog.Kind{catalog.KindSpeech},
	// 1: Supertonic's programs of its text's graphs (durations, encoders), with no ONNX Runtime
	// 2: Supertonic on AMD's NPU (amd: supertonic.PrepareAMD's export)
	// 3: its text's graphs in the export too (no programs)
	// 4: the tokenizer as its binary file (chat.PrepareTokenizer) in the files, as "tokenizer": a
	// store prepared before named none, and Load then read the checkpoint's tokenizer.json, which a
	// checkpoint fetched since is without
	Revision: 4,
}

// SpeakOptions are how a text is spoken (LFM3-Audio's settings).
type SpeakOptions struct {
	// Language is a code of the model card (en, en-gb, ja, zh, ...) or a language's name;
	// "" is a chunk's Text.Lang in SpeakStream, else en.
	Language string
	// Voice is female or male ("": female).
	Voice string
	// Seed makes the speech repeatable (0: a new one each call, told in Audio.Stats; SpeakStream
	// draws one for the whole stream, so that its voice stays the same).
	Seed uint64
	// MaxFrames is the most frames of 80 ms (0: until the model ends the speech).
	MaxFrames int
	// Speed is how fast the text is said, as a multiple of the model's own pace (0: 1), by a
	// model that paces its speech (Paces: Supertonic, from 0.67 to 1.9); the others leave it.
	Speed float64
}

// Paces reports whether the model says a text at SpeakOptions.Speed itself (Supertonic); the
// speech of another is to be stretched for it.
func (s *Speaker) Paces() bool { return airpc.ProxyValue(s.p, "Paces", speakerImpl.Paces) }

// Audio is mono audio.
type Audio struct {
	Samples []float32 // in [-1, 1]
	Rate    int       // samples a second
	// Stats are what speaking it took (Speak's; the zero value elsewhere).
	Stats SpeakStats
}

// SpeakStats are what speaking a text took.
type SpeakStats struct {
	// Frames are the audio frames of 80 ms made, and Seed the seed they were drawn with (the one
	// drawn when SpeakOptions had none: given again, it speaks the same).
	Frames int
	Seed   uint64
	// Took is the whole time; Text, Depth and Codec the time in the text model (the prompt and
	// the frames' steps), the depthformer and the codec decoder.
	Took, Text, Depth, Codec time.Duration
}

// Speaker speaks texts (LFM3-Audio). It runs one request at a time; others wait.
type Speaker struct {
	p airpc.Proxy[speakerImpl]
}

// speakerImpl is a speech model of the engine in this process.
type speakerImpl interface {
	Model() string
	Rate() int
	Paces() bool
	Stages() (text, depth, codec string)
	OnNPU() bool
	// Speak speaks a text that is not empty.
	Speak(ctx context.Context, text string, o SpeakOptions) (Audio, error)
	Close() error
}

// Model is the name of the model in the definitions.
func (s *Speaker) Model() string { return airpc.ProxyValue(s.p, "Model", speakerImpl.Model) }

// Rate is the samples a second of the speech (LFM3-Audio's 24000, Supertonic's 44100).
func (s *Speaker) Rate() int { return airpc.ProxyValue(s.p, "Rate", speakerImpl.Rate) }

// Stages names what SpeakStats' Text, Depth and Codec times are of: LFM3-Audio's text model,
// depthformer and codec decoder, or Supertonic's text graphs, flow and vocoder.
func (s *Speaker) Stages() (text, depth, codec string) {
	if h := s.p.Remote(); h != nil {
		out, err := h.Call(context.Background(), "Stages")
		if err != nil || len(out) != 3 {
			return "", "", ""
		}
		return out[0].String(), out[1].String(), out[2].String()
	}
	return s.p.Local().Stages()
}

// OnNPU reports whether the speech is made on the NPU (Supertonic's flow and vocoder; false on
// the CPU).
func (s *Speaker) OnNPU() bool { return airpc.ProxyValue(s.p, "OnNPU", speakerImpl.OnNPU) }

// Speak speaks a text.
func (s *Speaker) Speak(ctx context.Context, text string, o SpeakOptions) (Audio, error) {
	if strings.TrimSpace(text) == "" {
		return Audio{}, errors.New("speech: nothing to speak")
	}
	return airpc.ProxyCall1(s.p, ctx, "Speak", func(l speakerImpl) (Audio, error) {
		return l.Speak(ctx, text, o)
	}, text, o)
}

// SpeakStream speaks each final chunk of a stream of text (flow.Sentences, or a translation's)
// as it comes: a final chunk of audio each, timed on the output's own clock (its samples from
// the start, so that flow.Pace hands them to a player one after another), with one seed for the
// whole stream. Provisional chunks are left out. Speech that comes out as it is made, before
// the text's whole audio is, is to come.
func (s *Speaker) SpeakStream(ctx context.Context, in flow.Stream[flow.Text], o SpeakOptions) flow.Stream[[]float32] {
	return s.speakStream(ctx, in, func() SpeakOptions { return o })
}

// SpeakNode is the node of a speaker in a network or a call (flow, ai.App.From, ai.NewCall): texts in, their
// speech out, as SpeakStream gives it. Its settings may be changed while the network runs
// (Update): the next text is spoken with them.
func SpeakNode(u ai.Use[*Speaker], o SpeakOptions) *ai.ModelNode[flow.Text, []float32, SpeakOptions] {
	return ai.NewModelNode(u, o, func(ctx context.Context, s *Speaker, in flow.Stream[flow.Text], options func() SpeakOptions) flow.Stream[[]float32] {
		return s.speakStream(ctx, in, options)
	})
}

// speakStream is SpeakStream with the options asked for at each text.
func (s *Speaker) speakStream(ctx context.Context, in flow.Stream[flow.Text], options func() SpeakOptions) flow.Stream[[]float32] {
	return func(yield func(flow.Chunk[[]float32], error) bool) {
		seed := rand.Uint64() // one for the stream when the options have none
		pos := 0
		for c, err := range in {
			if err != nil {
				yield(flow.Chunk[[]float32]{}, err)
				return
			}
			if !c.Final || strings.TrimSpace(c.Data.Text) == "" {
				continue
			}
			oo := options()
			if oo.Seed == 0 {
				oo.Seed = seed
			}
			if oo.Language == "" {
				oo.Language = c.Data.Lang
			}
			a, err := s.Speak(ctx, c.Data.Text, oo)
			if err != nil {
				yield(flow.Chunk[[]float32]{}, err)
				return
			}
			out := flow.Chunk[[]float32]{Data: a.Samples, Start: flow.Samples(pos, a.Rate), Final: true}
			pos += len(a.Samples)
			out.Len = flow.Samples(pos, a.Rate) - out.Start
			if !yield(out, nil) {
				return
			}
		}
	}
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (s *Speaker) Close() error { return s.p.Close() }

func init() {
	airole.Register(speakRole)
	bridge.Roles[speakRole.Name] = speakRole
	bridge.Wrap[speakRole.Name] = func(l any) io.Closer { return &Speaker{p: airpc.LocalProxy(l.(speakerImpl))} }
}
