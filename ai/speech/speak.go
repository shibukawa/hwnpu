package speech

import (
	"context"
	"errors"
	"io"
	"math/rand/v2"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/speech"
)

// Speak is the use of a model of text to speech (LFM3-Audio) for an app: a model, a family or a
// use of the definitions ("speech").
func Speak(name string) ai.Use[*Speaker] {
	return ai.Define[*Speaker](speakRole, name)
}

var speakRole = &airole.Role{
	Name:  "speak",
	Kinds: []chat.Kind{chat.KindSpeech},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		var c speech.Config
		if e.NPU.AMD() {
			// AMD's NPU runs the export of tools/amd/iron_lfm3speech.py
			if c, err = speech.PrepareAMD(e.Ctx, e.Model, e.Log); err != nil {
				return f, err
			}
			iron.ShareExport(c.Dir, logf(e.Log))
		} else if c, err = speech.Prepare(e.Ctx, speech.Setup{Dir: e.Dir, Repo: e.Model.Source(), Backend: e.NPU.Format, Log: e.Log}); err != nil {
			return f, err
		}
		f.Set("text", c.Text...)
		f.Set("iron-text", c.IronText)
		f.Set("depth", c.Depth)
		f.Set("codec", c.Codec)
		f.Set("dir", c.Dir)
		f.Set("weights", c.Weights)
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		s, err := speech.Load(ctx, speech.Config{Text: f.Paths["text"], IronText: f.Path("iron-text"), Depth: f.Path("depth"),
			Codec: f.Path("codec"), Dir: f.Path("dir"), Weights: f.Path("weights"),
			CacheDir: e.Cache, Device: e.NPU.Device, Log: e.Log})
		if err != nil {
			return nil, err
		}
		return &Speaker{s: s, model: e.Model}, nil
	},
}

// SpeakOptions are how a text is spoken (LFM3-Audio's settings).
type SpeakOptions struct {
	// Language is a code of the model card (en, en-gb, ja, zh, ...) or a language's name;
	// "" is a chunk's Text.Lang in SpeakStream, else en.
	Language string
	// Voice is female or male ("": female).
	Voice string
	// Seed makes the speech repeatable (0: a new one each call; SpeakStream draws one for the
	// whole stream, so that its voice stays the same).
	Seed uint64
	// MaxFrames is the most frames of 80 ms (0: until the model ends the speech).
	MaxFrames int
}

// Audio is mono audio.
type Audio struct {
	Samples []float32 // in [-1, 1]
	Rate    int       // samples a second
}

// Speaker speaks texts (LFM3-Audio). It runs one request at a time; others wait.
type Speaker struct {
	mu    sync.Mutex
	s     *speech.Synth
	model string
}

// Model is the name of the model in the definitions.
func (s *Speaker) Model() string { return s.model }

// Rate is the samples a second of the speech (24000).
func (s *Speaker) Rate() int { return s.s.SampleRate() }

func (o SpeakOptions) request(text string) speech.Request {
	lang, voice := o.Language, o.Voice
	if lang == "" {
		lang = "en"
	}
	if voice == "" {
		voice = "female"
	}
	return speech.Request{Text: text, Language: lang, Voice: voice, Seed: o.Seed, MaxFrames: o.MaxFrames}
}

// Speak speaks a text.
func (s *Speaker) Speak(ctx context.Context, text string, o SpeakOptions) (Audio, error) {
	if strings.TrimSpace(text) == "" {
		return Audio{}, errors.New("speech: nothing to speak")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	pcm, _, err := s.s.Speak(ctx, o.request(text))
	if err != nil {
		return Audio{}, err
	}
	return Audio{Samples: pcm, Rate: s.s.SampleRate()}, nil
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
func (s *Speaker) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.s.Close()
}
