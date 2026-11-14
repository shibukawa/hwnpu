package ai

import (
	"context"
	"errors"
	"math/rand/v2"
	"strings"
	"sync"

	"github.com/shibukawa/hwnpu/internal/speech"
)

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
		return Audio{}, errors.New("ai: nothing to speak")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	pcm, _, err := s.s.Speak(ctx, o.request(text))
	if err != nil {
		return Audio{}, err
	}
	return Audio{Samples: pcm, Rate: s.s.SampleRate()}, nil
}

// SpeakStream speaks each final chunk of a stream of text (Sentences, or a translation's) as it
// comes: a final chunk of audio each, timed on the output's own clock (its samples from the
// start), with one seed for the whole stream. Provisional chunks are left out. Speech that comes
// out as it is made, before the text's whole audio is, is to come (docs/app-api-tasks.md B6).
func (s *Speaker) SpeakStream(ctx context.Context, in Stream[Text], o SpeakOptions) Stream[[]float32] {
	return func(yield func(Chunk[[]float32], error) bool) {
		if o.Seed == 0 {
			o.Seed = rand.Uint64()
		}
		var pos int64
		for c, err := range in {
			if err != nil {
				yield(Chunk[[]float32]{}, err)
				return
			}
			if !c.Final || strings.TrimSpace(c.Data.Text) == "" {
				continue
			}
			oo := o
			if oo.Language == "" {
				oo.Language = c.Data.Lang
			}
			a, err := s.Speak(ctx, c.Data.Text, oo)
			if err != nil {
				yield(Chunk[[]float32]{}, err)
				return
			}
			out := Chunk[[]float32]{Data: a.Samples, Start: samples(pos, a.Rate), Final: true}
			pos += int64(len(a.Samples))
			out.Len = samples(pos, a.Rate) - out.Start
			if !yield(out, nil) {
				return
			}
		}
	}
}

// Close unloads the model.
func (s *Speaker) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.s.Close()
}
