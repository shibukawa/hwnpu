package ai

import (
	"context"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm/whisper"
	"github.com/shibukawa/hwnpu/internal/transcribe"
)

// TranscribeRate is the samples a second Transcriber reads.
const TranscribeRate = whisper.SampleRate

// TranscribeOptions are how speech is transcribed.
type TranscribeOptions struct {
	// Language is the speech's language code (ja, en, ...); "" finds it in each 30 seconds.
	Language string
	// Translate writes English text whatever the speech's language (Whisper's translate task).
	Translate bool
	// MaxTokens is the most tokens of text per 30 seconds (0: as many as the model holds).
	MaxTokens int
}

// Transcript is the text of some speech.
type Transcript struct {
	Text string
	// Lang is the language found (or given) in the first 30 seconds.
	Lang string
	// Segments are the text's pieces with their times; one for up to 30 seconds of speech.
	Segments []Segment
}

// Segment is a piece of a transcript.
type Segment struct {
	Start, End time.Duration
	Text       string
}

// Transcriber turns speech into text (Whisper). It runs one request at a time; others wait.
type Transcriber struct {
	mu    sync.Mutex
	t     *transcribe.Transcriber
	dev   *hwnpu.Device
	model string
	cpu   bool
}

// Model is the name of the model in the definitions.
func (t *Transcriber) Model() string { return t.model }

// OnNPU reports whether the programs were verified to run on the NPU.
func (t *Transcriber) OnNPU() bool { return !t.cpu && t.t.OnNPU() }

// Transcribe turns speech (mono, at TranscribeRate, in [-1, 1]) into text. Speech longer than 30
// seconds is read in windows, each starting where the last one's last whole segment ended.
func (t *Transcriber) Transcribe(ctx context.Context, pcm []float32, o TranscribeOptions) (Transcript, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	res, err := t.t.Transcribe(ctx, pcm, transcribe.Options{Language: o.Language, Translate: o.Translate, MaxTokens: o.MaxTokens})
	if err != nil {
		return Transcript{}, err
	}
	out := Transcript{Text: strings.TrimSpace(res.Text)}
	for _, w := range res.Windows {
		if out.Lang == "" {
			out.Lang = w.Language
		}
		for _, s := range w.Segments {
			out.Segments = append(out.Segments, Segment{Start: w.Start + s.Start, End: w.Start + s.End, Text: strings.TrimSpace(t.t.SegmentText(s))})
		}
	}
	if o.Translate {
		out.Lang = "en"
	}
	return out, nil
}

// TranscribeStream transcribes a stream of speech (VAD.Detect's) segment by segment: the
// chunks up to each final one are read together once it comes, and their text is a final
// chunk with their times. Captions while someone speaks, provisional chunks read again as the
// speech grows, are to come (docs/app-api-tasks.md B5).
func (t *Transcriber) TranscribeStream(ctx context.Context, s Stream[[]float32], o TranscribeOptions) Stream[Text] {
	return func(yield func(Chunk[Text], error) bool) {
		var pcm []float32
		var start time.Duration
		flush := func(end time.Duration) bool {
			if len(pcm) == 0 {
				return true
			}
			tr, err := t.Transcribe(ctx, pcm, o)
			pcm = pcm[:0]
			if err != nil {
				yield(Chunk[Text]{}, err)
				return false
			}
			if tr.Text == "" {
				return true
			}
			return yield(Chunk[Text]{Data: Text{Text: tr.Text, Lang: tr.Lang}, Start: start, Len: end - start, Final: true}, nil)
		}
		var end time.Duration
		for c, err := range s {
			if err != nil {
				yield(Chunk[Text]{}, err)
				return
			}
			if len(pcm) == 0 {
				start = c.Start
			}
			pcm = append(pcm, c.Data...)
			end = c.Start + c.Len
			if c.Final && !flush(end) {
				return
			}
		}
		flush(end)
	}
}

// Close unloads the model.
func (t *Transcriber) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	err := t.t.Close()
	if t.dev != nil {
		if e := t.dev.Close(); err == nil {
			err = e
		}
	}
	return err
}
