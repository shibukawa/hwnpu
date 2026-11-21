package speech

import (
	"context"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/llm/whisper"
	"github.com/shibukawa/hwnpu/internal/transcribe"
)

// TranscribeRate is the samples a second Transcriber reads.
const TranscribeRate = whisper.SampleRate

// Transcribe is the use of a model of speech to text (Whisper) for an app: a model, a family or
// a use of the definitions ("transcribe", "whisper:large-v3-turbo").
func Transcribe(name string) ai.Use[*Transcriber] {
	return ai.Define[*Transcriber](transcribeRole, name)
}

var transcribeRole = &airole.Role{
	Name:  "transcribe",
	Kinds: []chat.Kind{chat.KindTranscribe},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		if e.NPU.AMD() {
			// AMD's NPU runs the graphs of tools/amd/iron_whisper.py
			dir, err := transcribe.PrepareAMD(e.Ctx, e.Model, e.Log)
			if err != nil {
				return f, err
			}
			iron.ShareExport(dir, logf(e.Log))
			f.Set("iron", dir)
			return f, nil
		}
		c, err := transcribe.Prepare(e.Ctx, transcribe.Setup{Dir: e.Dir, Repo: e.Model.Source(), Backend: e.NPU.Format, Log: e.Log})
		if err != nil {
			return f, err
		}
		f.Set("dir", c.Dir)
		f.Set("encoder", c.Encoder)
		f.Set("cross", c.Cross)
		f.Set("decoder", c.Decoder)
		f.Set("weights", c.Weights)
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		var (
			dev *hwnpu.Device
			t   *transcribe.Transcriber
			err error
		)
		if dir := f.Path("iron"); dir != "" {
			if dev, err = hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD)); err != nil {
				return nil, err
			}
			t, err = transcribe.LoadIron(ctx, dev, dir)
		} else {
			if dev, _, err = llm.OpenDevice(ctx, e.NPU.Device, e.Cache); err != nil {
				return nil, err
			}
			t, err = transcribe.Load(ctx, dev, transcribe.Config{Dir: f.Path("dir"), Encoder: f.Path("encoder"),
				Cross: f.Path("cross"), Decoder: f.Path("decoder"), Weights: f.Path("weights")})
		}
		if err != nil {
			dev.Close()
			return nil, err
		}
		return &Transcriber{t: t, dev: dev, model: e.Model}, nil
	},
}

// logf writes lines to w (nil for nowhere).
func logf(w io.Writer) func(string, ...any) {
	return func(format string, args ...any) {
		if w != nil {
			fmt.Fprintf(w, format+"\n", args...)
		}
	}
}

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
}

// Model is the name of the model in the definitions.
func (t *Transcriber) Model() string { return t.model }

// OnNPU reports whether the programs were verified to run on the NPU.
func (t *Transcriber) OnNPU() bool { return t.t.OnNPU() }

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
// chunk with their times. Captions while someone speaks (provisional chunks read again as the
// speech grows) are to come.
func (t *Transcriber) TranscribeStream(ctx context.Context, s flow.Stream[[]float32], o TranscribeOptions) flow.Stream[flow.Text] {
	return func(yield func(flow.Chunk[flow.Text], error) bool) {
		var pcm []float32
		var start, end time.Duration
		flush := func() bool {
			if len(pcm) == 0 {
				return true
			}
			tr, err := t.Transcribe(ctx, pcm, o)
			pcm = pcm[:0]
			if err != nil {
				yield(flow.Chunk[flow.Text]{}, err)
				return false
			}
			if tr.Text == "" {
				return true
			}
			return yield(flow.Chunk[flow.Text]{Data: flow.Text{Text: tr.Text, Lang: tr.Lang}, Start: start, Len: end - start, Final: true}, nil)
		}
		for c, err := range s {
			if err != nil {
				yield(flow.Chunk[flow.Text]{}, err)
				return
			}
			if len(pcm) == 0 {
				start = c.Start
			}
			pcm = append(pcm, c.Data...)
			end = c.End()
			if c.Final && !flush() {
				return
			}
		}
		flush()
	}
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
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
