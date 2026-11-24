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
	"github.com/shibukawa/hwnpu/internal/airpc"
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

// The settings and programs a transcription model's definition may give (ai.Definition):
//
//	settings: format    the programs' weight format: lut8 (the default), lut6, lut4 or fp16
//	programs: iron      on AMD's NPU, an export of tools/amd/iron_whisper.py (its directory)
//
// A model of device cpu runs on hwnpu's CPU interpreter (slow).
var transcribeRole = &airole.Role{
	Name:   "transcribe",
	Remote: func(h *airpc.Handle) io.Closer { return &Transcriber{remote: h} },
	Kinds:  []chat.Kind{chat.KindTranscribe},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		cpu := e.NPU.CPU(e.Model)
		if dir := first(e.Model.Programs["iron"]); dir != "" {
			if !transcribe.IsIronExport(dir) {
				return f, fmt.Errorf("%s is not an export of tools/amd/iron_whisper.py", dir)
			}
			f.Set("iron", dir)
			return f, nil
		}
		if e.NPU.AMD() && !cpu {
			// AMD's NPU runs the graphs of tools/amd/iron_whisper.py
			dir, err := transcribe.PrepareAMD(e.Ctx, e.Base, e.Log)
			if err != nil {
				return f, err
			}
			iron.ShareExport(dir, logf(e.Log))
			f.Set("iron", dir)
			return f, nil
		}
		backend := e.NPU.Format
		if cpu {
			backend = llm.BackendCoreML // what the CPU interpreter runs
		}
		c, err := transcribe.Prepare(e.Ctx, transcribe.Setup{Dir: e.Dir, Repo: e.Model.Source(), Format: e.Model.Settings["format"], Backend: backend, Log: e.Log})
		if err != nil {
			return f, err
		}
		f.Set("dir", c.Dir)
		f.Set("tokenizer", c.Tokenizer)
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
		cpu := e.NPU.CPU(e.Def)
		if dir := f.Path("iron"); dir != "" {
			if dev, err = hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD)); err != nil {
				return nil, err
			}
			t, err = transcribe.LoadIron(ctx, dev, dir)
		} else {
			if !cpu {
				if dev, _, err = llm.OpenDevice(ctx, e.NPU.Device, e.Cache); err != nil {
					return nil, err
				}
			}
			t, err = transcribe.Load(ctx, dev, transcribe.Config{Dir: f.Path("dir"), Tokenizer: f.Path("tokenizer"), Encoder: f.Path("encoder"),
				Cross: f.Path("cross"), Decoder: f.Path("decoder"), Weights: f.Path("weights")})
		}
		if err != nil {
			if dev != nil {
				dev.Close()
			}
			return nil, err
		}
		return &Transcriber{t: t, dev: dev, model: e.Model, cpu: cpu}, nil
	},
}

// first is the first of ps, "" for none.
func first(ps []string) string {
	if len(ps) == 0 {
		return ""
	}
	return ps[0]
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
	// Windows are the windows of 30 seconds the speech was read in, with what each took.
	Windows []Window
	// Took is how long the transcription took.
	Took time.Duration
}

// Window is a window of 30 seconds of a transcription and what reading it took.
type Window struct {
	// Start is where it begins in the speech, Lang its language.
	Start time.Duration
	Lang  string
	// Tokens are the tokens of text generated (its timestamps among them).
	Tokens int
	// Features is the time spent on the features (the log-mel spectrogram), Encode on the
	// encoder and the cross attention's keys and values, Decode on decoding the tokens.
	Features, Encode, Decode time.Duration
}

// Segment is a piece of a transcript.
type Segment struct {
	Start, End time.Duration
	Text       string
}

// Transcriber turns speech into text (Whisper). It runs one request at a time; others wait.
type Transcriber struct {
	remote *airpc.Handle // the model in hwnpud (ai's HWNPU_DAEMON), nil when it is here
	mu     sync.Mutex
	t      *transcribe.Transcriber
	dev    *hwnpu.Device
	model  string
	cpu    bool
}

// OnCPU reports whether the model runs on hwnpu's CPU interpreter (its definition's device cpu).
func (t *Transcriber) OnCPU() bool {
	if t.remote != nil {
		return airpc.Value[bool](t.remote, "OnCPU")
	}
	return t.cpu
}

// Model is the name of the model in the definitions.
func (t *Transcriber) Model() string {
	if t.remote != nil {
		return airpc.Value[string](t.remote, "Model")
	}
	return t.model
}

// OnNPU reports whether the programs were verified to run on the NPU.
func (t *Transcriber) OnNPU() bool {
	if t.remote != nil {
		return airpc.Value[bool](t.remote, "OnNPU")
	}
	return t.t.OnNPU()
}

// Transcribe turns speech (mono, at TranscribeRate, in [-1, 1]) into text. Speech longer than 30
// seconds is read in windows, each starting where the last one's last whole segment ended.
func (t *Transcriber) Transcribe(ctx context.Context, pcm []float32, o TranscribeOptions) (Transcript, error) {
	if t.remote != nil {
		return airpc.Call1[Transcript](t.remote, ctx, "Transcribe", pcm, o)
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	t0 := time.Now()
	res, err := t.t.Transcribe(ctx, pcm, transcribe.Options{Language: o.Language, Translate: o.Translate, MaxTokens: o.MaxTokens})
	if err != nil {
		return Transcript{}, err
	}
	out := Transcript{Text: strings.TrimSpace(res.Text), Took: time.Since(t0)}
	for _, w := range res.Windows {
		if out.Lang == "" {
			out.Lang = w.Language
		}
		out.Windows = append(out.Windows, Window{Start: w.Start, Lang: w.Language, Tokens: len(w.Tokens),
			Features: w.Features, Encode: w.Encode, Decode: w.Decode})
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
	return t.transcribeStream(ctx, s, func() TranscribeOptions { return o })
}

// TranscribeNode is the node of a transcriber in a network or a call (flow, ai.App.From, ai.NewCall): utterances
// in (the VAD's node before it), their texts out, as TranscribeStream gives them. Its settings
// may be changed while the network runs (Update): the next utterance is read with them.
func TranscribeNode(u ai.Use[*Transcriber], o TranscribeOptions) *ai.ModelNode[[]float32, flow.Text, TranscribeOptions] {
	return ai.NewModelNode(u, o, func(ctx context.Context, t *Transcriber, in flow.Stream[[]float32], options func() TranscribeOptions) flow.Stream[flow.Text] {
		return t.transcribeStream(ctx, in, options)
	})
}

// transcribeStream is TranscribeStream with the options asked for at each segment.
func (t *Transcriber) transcribeStream(ctx context.Context, s flow.Stream[[]float32], options func() TranscribeOptions) flow.Stream[flow.Text] {
	return func(yield func(flow.Chunk[flow.Text], error) bool) {
		var pcm []float32
		var start, end time.Duration
		flush := func() bool {
			if len(pcm) == 0 {
				return true
			}
			tr, err := t.Transcribe(ctx, pcm, options())
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
	if t.remote != nil {
		return t.remote.Release() // the Runtime in hwnpud unloads it
	}
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

func init() { airole.Register(transcribeRole) }
