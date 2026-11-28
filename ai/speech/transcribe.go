package speech

import (
	"context"
	"io"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/speech/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/speechwire"
)

// TranscribeRate is the samples a second Transcriber reads.
const TranscribeRate = speechwire.TranscribeRate

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
// transcribeRole's Prepare and Load are the engine's (ai/speech/engine).
var transcribeRole = &airole.Role{
	Name:   "transcribe",
	Remote: func(h *airpc.Handle) io.Closer { return &Transcriber{p: airpc.RemoteProxy[transcriberImpl](h)} },
	Kinds:  []catalog.Kind{catalog.KindTranscribe},
	// 1: the tokenizer as its binary file (chat.PrepareTokenizer) in the files, as "tokenizer":
	// a store prepared before named none, and Load then read the checkpoint's tokenizer.json, which
	// a checkpoint fetched since is without
	Revision: 1,
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
	p airpc.Proxy[transcriberImpl]
}

// transcriberImpl is a transcription model of the engine in this process.
type transcriberImpl interface {
	Model() string
	OnCPU() bool
	OnNPU() bool
	Transcribe(ctx context.Context, pcm []float32, o TranscribeOptions) (Transcript, error)
	Close() error
}

// OnCPU reports whether the model runs on hwnpu's CPU interpreter (its definition's device cpu).
func (t *Transcriber) OnCPU() bool { return airpc.ProxyValue(t.p, "OnCPU", transcriberImpl.OnCPU) }

// Model is the name of the model in the definitions.
func (t *Transcriber) Model() string { return airpc.ProxyValue(t.p, "Model", transcriberImpl.Model) }

// OnNPU reports whether the programs were verified to run on the NPU.
func (t *Transcriber) OnNPU() bool { return airpc.ProxyValue(t.p, "OnNPU", transcriberImpl.OnNPU) }

// Transcribe turns speech (mono, at TranscribeRate, in [-1, 1]) into text. Speech longer than 30
// seconds is read in windows, each starting where the last one's last whole segment ended.
func (t *Transcriber) Transcribe(ctx context.Context, pcm []float32, o TranscribeOptions) (Transcript, error) {
	return airpc.ProxyCall1(t.p, ctx, "Transcribe", func(l transcriberImpl) (Transcript, error) {
		return l.Transcribe(ctx, pcm, o)
	}, pcm, o)
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
func (t *Transcriber) Close() error { return t.p.Close() }

func init() {
	airole.Register(transcribeRole)
	bridge.Roles[transcribeRole.Name] = transcribeRole
	bridge.Wrap[transcribeRole.Name] = func(l any) io.Closer { return &Transcriber{p: airpc.LocalProxy(l.(transcriberImpl))} }
}
