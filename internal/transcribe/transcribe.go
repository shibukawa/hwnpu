// Package transcribe turns speech into text with Whisper (internal/llm/whisper) on the NPU, as
// transformers' WhisperForConditionalGeneration.generate does with greedy decoding and no
// timestamps: the audio in windows of 30 seconds, each window's log mel frames through the
// encoder, the cross program giving the decoder's cross attention keys and values, then the
// decoder reading <|startoftranscript|>, the language (detected from the logits after
// <|startoftranscript|> when not given), the task and <|notimestamps|>, and generating the
// window's text until <|endoftext|>, with the checkpoint's suppress_tokens (and
// begin_suppress_tokens for the first token) left out.
package transcribe

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/decode"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
	"github.com/shibukawa/hwnpu/internal/llm/whisper"
	"github.com/shibukawa/hwnpu/internal/tokenizer"
)

// Config names a checkpoint's directory (config.json, generation_config.json, tokenizer.json)
// and its three programs. Weights is the file the decoder's token embeddings are read from: the
// tables Prepare writes (chat.TablesFile), or the checkpoint ("": model.safetensors in Dir).
type Config struct {
	Dir                     string
	Encoder, Cross, Decoder string
	Weights                 string
	// Tokenizer is the tokenizer to load ("": Dir's tokenizer.json, or its binary file):
	// Prepare's of chat.PrepareTokenizer.
	Tokenizer string
}

// generation is what decoding needs of generation_config.json.
type generation struct {
	SuppressTokens      []int32          `json:"suppress_tokens"`
	BeginSuppressTokens []int32          `json:"begin_suppress_tokens"`
	LangToID            map[string]int32 `json:"lang_to_id"`
	TaskToID            map[string]int32 `json:"task_to_id"`
	NoTimestamps        int32            `json:"no_timestamps_token_id"`
	MaxInitialTimestamp int              `json:"max_initial_timestamp_index"`
	DecoderStart        int32            `json:"decoder_start_token_id"`
	EOS                 int32            `json:"eos_token_id"`
}

// Transcriber holds the loaded programs of a Whisper checkpoint.
type Transcriber struct {
	cfg   *whisper.Config
	gen   generation
	tok   *tokenizer.Tokenizer
	p     programs
	langs map[int32]string // language tokens' codes
}

// programs run a checkpoint's encoder and decoder: the converted programs (Core AI's, OpenVINO
// IR, Core ML's on the CPU interpreter) or the graphs of tools/amd/iron_whisper.py on AMD's NPU
// (ironPrograms).
type programs interface {
	// encode reads a window's log mel frames (frame-major, 3000 x mel bins) and gives the
	// decoder its cross attention's keys and values.
	encode(ctx context.Context, mel []float32) error
	// reset starts the decoder at position 0.
	reset()
	// step reads one token at the next position and returns the logits after it.
	step(ctx context.Context, id int32) ([]float32, error)
	// positions is the decoder's positions; pos the next one.
	positions() int
	pos() int
	onNPU() bool
	Close() error
}

// converted are the programs Prepare converts: the encoder, the cross program and the decoder
// with a KV cache the host holds.
type converted struct {
	cfg      *whisper.Config
	dec      *whisper.Decoder
	w        llm.Weights // the token embeddings, looked up per step
	enc, crs runner.Program
	sess     *decode.Session
	closers  []io.Closer
}

// Load reads the checkpoint's settings and tokenizer and loads the programs on dev (nil: the CPU
// interpreter).
func Load(ctx context.Context, dev *hwnpu.Device, c Config) (*Transcriber, error) {
	raw, err := os.ReadFile(filepath.Join(c.Dir, "config.json"))
	if err != nil {
		return nil, err
	}
	dec, err := whisper.ParseDecoderConfig(raw)
	if err != nil {
		return nil, err
	}
	t, err := newTranscriber(c.Dir, c.Tokenizer, dec.Config)
	if err != nil {
		return nil, err
	}
	p := &converted{cfg: dec.Config, dec: dec}
	ok := false
	defer func() {
		if !ok {
			p.Close()
		}
	}()
	if p.w, err = llm.OpenWeights(cmp.Or(c.Weights, filepath.Join(c.Dir, "model.safetensors")), whisper.Arch); err != nil {
		return nil, err
	}
	if p.enc, err = runner.Open(ctx, dev, c.Encoder); err != nil {
		return nil, fmt.Errorf("transcribe: encoder: %w", err)
	}
	if p.crs, err = runner.Open(ctx, dev, c.Cross); err != nil {
		return nil, fmt.Errorf("transcribe: cross: %w", err)
	}
	s, cl, err := runner.OpenSession(ctx, dev, []string{c.Decoder}, dec)
	if err != nil {
		return nil, fmt.Errorf("transcribe: decoder: %w", err)
	}
	p.sess = s
	p.closers = append(p.closers, cl)
	ok = true
	t.p = p
	return t, nil
}

// newTranscriber reads the settings and the tokenizer of the checkpoint in dir.
// newTranscriber reads the checkpoint's generation_config.json in dir and the tokenizer tok
// ("": dir's tokenizer.json).
func newTranscriber(dir, tok string, cfg *whisper.Config) (*Transcriber, error) {
	t := &Transcriber{cfg: cfg, langs: map[int32]string{}}
	raw, err := os.ReadFile(filepath.Join(dir, "generation_config.json"))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &t.gen); err != nil {
		return nil, fmt.Errorf("transcribe: generation_config.json: %w", err)
	}
	if len(t.gen.LangToID) == 0 || t.gen.TaskToID["transcribe"] == 0 || t.gen.NoTimestamps == 0 {
		return nil, fmt.Errorf("transcribe: generation_config.json has no lang_to_id, task_to_id or no_timestamps_token_id (an English only checkpoint is not supported)")
	}
	for tok, id := range t.gen.LangToID {
		t.langs[id] = strings.TrimSuffix(strings.TrimPrefix(tok, "<|"), "|>")
	}
	if tok == "" {
		tok = filepath.Join(dir, "tokenizer.json")
	}
	if t.tok, err = tokenizer.Load(tok); err != nil {
		return nil, err
	}
	return t, nil
}

// OnNPU reports whether the encoder (and the cross program) run on the NPU.
func (t *Transcriber) OnNPU() bool { return t.p.onNPU() }

// Close releases the programs.
func (t *Transcriber) Close() error {
	if t.p == nil {
		return nil
	}
	return t.p.Close()
}

func (p *converted) onNPU() bool { return p.enc.OnNPU() && p.crs.OnNPU() }

func (p *converted) Close() error {
	for _, x := range []runner.Program{p.enc, p.crs} {
		if x != nil {
			x.Close()
		}
	}
	for _, c := range p.closers {
		c.Close()
	}
	if p.w != nil {
		p.w.Close()
	}
	return nil
}

func (p *converted) encode(ctx context.Context, mel []float32) error {
	h, err := p.enc.Run(ctx, map[string][]float32{"x": mel})
	if err != nil {
		return fmt.Errorf("transcribe: encoder: %w", err)
	}
	kv, err := p.crs.Run(ctx, map[string][]float32{"x": h})
	if err != nil {
		return fmt.Errorf("transcribe: cross: %w", err)
	}
	n := len(kv) / (2 * p.cfg.DecoderLayers)
	buf := make([]byte, 2*n)
	for i := 0; i < p.cfg.DecoderLayers; i++ {
		for j, name := range []string{"xk", "xv"} {
			hwnpu.EncodeF16(buf, kv[(2*i+j)*n:(2*i+j+1)*n])
			if err := p.sess.SetInputF16(fmt.Sprintf("%s_%d", name, i), buf); err != nil {
				return err
			}
		}
	}
	return nil
}

func (p *converted) reset()         { p.sess.Reset() }
func (p *converted) positions() int { return p.sess.Cache() }
func (p *converted) pos() int       { return p.sess.Pos() }

func (p *converted) step(_ context.Context, id int32) ([]float32, error) {
	row, err := llm.TokenRow(p.w, p.dec, id)
	if err != nil {
		return nil, err
	}
	logits, err := p.sess.Step([][]float32{row})
	if err != nil {
		return nil, fmt.Errorf("transcribe: decoder: %w", err)
	}
	return logits, nil
}

// Options are a transcription's settings.
type Options struct {
	// Language is a language code of the checkpoint (ja, en, ...); empty detects it in each window.
	Language string
	// Translate turns the speech into English (the task translate) rather than its own language.
	Translate bool
	// MaxTokens is the most tokens of text per window (0: as many as the decoder holds).
	MaxTokens int
}

// Window is one window of up to 30 seconds and what came of it.
type Window struct {
	Start    time.Duration // of the audio
	Language string
	// Segments are the window's text: one, from 0 to the window's length, for a clip of up to 30
	// seconds (decoded without timestamps); otherwise those between the timestamps, relative to
	// Start.
	Segments []Segment
	Tokens   []int32 // generated, the timestamps among them
	// Time spent: the features, the encoder and the cross program, and decoding.
	Features, Encode, Decode time.Duration
}

// Result is a transcription.
type Result struct {
	Text    string
	Windows []Window
}

// Transcribe turns samples (mono, at whisper.SampleRate, in [-1, 1]) into text. A clip of up to 30
// seconds is decoded in one window without timestamps; a longer one as transformers' long form
// generation does: each window of 30 seconds with timestamps, the next starting where the
// window's last complete segment ends (the speech after it is read again there), or after the
// whole window when its last segment is left open.
func (t *Transcriber) Transcribe(ctx context.Context, samples []float32, o Options) (Result, error) {
	var res Result
	mels, window := t.cfg.MelBins, whisper.ClipSamples/hop
	if len(samples) <= whisper.ClipSamples {
		t0 := time.Now()
		mel := whisper.Features(samples, mels)
		w, err := t.window(ctx, mel, o, false)
		if err != nil {
			return res, err
		}
		w.Features += time.Since(t0) - w.Encode - w.Decode
		w.Segments = []Segment{{End: time.Duration(len(samples)) * time.Second / whisper.SampleRate, Tokens: w.Tokens}}
		res.Windows = append(res.Windows, w)
	} else {
		// the features of the whole clip, their floor from its loudest frame, read 30 seconds at
		// a time, the last window's past the clip zeros
		mel := whisper.FeaturesLong(samples, mels)
		total := len(mel) / mels
		in := make([]float32, window*mels)
		for seek := 0; seek < total; {
			frames := min(total-seek, window)
			clear(in)
			copy(in, mel[seek*mels:(seek+frames)*mels])
			w, err := t.window(ctx, in, o, true)
			if err != nil {
				return res, err
			}
			var advance int
			w.Segments, advance = segments(w.Tokens, t.gen.NoTimestamps+1, frames)
			w.Start = time.Duration(seek) * 10 * time.Millisecond
			res.Windows = append(res.Windows, w)
			seek += advance
		}
	}
	var text strings.Builder
	for _, w := range res.Windows {
		for _, s := range w.Segments {
			text.WriteString(t.tok.Decode(s.Tokens, true))
		}
	}
	res.Text = strings.TrimSpace(text.String())
	return res, nil
}

// hop is the samples of a mel frame.
const hop = whisper.SampleRate / 100

// SegmentText decodes a segment's tokens.
func (t *Transcriber) SegmentText(s Segment) string { return t.tok.Decode(s.Tokens, true) }

// window decodes one window of 30 seconds of mel frames, with timestamps or without.
func (t *Transcriber) window(ctx context.Context, mel []float32, o Options, timestamps bool) (Window, error) {
	var w Window
	t1 := time.Now()
	if err := t.p.encode(ctx, mel); err != nil {
		return w, err
	}
	t2 := time.Now()
	w.Encode = t2.Sub(t1)

	t.p.reset()
	logits, err := t.p.step(ctx, cmp.Or(t.gen.DecoderStart, int32(t.cfg.VocabSize)))
	if err != nil {
		return w, err
	}
	lang, ok := t.gen.LangToID["<|"+o.Language+"|>"]
	if o.Language == "" {
		lang = t.detect(logits)
	} else if !ok {
		return w, fmt.Errorf("transcribe: language %q is not among the checkpoint's", o.Language)
	}
	w.Language = t.langs[lang]
	task := t.gen.TaskToID["transcribe"]
	if o.Translate {
		if task = t.gen.TaskToID["translate"]; task == 0 {
			return w, fmt.Errorf("transcribe: the checkpoint has no task translate")
		}
	}
	prompt := []int32{lang, task}
	if !timestamps {
		prompt = append(prompt, t.gen.NoTimestamps)
	}
	for _, id := range prompt {
		if logits, err = t.p.step(ctx, id); err != nil {
			return w, err
		}
	}
	most := t.p.positions() - t.p.pos()
	if o.MaxTokens > 0 {
		most = min(most, o.MaxTokens)
	}
	inf := float32(math.Inf(-1))
	for len(w.Tokens) < most {
		if err := ctx.Err(); err != nil {
			return w, err
		}
		for _, id := range t.gen.SuppressTokens {
			logits[id] = inf
		}
		if len(w.Tokens) == 0 {
			for _, id := range t.gen.BeginSuppressTokens {
				logits[id] = inf
			}
		}
		if timestamps {
			timestampRules(logits, w.Tokens, t.gen.NoTimestamps+1, t.gen.EOS, cmp.Or(t.gen.MaxInitialTimestamp, 50))
		}
		next := argmax(logits)
		if next == t.gen.EOS {
			break
		}
		w.Tokens = append(w.Tokens, next)
		if len(w.Tokens) == most {
			break
		}
		if logits, err = t.p.step(ctx, next); err != nil {
			return w, err
		}
	}
	w.Decode = time.Since(t2)
	return w, nil
}

// detect is the language token with the largest logit (transformers' detect_language).
func (t *Transcriber) detect(logits []float32) int32 {
	best, bestV := int32(-1), float32(math.Inf(-1))
	for id := range t.langs {
		if int(id) < len(logits) && (best < 0 || logits[id] > bestV || logits[id] == bestV && id < best) {
			best, bestV = id, logits[id]
		}
	}
	return best
}

func argmax(v []float32) int32 {
	best := 0
	for i, x := range v {
		if x > v[best] {
			best = i
		}
	}
	return int32(best)
}
