package engine

import (
	"context"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/speech"
	"github.com/shibukawa/hwnpu/ai/speech/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	"github.com/shibukawa/hwnpu/internal/transcribe"
)

// prepareTranscribe is the transcription role's Prepare.
func prepareTranscribe(e airole.PrepareEnv) (f airole.Files, err error) {
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
}

// loadTranscribe is the transcription role's Load.
func loadTranscribe(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
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
	return bridge.Wrap["transcribe"](&transcriber{t: t, dev: dev, model: e.Model, cpu: cpu}), nil
}

// transcriber is a transcription model of the engine (speech.Transcriber's). It runs one request
// at a time; others wait.
type transcriber struct {
	mu    sync.Mutex
	t     *transcribe.Transcriber
	dev   *hwnpu.Device
	model string
	cpu   bool
}

func (t *transcriber) OnCPU() bool   { return t.cpu }
func (t *transcriber) Model() string { return t.model }
func (t *transcriber) OnNPU() bool   { return t.t.OnNPU() }

func (t *transcriber) Transcribe(ctx context.Context, pcm []float32, o speech.TranscribeOptions) (speech.Transcript, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t0 := time.Now()
	res, err := t.t.Transcribe(ctx, pcm, transcribe.Options{Language: o.Language, Translate: o.Translate, MaxTokens: o.MaxTokens})
	if err != nil {
		return speech.Transcript{}, err
	}
	out := speech.Transcript{Text: strings.TrimSpace(res.Text), Took: time.Since(t0)}
	for _, w := range res.Windows {
		if out.Lang == "" {
			out.Lang = w.Language
		}
		out.Windows = append(out.Windows, speech.Window{Start: w.Start, Lang: w.Language, Tokens: len(w.Tokens),
			Features: w.Features, Encode: w.Encode, Decode: w.Decode})
		for _, s := range w.Segments {
			out.Segments = append(out.Segments, speech.Segment{Start: w.Start + s.Start, End: w.Start + s.End, Text: strings.TrimSpace(t.t.SegmentText(s))})
		}
	}
	if o.Translate {
		out.Lang = "en"
	}
	return out, nil
}

func (t *transcriber) Close() error {
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
