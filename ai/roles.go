package ai

import (
	"cmp"
	"context"
	"io"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/speech"
	"github.com/shibukawa/hwnpu/internal/transcribe"
)

// The roles' preparing (download and conversion, as hwnpu-audio and hwnpu-chat do on their
// first run, with the same programs, so that what one prepared the other uses) and loading
// (from the record alone: no network, no conversion).

func prepareTranscribe(e prepareEnv) (*record, error) {
	c, err := transcribe.Prepare(e.ctx, transcribe.Setup{Dir: e.dir, Repo: e.model.Source(), Log: e.log})
	if err != nil {
		return nil, err
	}
	return &record{Transcribe: &transcribeRecord{Dir: c.Dir, Encoder: c.Encoder, Cross: c.Cross, Decoder: c.Decoder, Weights: c.Weights}}, nil
}

func loadTranscribe(ctx context.Context, l located, env loadEnv) (io.Closer, error) {
	r := l.Transcribe
	var dev *hwnpu.Device
	if !env.cpu {
		var err error
		if dev, _, err = llm.OpenDevice(ctx, "", env.cache); err != nil {
			return nil, err
		}
	}
	t, err := transcribe.Load(ctx, dev, transcribe.Config{Dir: r.Dir, Encoder: r.Encoder, Cross: r.Cross, Decoder: r.Decoder, Weights: r.Weights})
	if err != nil {
		if dev != nil {
			dev.Close()
		}
		return nil, err
	}
	return &Transcriber{t: t, dev: dev, model: l.Model, cpu: env.cpu}, nil
}

func prepareSpeech(e prepareEnv) (*record, error) {
	c, err := speech.Prepare(e.ctx, speech.Setup{Dir: e.dir, Repo: e.model.Source(), Log: e.log})
	if err != nil {
		return nil, err
	}
	return &record{Speech: &speechRecord{Text: c.Text, Depth: c.Depth, Codec: c.Codec, Dir: c.Dir, Weights: c.Weights}}, nil
}

func loadSpeech(ctx context.Context, l located, env loadEnv) (io.Closer, error) {
	r := l.Speech
	s, err := speech.Load(ctx, speech.Config{Text: r.Text, Depth: r.Depth, Codec: r.Codec, Dir: r.Dir, Weights: r.Weights,
		CacheDir: env.cache, CPU: env.cpu, Log: env.log})
	if err != nil {
		return nil, err
	}
	return &Speaker{s: s, model: l.Model}, nil
}

func prepareTranslate(e prepareEnv) (*record, error) {
	// AMD's NPU runs an export of IRON instead of the programs of the other NPUs
	iron, err := chat.AMDExport(e.ctx, e.model.Name, "", e.root, e.log)
	if err != nil {
		return nil, err
	}
	if iron != "" {
		return &record{Chat: &chatRecord{IRON: iron, Name: e.model.Name}}, nil
	}
	tiers, err := chat.Tiers(chat.DefaultTiers)
	if err != nil {
		return nil, err
	}
	m := e.model
	// hwnpu-chat's defaults with the model's definition (models.yaml's context, paged, prefill
	// and format), so that the programs are the ones it converts
	c, err := chat.Prepare(e.ctx, chat.Setup{Dir: e.dir, Repo: m.Source(), Arch: m.Arch, Name: m.Name,
		Context: m.Run.Context, Paged: m.Run.Paged, Prefill: m.Run.Prefill, Format: m.Run.Format,
		Draft: 8, MTP: chat.MTPAuto, Tiers: tiers, Log: e.log})
	if err != nil {
		return nil, err
	}
	return &record{Chat: &chatRecord{Packages: c.Packages, Tiers: c.Tiers, MTP: c.MTP, ModelConfig: c.ModelConfig,
		Weights: c.Weights, Tokenizer: c.Tokenizer, Name: cmp.Or(c.Name, m.Name), Backend: c.Backend}}, nil
}

func loadTranslate(ctx context.Context, l located, env loadEnv) (io.Closer, error) {
	r := l.Chat
	e, err := chat.Load(ctx, chat.Config{Packages: r.Packages, Tiers: r.Tiers, MTP: r.MTP, ModelConfig: r.ModelConfig,
		Weights: r.Weights, Tokenizer: r.Tokenizer, IRON: r.IRON, Name: r.Name, Backend: r.Backend,
		CacheDir: env.cache, Log: env.log})
	if err != nil {
		return nil, err
	}
	return &Translator{e: e, model: l.Model}, nil
}
