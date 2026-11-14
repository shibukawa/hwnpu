package main

import (
	"context"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/speech"
)

// TTSCmd is "hwnpu-audio tts": speaks a text with LFM3-Audio-2B on the Neural Engine into a WAV
// file (24 kHz). The first run downloads LFM3-Audio-2B into --dir (its download is shared with
// hwnpu-chat's LFM3-Audio-2B) and converts the programs by itself (internal/speech Prepare).
// Programs converted by hand are given with --checkpoint, --text, --depth and --codec instead:
//
//	hwnpu-dev convert -backend coreai -config DIR/config.json -weights DIR/model.safetensors -seq 1 -cache 2048 -hidden-out 1 -o DIR/tts-q1.aimodel
//	hwnpu-dev convert -backend coreai -config DIR/config.json -weights DIR/model.safetensors -seq 64 -cache 2048 -hidden-out 1 -o DIR/tts-q64.aimodel
//	hwnpu-dev convert -backend coreai -arch lfm3-depthformer -config DIR/config.json -weights DIR/model.safetensors -format fp16 -o DIR/depth.aimodel
//	hwnpu-dev convert -backend coreai -config DIR/audio_tokenizer/config.json -weights DIR/audio_tokenizer/model.safetensors -codec-frames 64 -format fp16 -o DIR/codec.aimodel
//	hwnpu-audio tts --checkpoint DIR --text DIR/tts-q1.parts.json,DIR/tts-q64.aimodel --depth DIR/depth.aimodel --codec DIR/codec.aimodel --lang ja -o out.wav こんにちは
//
// On a Mac tts runs on the Neural Engine through Core AI (macOS 27); its own conversion writes
// the text model's two programs as the functions of one Core AI program, the weights once.
// --cpu runs Core ML packages (-backend coreml) with the CPU interpreter.
//
// The text model is converted with a KV cache and -hidden-out 1 (it outputs its final hidden
// state, which the depthformer reads); the depthformer and the codec decoder are programs of
// their own (internal/speech).
type TTSCmd struct {
	Text      []string `arg:"" name:"TEXT" help:"The text to speak (the words are joined with spaces)."`
	Model     string   `short:"m" default:"speech" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (speech, lfm3-audio-tts:2b), or a Hugging Face repository OWNER/NAME of an LFM3-Audio checkpoint."`
	Out       string   `short:"o" default:"out.wav" type:"path" placeholder:"FILE" help:"The WAV file to write."`
	Lang      string   `default:"en" placeholder:"LANG" help:"The language: a code of the model card (en, en-gb, ja, zh, ...) or a language's name."`
	Voice     string   `enum:"female,male" default:"female" help:"The voice: female or male."`
	Seed      uint64   `placeholder:"N" help:"The random seed of the speech codes (0: a new one, logged so that it can be repeated)."`
	MaxFrames int      `name:"max-frames" placeholder:"N" help:"Most audio frames of 80 ms (0: until the end of audio or a full cache)."`
	Format    string   `placeholder:"FORMAT" help:"Weight format of the text model's programs (default: the model's, 8 bit tables)."`
	Dir       string   `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's --dir)."`
	Cache     string   `placeholder:"DIR" help:"Compile cache (default: hwnpu in the user cache directory)."`
	Models    string   `placeholder:"FILES" help:"${models_usage}."`
	CPU       bool     `name:"cpu" help:"Run on the CPU interpreter instead of the Neural Engine (slow)."`

	Checkpoint string `group:"Converted by hand" type:"existingdir" placeholder:"DIR" help:"The checkpoint's directory (config.json, model.safetensors, tokenizer.json, audio_tokenizer/config.json) of the programs below."`
	TextModel  string `group:"Converted by hand" name:"text" placeholder:"PKG,..." help:"The text model converted with -cache and -hidden-out 1, packages of different -seq, comma separated."`
	Depth      string `group:"Converted by hand" placeholder:"PKG" help:"The depthformer (hwnpu-dev convert -arch lfm3-depthformer)."`
	Codec      string `group:"Converted by hand" placeholder:"PKG" help:"The codec decoder (hwnpu-dev convert on audio_tokenizer/, -arch lfm3-codec)."`
}

// Run speaks the text.
func (c *TTSCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	manual := c.Checkpoint != "" || c.TextModel != "" || c.Depth != "" || c.Codec != ""
	if manual && (c.Checkpoint == "" || c.TextModel == "" || c.Depth == "" || c.Codec == "") {
		return fmt.Errorf("programs converted by hand take all of --checkpoint, --text, --depth and --codec")
	}
	cfg := speech.Config{Text: strings.Split(c.TextModel, ","), Depth: c.Depth, Codec: c.Codec, Dir: c.Checkpoint}
	if !manual {
		m, err := resolveKind(c.Model, chat.KindSpeech, c.Models, lw)
		if err != nil {
			return err
		}
		backend := ""
		if !c.CPU { // the CPU interpreter runs Core ML packages
			if backend, err = audioFormat(ctx); err != nil {
				return err
			}
		}
		if cfg, err = speech.Prepare(ctx, speech.Setup{Dir: filepath.Join(modelsDir(c.Dir), m.DirName()), Repo: m.Source(), Format: c.Format, Backend: backend, Log: lw}); err != nil {
			return err
		}
	}
	cfg.CacheDir, cfg.CPU, cfg.Device, cfg.Log = cacheDir(c.Cache), c.CPU, audioDevice(), lw
	seed := c.Seed
	if seed == 0 {
		seed = rand.Uint64() // drawn here so that it can be told (--seed repeats the speech)
	}
	s, err := speech.Load(ctx, cfg)
	if err != nil {
		return err
	}
	defer s.Close()
	console.Running()
	t0 := time.Now()
	pcm, frames, err := s.Speak(ctx, speech.Request{Text: strings.Join(c.Text, " "), Language: c.Lang, Voice: c.Voice, Seed: seed, MaxFrames: c.MaxFrames})
	if err != nil {
		return err
	}
	f, err := os.Create(c.Out)
	if err != nil {
		return err
	}
	if err := speech.WriteWAV(f, pcm, s.SampleRate()); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	secs := float64(len(pcm)) / float64(s.SampleRate())
	fmt.Fprintf(lw, "%d frames, %.2f s of audio in %.1f s (seed %d)\n", len(frames), secs, time.Since(t0).Seconds(), seed)
	fmt.Println(c.Out)
	return nil
}
