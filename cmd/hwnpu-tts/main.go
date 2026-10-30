// hwnpu-tts speaks text with LFM3-Audio-2B on the Neural Engine and writes a WAV file (24 kHz).
//
//	hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -seq 1 -cache 2048 -hidden-out 1 -o DIR/tts-q1.mlpackage
//	hwnpu-convert -config DIR/config.json -weights DIR/model.safetensors -seq 64 -cache 2048 -hidden-out 1 -o DIR/tts-q64.mlpackage
//	hwnpu-convert -arch lfm3-depthformer -config DIR/config.json -weights DIR/model.safetensors -format fp16 -o DIR/depth.mlpackage
//	hwnpu-convert -config DIR/audio_tokenizer/config.json -weights DIR/audio_tokenizer/model.safetensors -codec-frames 64 -format fp16 -o DIR/codec.mlpackage
//	hwnpu-tts -model DIR -text DIR/tts-q1.parts.json,DIR/tts-q64.mlpackage -depth DIR/depth.mlpackage -codec DIR/codec.mlpackage -lang ja -o out.wav こんにちは
//
// The text model is converted with a KV cache and -hidden-out 1 (it outputs its final hidden
// state, which the depthformer reads); the depthformer and the codec decoder are programs of
// their own (internal/speech). -lang is a language code of the model card (en, ja, ...) or a
// language's name, -voice female or male. -cpu runs on the CPU interpreter, slowly.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/speech"
)

func main() {
	model := flag.String("model", "", "directory of the checkpoint: config.json, model.safetensors, tokenizer.json, audio_tokenizer/config.json")
	text := flag.String("text", "", "the text model converted with -cache and -hidden-out 1, comma separated packages of different -seq")
	depth := flag.String("depth", "", "the depthformer (hwnpu-convert -arch lfm3-depthformer)")
	codec := flag.String("codec", "", "the codec decoder (hwnpu-convert on audio_tokenizer/, -arch lfm3-codec)")
	lang := flag.String("lang", "en", "language: a code of the model card (en, en-gb, ja, zh, ...) or a language's name")
	voice := flag.String("voice", "female", "voice: female or male")
	out := flag.String("o", "out.wav", "WAV file to write")
	seed := flag.Uint64("seed", 0, "random seed of the speech codes (0: random)")
	maxFrames := flag.Int("max-frames", 0, "most audio frames of 80 ms (0: until the end of audio or a full cache)")
	cpu := flag.Bool("cpu", false, "run on the CPU interpreter instead of the Neural Engine (slow)")
	cacheDir := flag.String("cache", "", "compile cache (default: hwnpu in the user cache directory)")
	flag.Parse()
	if *model == "" || *text == "" || *depth == "" || *codec == "" || flag.NArg() == 0 {
		flag.Usage()
		os.Exit(2)
	}
	if *cacheDir == "" {
		if d, err := os.UserCacheDir(); err == nil {
			*cacheDir = filepath.Join(d, "hwnpu")
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, speech.Config{Text: strings.Split(*text, ","), Depth: *depth, Codec: *codec, Dir: *model, CacheDir: *cacheDir, CPU: *cpu, Log: os.Stderr},
		speech.Request{Text: strings.Join(flag.Args(), " "), Language: *lang, Voice: *voice, Seed: *seed, MaxFrames: *maxFrames}, *out); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-tts:", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, c speech.Config, r speech.Request, out string) error {
	s, err := speech.Load(ctx, c)
	if err != nil {
		return err
	}
	defer s.Close()
	t0 := time.Now()
	pcm, frames, err := s.Speak(ctx, r)
	if err != nil {
		return err
	}
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	if err := speech.WriteWAV(f, pcm, s.SampleRate()); err != nil {
		f.Close()
		return err
	}
	secs := float64(len(pcm)) / float64(s.SampleRate())
	fmt.Fprintf(os.Stderr, "%s: %d frames, %.2f s of audio in %.1f s\n", out, len(frames), secs, time.Since(t0).Seconds())
	return f.Close()
}
