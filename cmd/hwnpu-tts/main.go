// hwnpu-tts speaks text with LFM3-Audio-2B on the Neural Engine and writes a WAV file (24 kHz).
//
//	hwnpu-tts -lang ja -o out.wav こんにちは            # -use speech: downloaded and converted when missing
//	hwnpu-tts -list
//
// The first run downloads LFM3-Audio-2B into -dir (its download is shared with hwnpu-chat's
// LFM3-Audio-2B) and converts the programs below by itself (internal/speech Prepare). Programs
// converted by hand are given with -model, -text, -depth and -codec instead:
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
	"math/rand/v2"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/speech"
)

func main() {
	use := flag.String("use", "speech", "the model: a name, a use or a family of -list (speech, lfm3-audio-tts:2b), or a Hugging Face repository OWNER/NAME of an LFM3-Audio checkpoint")
	dir := flag.String("dir", chat.ModelsDir(), "where the models are kept (as hwnpu-chat's -dir)")
	format := flag.String("format", "", "weight format of the text model's programs (default: the model's, 8 bit tables)")
	list := flag.Bool("list", false, "list the speech models of the definitions and exit")
	modelFiles := flag.String("models", "", chat.ModelsUsage)
	model := flag.String("model", "", "with -text, -depth and -codec: directory of the checkpoint (config.json, model.safetensors, tokenizer.json, audio_tokenizer/config.json) of programs converted by hand")
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
	if err := chat.LoadUserModels(*modelFiles, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-tts:", err)
		os.Exit(2)
	}
	if *list {
		chat.List(os.Stdout, *dir, chat.KindSpeech)
		return
	}
	manual := *model != "" || *text != "" || *depth != "" || *codec != ""
	if flag.NArg() == 0 || manual && (*model == "" || *text == "" || *depth == "" || *codec == "") {
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
	c := speech.Config{Text: strings.Split(*text, ","), Depth: *depth, Codec: *codec, Dir: *model}
	if !manual {
		m, err := chat.ResolveFor(*use, chat.KindSpeech, os.Stderr)
		if err == nil {
			c, err = speech.Prepare(ctx, speech.Setup{Dir: filepath.Join(*dir, m.DirName()), Repo: m.Source(), Format: *format, Log: os.Stderr})
		}
		if err != nil {
			fmt.Fprintln(os.Stderr, "hwnpu-tts:", err)
			os.Exit(1)
		}
	}
	c.CacheDir, c.CPU, c.Log = *cacheDir, *cpu, os.Stderr
	if *seed == 0 {
		*seed = rand.Uint64() // drawn here so that it can be told (-seed repeats the speech)
	}
	if err := run(ctx, c,
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
	fmt.Fprintf(os.Stderr, "%s: %d frames, %.2f s of audio in %.1f s (seed %d)\n", out, len(frames), secs, time.Since(t0).Seconds(), r.Seed)
	return f.Close()
}
