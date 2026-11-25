package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/shibukawa/bonsaiai/ai"
	"github.com/shibukawa/bonsaiai/ai/speech"
	"github.com/shibukawa/bonsaiai/internal/console"
	ispeech "github.com/shibukawa/bonsaiai/internal/speech"
)

// TTSCmd is "bonsaiai-audio tts": speaks a text with LFM3-Audio-2B on the Neural Engine into a WAV
// file (24 kHz). The first run downloads LFM3-Audio-2B (its download is shared with
// bonsaiai-chat's LFM3-Audio-2B) and converts the programs by itself (internal/speech Prepare).
// Programs converted by hand are given with --checkpoint, --text, --depth and --codec instead:
//
//	bonsaiai-dev convert -backend coreai -config DIR/config.json -weights DIR/model.safetensors -seq 1 -cache 2048 -hidden-out 1 -o DIR/tts-q1.aimodel
//	bonsaiai-dev convert -backend coreai -config DIR/config.json -weights DIR/model.safetensors -seq 64 -cache 2048 -hidden-out 1 -o DIR/tts-q64.aimodel
//	bonsaiai-dev convert -backend coreai -arch lfm3-depthformer -config DIR/config.json -weights DIR/model.safetensors -format fp16 -o DIR/depth.aimodel
//	bonsaiai-dev convert -backend coreai -config DIR/audio_tokenizer/config.json -weights DIR/audio_tokenizer/model.safetensors -codec-frames 64 -format fp16 -o DIR/codec.aimodel
//	bonsaiai-audio tts --checkpoint DIR --text DIR/tts-q1.parts.json,DIR/tts-q64.aimodel --depth DIR/depth.aimodel --codec DIR/codec.aimodel --lang ja -o out.wav こんにちは
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
	Model     string   `short:"m" default:"speech" placeholder:"NAME" help:"The model: a name, a use or a family of models.yaml (speech, lfm3-audio-tts:2b), a Hugging Face repository OWNER/NAME of an LFM3-Audio checkpoint, or on AMD's NPU an export's directory (tools/amd/iron_lfm3speech.py)."`
	Out       string   `short:"o" default:"out.wav" type:"path" placeholder:"FILE" help:"The WAV file to write."`
	Lang      string   `default:"en" placeholder:"LANG" help:"The language: a code of the model card (en, en-gb, ja, zh, ...) or a language's name."`
	Voice     string   `default:"female" help:"The voice: female or male; Supertonic's by its name too (F1 ... F5, M1 ... M5)."`
	Seed      uint64   `placeholder:"N" help:"The random seed of the speech codes (0: a new one, logged so that it can be repeated)."`
	MaxFrames int      `name:"max-frames" placeholder:"N" help:"Most audio frames of 80 ms (0: until the end of audio or a full cache)."`
	Format    string   `placeholder:"FORMAT" help:"Weight format of the text model's programs (default: the model's, 8 bit tables)."`
	Models    string   `placeholder:"FILES" help:"${models_usage}."`
	CPU       bool     `name:"cpu" help:"Run on the CPU interpreter instead of the Neural Engine (slow)."`

	Checkpoint string `group:"Converted by hand" type:"existingdir" placeholder:"DIR" help:"The checkpoint's directory (config.json, model.safetensors, tokenizer.json, audio_tokenizer/config.json) of the programs below."`
	TextModel  string `group:"Converted by hand" name:"text" placeholder:"PKG,..." help:"The text model converted with -cache and -hidden-out 1, packages of different -seq, comma separated."`
	Depth      string `group:"Converted by hand" placeholder:"PKG" help:"The depthformer (bonsaiai-dev convert -arch lfm3-depthformer)."`
	Codec      string `group:"Converted by hand" placeholder:"PKG" help:"The codec decoder (bonsaiai-dev convert on audio_tokenizer/, -arch lfm3-codec)."`
	IronText   string `group:"Converted by hand" name:"iron-text" type:"existingdir" placeholder:"DIR" help:"On AMD's NPU, the text model's export of iron_export_lm.py lfm3tts in place of --text (--depth and --codec graphs of tools/amd/iron_lfm3speech.py, or Core ML packages run on the CPU)."`
}

// Run speaks the text.
func (c *TTSCmd) Run(ctx context.Context) error {
	lw := console.Log(os.Stderr)
	st := store(c.Models)
	use, defs, err := c.use(st)
	if err != nil {
		return err
	}
	st.Definitions = defs
	app := ai.NewApp("bonsaiai-audio", use)
	app.Store = st
	rt, err := open(ctx, app, lw)
	if err != nil {
		return err
	}
	defer rt.Close()
	s, err := use.Get(ctx, rt)
	if err != nil {
		return err
	}
	console.Running()
	a, err := s.Speak(ctx, strings.Join(c.Text, " "), speech.SpeakOptions{Language: c.Lang, Voice: c.Voice, Seed: c.Seed, MaxFrames: c.MaxFrames})
	if err != nil {
		return err
	}
	f, err := os.Create(c.Out)
	if err != nil {
		return err
	}
	if err := speech.WriteWAV(f, a.Samples, a.Rate); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	secs, t := float64(len(a.Samples))/float64(a.Rate), a.Stats
	n1, n2, n3 := s.Stages()
	fmt.Fprintf(lw, "%d frames, %.2f s of audio in %.2f s, %.1fx real time (seed %d; %s %.2f s, %s %.2f s, %s %.2f s)\n",
		t.Frames, secs, t.Took.Seconds(), secs/t.Took.Seconds(), t.Seed, n1, t.Text.Seconds(), n2, t.Depth.Seconds(), n3, t.Codec.Seconds())
	fmt.Println(c.Out)
	return nil
}

// use is the use of the model --model names, as the flags make it: a model of its own for
// --format, --cpu, the programs converted by hand, or an export's directory.
func (c *TTSCmd) use(st ai.Store) (ai.Use[*speech.Speaker], []ai.Definition, error) {
	var d ai.Definition
	name := c.Model
	manual := c.Checkpoint != "" || c.TextModel != "" || c.IronText != "" || c.Depth != "" || c.Codec != ""
	switch dir := expandHome(c.Model); {
	case manual:
		if c.Checkpoint == "" || c.TextModel == "" && c.IronText == "" || c.Depth == "" || c.Codec == "" {
			return ai.Use[*speech.Speaker]{}, nil, fmt.Errorf("programs converted by hand take all of --checkpoint, --text (or --iron-text), --depth and --codec")
		}
		d.Programs = map[string][]string{"checkpoint": {absolute(c.Checkpoint)}, "depth": {absolute(c.Depth)}, "codec": {absolute(c.Codec)}}
		if c.TextModel != "" {
			for _, p := range strings.Split(c.TextModel, ",") {
				d.Programs["text"] = append(d.Programs["text"], absolute(p))
			}
		}
		if c.IronText != "" {
			d.Programs["iron-text"] = []string{absolute(c.IronText)}
		}
	case isSpeechExport(dir):
		// an export of tools/amd/iron_lfm3speech.py, for AMD's NPU: a model like the definitions' own
		d.Programs = map[string][]string{"iron": {absolute(dir)}}
		name = "speech"
	}
	use, defs, err := useFor(st, speech.Speak(name))
	if err != nil {
		return use, nil, err
	}
	if c.Format != "" {
		d.Settings = map[string]string{"format": c.Format}
	}
	if c.CPU {
		d.Device = "cpu"
	}
	if d.Programs == nil && d.Settings == nil && d.Device == "" {
		return use, defs, nil
	}
	d.Like = use.Name()
	return speech.Speak(d.ID()), append(defs, d), nil
}

// isSpeechExport reports whether dir is an export of tools/amd/iron_lfm3speech.py.
func isSpeechExport(dir string) bool { return ispeech.IsIronExport(dir) }
