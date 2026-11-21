// interpreter is a sample app of the ai package: it interprets a recording of speech into
// another language, as speech. Three models run one after another on the NPU for each utterance,
// joined as streams (ai/flow):
//
//	the recording → utterances (VAD) → their text (Whisper) → its translation (PLaMo 3
//	Translate) → its speech (LFM3-Audio)
//
// The stages overlap: the speaker speaks an utterance while the transcriber and the translator
// work on the next.
//
//	interpreter models                       # the models it uses and what they take
//	interpreter models prepare               # download, convert and compile them ahead
//	interpreter talk.wav -o english.wav      # Japanese speech to English speech (or the reverse)
//	interpreter talk.wav --to ja --live      # paced as a microphone would give it
//	interpreter status                       # this machine's NPU
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"time"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/aicli"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/speech"
	"github.com/shibukawa/hwnpu/ai/text"
)

// The models the app uses: uses of the definitions, which pick a model for the machine's memory.
// The translator alone would be the 8B from 16 GB; beside the transcriber and the speaker (about
// 5 GB together) the app takes the 2B there and the 8B from 32 GB.
var (
	transcribe = speech.Transcribe("transcribe")
	translate  = text.Translate("translate").
			At(16, "plamo3translate:2b").
			At(32, "plamo3translate:8b")
	speak = speech.Speak("speech")

	app = ai.NewApp("interpreter", transcribe, translate, speak)
)

var cli struct {
	Run    RunCmd       `cmd:"" default:"withargs" help:"Interpret a recording of speech into another language, as speech."`
	Models aicli.Models `cmd:"" help:"The models interpreter uses: what they take, download them, compile them for this machine's NPU."`
	Status aicli.Status `cmd:"" help:"This machine's NPU: the device, how busy it is and what uses it."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	k := kong.Parse(&cli,
		kong.Name("interpreter"),
		kong.Description("Interpret speech into another language on the NPU: a sample of hwnpu's ai package."),
		kong.UsageOnError(),
		aicli.Bind(ctx, app))
	k.FatalIfErrorf(k.Run())
}

// RunCmd is interpreter FILE.wav.
type RunCmd struct {
	File     string `arg:"" type:"existingfile" name:"FILE.wav" help:"The recording of speech."`
	Out      string `short:"o" default:"out.wav" type:"path" placeholder:"FILE" help:"The WAV file of the interpretation."`
	To       string `placeholder:"LANG" help:"The language to interpret into (en, ja, ...); none: English for Japanese speech, Japanese for any other."`
	Language string `short:"l" placeholder:"LANG" help:"The language of the speech; none: found in each utterance."`
	Voice    string `enum:"female,male" default:"female" help:"The voice of the interpretation."`
	Live     bool   `help:"Give the recording at the pace a microphone would, and time each stage against it."`
	Prepare  bool   `help:"Download, convert and compile the models that are not ready, as \"models prepare\" does, instead of failing."`
}

func (c *RunCmd) Run(ctx context.Context, app *ai.App) error {
	raw, err := os.ReadFile(c.File)
	if err != nil {
		return err
	}
	pcm, err := speech.ReadWAV(raw, speech.TranscribeRate)
	if err != nil {
		return fmt.Errorf("%s: %w", c.File, err)
	}

	var opts []ai.Option
	if c.Prepare {
		opts = append(opts, ai.WithAutoPrepare(func(p ai.Progress) {
			if p.Message != "" {
				fmt.Fprintf(os.Stderr, "  %s: %s\n", p.Model, p.Message)
			}
		}))
	}
	rt, err := app.Open(ctx, opts...)
	if err != nil {
		return err
	}
	defer rt.Close()
	stt, err := transcribe.Get(ctx, rt)
	if err != nil {
		return hint(err)
	}
	tr, err := translate.Get(ctx, rt)
	if err != nil {
		return hint(err)
	}
	tts, err := speak.Get(ctx, rt)
	if err != nil {
		return hint(err)
	}
	for _, l := range rt.Loaded() {
		fmt.Fprintf(os.Stderr, "loaded %s (%s) in %.1f s\n", l.Model, l.Role, l.Took.Seconds())
	}

	// the recording as a microphone's frames of 100 ms; with --live, at a microphone's pace
	start := time.Now()
	mic := speech.Chunks(pcm, speech.TranscribeRate, speech.TranscribeRate/10)
	if c.Live {
		mic = flow.Live(ctx, mic)
	}
	utterances := speech.NewVAD(speech.VADOptions{}).Detect(ctx, mic)
	texts := stt.TranscribeStream(ctx, utterances, speech.TranscribeOptions{Language: c.Language})
	heard := flow.Map(texts, func(t flow.Chunk[flow.Text]) (flow.Text, error) {
		fmt.Printf("[%6.1fs] %s: %s\n", t.Start.Seconds(), t.Data.Lang, t.Data.Text)
		return t.Data, nil
	})
	translated := tr.TranslateStream(ctx, heard, text.TranslateOptions{To: c.To})
	said := flow.Map(translated, func(t flow.Chunk[flow.Text]) (flow.Text, error) {
		fmt.Printf("[%6.1fs] %s: %s\n", t.Start.Seconds(), t.Data.Lang, t.Data.Text)
		return t.Data, nil
	})
	// the speaker runs behind the other two by up to 4 utterances: they go on while it speaks
	audio := tts.SpeakStream(ctx, flow.Async(ctx, said, 4), speech.SpeakOptions{Voice: c.Voice})

	var out []float32
	for a, err := range audio {
		if err != nil {
			return err
		}
		out = append(out, a.Data...)
		fmt.Fprintf(os.Stderr, "[%6.1fs] %.1f s of speech\n", time.Since(start).Seconds(), a.Len.Seconds())
	}
	f, err := os.Create(c.Out)
	if err != nil {
		return err
	}
	if err := speech.WriteWAV(f, out, tts.Rate()); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	in := flow.Samples(len(pcm), speech.TranscribeRate)
	fmt.Fprintf(os.Stderr, "%.1f s of speech interpreted into %.1f s in %.1f s: %s\n",
		in.Seconds(), flow.Samples(len(out), tts.Rate()).Seconds(), time.Since(start).Seconds(), c.Out)
	return nil
}

// hint adds how to prepare the models to the error of one that is not ready.
func hint(err error) error {
	return fmt.Errorf("%w (\"interpreter models prepare\" makes the models ready, or --prepare)", err)
}
