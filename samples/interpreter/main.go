// interpreter is a sample app of the ai package: it interprets speech, from the microphone or
// a recording, into another language, as speech. Three models run one after another on the NPU
// for each utterance, joined as streams (ai/flow):
//
//	the microphone or a recording → utterances (VAD) → their text (Whisper) → its translation
//	(PLaMo 3 Translate) → its speech (LFM3-Audio)
//
// The stages overlap: the speaker speaks an utterance while the transcriber and the translator
// work on the next. The microphone is read through hwcamera (github.com/shibukawa/hwcamera,
// its audio package); a recording is the same kind of stream.
//
//	interpreter models                       # the models it uses and what they take
//	interpreter models prepare               # download, convert and compile them ahead
//	interpreter --mic -o english.wav         # from the microphone, until Ctrl-C
//	interpreter --mic --device ID            # from another input (interpreter mics lists them)
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
	"github.com/shibukawa/hwnpu/samples/capture"
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
	Run    RunCmd       `cmd:"" default:"withargs" help:"Interpret speech, from the microphone or a recording, into another language, as speech."`
	Mics   MicsCmd      `cmd:"" help:"The audio inputs --device takes."`
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

// MicsCmd is interpreter mics.
type MicsCmd struct{}

func (MicsCmd) Run(ctx context.Context) error {
	ins, err := capture.Mics(ctx)
	if err != nil {
		return err
	}
	if len(ins) == 0 {
		fmt.Println("no audio inputs")
	}
	for _, in := range ins {
		note := ""
		switch {
		case in.Default:
			note = " (the default)"
		case in.Monitor:
			note = " (what an output plays)"
		}
		fmt.Printf("%s\t%s%s\n", in.ID, in.Name, note)
	}
	return nil
}

// RunCmd is interpreter FILE.wav, and interpreter --mic.
type RunCmd struct {
	File     string        `arg:"" optional:"" type:"existingfile" name:"FILE.wav" help:"A recording of speech (or --mic)."`
	Mic      bool          `help:"Interpret what the microphone hears, until Ctrl-C (a second Ctrl-C ends at once)."`
	Device   string        `placeholder:"ID" help:"With --mic, the audio input (interpreter mics lists them); none: the system's default input."`
	Seconds  time.Duration `placeholder:"TIME" help:"With --mic, stop listening after this long (30s)."`
	Out      string        `short:"o" default:"out.wav" type:"path" placeholder:"FILE" help:"The WAV file of the interpretation."`
	To       string        `placeholder:"LANG" help:"The language to interpret into (en, ja, ...); none: English for Japanese speech, Japanese for any other."`
	Language string        `short:"l" placeholder:"LANG" help:"The language of the speech; none: found in each utterance."`
	Voice    string        `enum:"female,male" default:"female" help:"The voice of the interpretation."`
	Live     bool          `help:"Give the recording at the pace a microphone would, and time each stage against it."`
	Prepare  bool          `help:"Download, convert and compile the models that are not ready, as \"models prepare\" does, instead of failing."`
}

func (c *RunCmd) Run(ctx context.Context, app *ai.App) error {
	var pcm []float32
	switch {
	case c.Mic && c.File != "":
		return fmt.Errorf("--mic and a recording are two inputs: give one")
	case !c.Mic && c.File == "":
		return fmt.Errorf("give a recording (FILE.wav) or --mic")
	case !c.Mic:
		raw, err := os.ReadFile(c.File)
		if err != nil {
			return err
		}
		if pcm, err = speech.ReadWAV(raw, speech.TranscribeRate); err != nil {
			return fmt.Errorf("%s: %w", c.File, err)
		}
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

	var (
		mic     flow.Stream[[]float32]
		samples int
	)
	if c.Mic {
		// The microphone (hwcamera), until the first Ctrl-C or --seconds: the stream then ends and
		// the stages after it, which run under a context of their own, work off what was heard.
		// The second Ctrl-C ends the program as usual.
		listen := ctx
		if c.Seconds > 0 {
			var cancel context.CancelFunc
			listen, cancel = context.WithTimeout(ctx, c.Seconds)
			defer cancel()
		}
		mic = capture.Mic(listen, capture.MicOptions{Device: c.Device, Rate: speech.TranscribeRate, Lost: func() {
			fmt.Fprintln(os.Stderr, "sound was lost: the models are more than a minute behind the microphone")
		}})
		ctx = context.WithoutCancel(ctx)
		go func() {
			<-listen.Done()
			signal.Reset(os.Interrupt)
			fmt.Fprintln(os.Stderr, "the microphone is off; finishing what was heard")
		}()
		fmt.Fprintln(os.Stderr, "listening (Ctrl-C to stop)")
	} else {
		// the recording as a microphone's frames of 100 ms; with --live, at a microphone's pace
		mic = speech.Chunks(pcm, speech.TranscribeRate, speech.TranscribeRate/10)
		if c.Live {
			mic = flow.Live(ctx, mic)
		}
	}
	start := time.Now()
	mic = flow.Map(mic, func(c flow.Chunk[[]float32]) ([]float32, error) {
		samples += len(c.Data)
		return c.Data, nil
	})
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
	in := flow.Samples(samples, speech.TranscribeRate)
	fmt.Fprintf(os.Stderr, "%.1f s of speech interpreted into %.1f s in %.1f s: %s\n",
		in.Seconds(), flow.Samples(len(out), tts.Rate()).Seconds(), time.Since(start).Seconds(), c.Out)
	return nil
}

// hint adds how to prepare the models to the error of one that is not ready.
func hint(err error) error {
	return fmt.Errorf("%w (\"interpreter models prepare\" makes the models ready, or --prepare)", err)
}
