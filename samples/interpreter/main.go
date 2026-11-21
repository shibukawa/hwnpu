// interpreter is a sample app of the ai package: it interprets speech, from the microphone or
// a recording, into another language, as speech. Its models and the program's own nodes are
// joined as a network (ai/flow), which the app declares and the library runs:
//
//	mic → vad → transcribe → heard → commands → translate → said → speech → out
//
// transcribe, translate and speech are models on the NPU (Whisper, PLaMo 3 Translate,
// LFM3-Audio). heard and said print what passes. commands is a node of the program: it hears
// spoken commands, which it does not pass on to be translated:
//
//	"変換、スペイン語" / "switch to Spanish"    the next utterances are interpreted into Spanish
//	"終わり" / "that's all"                     the interpreter ends, once what it heard is spoken
//
// The speaker runs behind the others (the edge into it keeps 4 texts), so the transcriber and
// the translator work on the next utterance while it speaks. The microphone is read through
// hwcamera (github.com/shibukawa/hwcamera, its audio package); a recording is the same kind of
// source.
//
// Beside the network, which keeps running on what its source gives, the app has a call, which
// runs once for a request (interpreter say): a text → translate → speech. The two share the
// models, each with nodes of its own.
//
//	interpreter models                       # the models and the network, and what they take
//	interpreter models prepare               # download, convert and compile the models ahead
//	interpreter run --mic                        # from the microphone to the speaker, until Ctrl-C or "終わり"
//	interpreter run --mic --device 2             # from another input (its number in "interpreter mics")
//	interpreter run talk.wav -o english.wav      # Japanese speech to English speech in a file (or the reverse)
//	interpreter run talk.wav --to ja --live      # paced as a microphone would give it
//	interpreter run talk.wav --speed 1.3         # said 30 % faster, its pitch kept
//	interpreter say こんにちは                 # a text, translated and spoken: one call
//	interpreter status                       # this machine's NPU
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/cli/konghelper"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/speech"
	"github.com/shibukawa/hwnpu/ai/text"
	"github.com/shibukawa/hwnpu/samples/internal/capture"
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
)

// network is the app's network: the nodes a run gives its input, its settings and its output,
// and what a run keeps.
type network struct {
	mic   *flow.Src[[]float32]
	hear  *ai.ModelNode[[]float32, flow.Text, speech.TranscribeOptions]
	trans *ai.ModelNode[flow.Text, flow.Text, text.TranslateOptions]
	say   *ai.ModelNode[flow.Text, []float32, speech.SpeakOptions]
	out   *flow.Snk[[]float32]

	// the call of "say": a text, translated and spoken, with nodes of its own on the same models
	sayTrans *ai.ModelNode[flow.Text, flow.Text, text.TranslateOptions]
	sayVoice *ai.ModelNode[flow.Text, []float32, speech.SpeakOptions]
	sayCall  *ai.Call[flow.Text, []float32]

	// of a run
	start    time.Time
	live     bool // the input comes in real time: the delays are shown
	commands bool
	mu       sync.Mutex
	ends     []time.Duration  // the ends of the utterances whose speech is to come
	pcm      []float32        // the interpretation
	rate     int              // its samples a second
	speed    float64          // how fast it is said (1: as the model says it)
	speaker  *capture.Speaker // when not nil, it is played as each utterance is spoken
}

// build declares the network and the app whose models are those of its nodes.
func build() (*ai.App, *network) {
	n := &network{
		mic:   flow.Source[[]float32]("mic"),
		hear:  speech.TranscribeNode(transcribe, speech.TranscribeOptions{}),
		trans: text.TranslateNode(translate, text.TranslateOptions{}),
		say:   speech.SpeakNode(speak, speech.SpeakOptions{}),
		out:   flow.Sink[[]float32]("out"),
	}
	vad := speech.VADNode(speech.VADOptions{})
	heard := flow.Tap("heard", n.print)
	commands := flow.Each("commands", n.command)
	said := flow.Tap("said", func(t flow.Chunk[flow.Text]) {
		n.print(t)
		n.mu.Lock()
		n.ends = append(n.ends, t.End())
		n.mu.Unlock()
	})
	n.out.Do(n.spoken)

	app := ai.NewApp("interpreter")
	// the network; the speaker runs behind the others by up to 4 utterances, which go on while it
	// speaks
	app.From(n.mic).To(vad).To(n.hear).To(heard).To(commands).To(n.trans).To(said).To(n.say, flow.Keep(4)).To(n.out)
	// the call
	n.sayTrans = text.TranslateNode(translate, text.TranslateOptions{})
	n.sayVoice = speech.SpeakNode(speak, speech.SpeakOptions{})
	n.sayCall = ai.NewCall[flow.Text, []float32](app, "say").Then(n.sayTrans).Then(n.sayVoice)
	return app, n
}

// after is, with a live input, how long after the end of an utterance a node gave its result
// (" +1.9s").
func (n *network) after(end time.Duration) string {
	if !n.live {
		return ""
	}
	return fmt.Sprintf(" +%.1fs", (time.Since(n.start) - end).Seconds())
}

// print shows a text that passes, with its utterance's start.
func (n *network) print(t flow.Chunk[flow.Text]) {
	fmt.Printf("[%6.1fs%s] %s: %s\n", t.Start.Seconds(), n.after(t.End()), t.Data.Lang, t.Data.Text)
}

// spoken keeps the speech of an utterance, at the speed asked, and plays it on the speaker when
// there is one (while the next utterances are transcribed and translated).
func (n *network) spoken(a flow.Chunk[[]float32]) error {
	pcm := capture.Stretch(a.Data, n.rate, n.speed)
	if n.speaker != nil {
		n.speaker.Play(pcm)
	}
	n.mu.Lock()
	n.pcm = append(n.pcm, pcm...)
	end := n.ends[0]
	n.ends = n.ends[1:]
	n.mu.Unlock()
	fmt.Fprintf(os.Stderr, "[%6.1fs%s] %.1f s of speech\n", time.Since(n.start).Seconds(), n.after(end), flow.Samples(len(pcm), n.rate).Seconds())
	return nil
}

// languages are the languages a spoken command names, by their names in Japanese and English.
var languages = map[string]string{
	"英語": "en", "日本語": "ja", "スペイン語": "es", "フランス語": "fr", "ドイツ語": "de", "イタリア語": "it",
	"ポルトガル語": "pt", "中国語": "zh", "韓国語": "ko",
	"english": "en", "japanese": "ja", "spanish": "es", "french": "fr", "german": "de", "italian": "it",
	"portuguese": "pt", "chinese": "zh", "korean": "ko",
}

// command is the node that hears spoken commands among the utterances. A command is a whole
// utterance: "終わり" ends the network (flow.Stop: the nodes before it stop, the microphone is
// closed, and the nodes after it speak what they were given), "変換、スペイン語" changes the
// language the translator's node translates into. Neither is passed on; everything else is.
func (n *network) command(c flow.Chunk[flow.Text], emit func(flow.Chunk[flow.Text])) error {
	if !n.commands {
		emit(c)
		return nil
	}
	// the words alone, without the punctuation and spaces a transcript has
	words := strings.ToLower(strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) || unicode.IsSpace(r) {
			return -1
		}
		return r
	}, c.Data.Text))
	switch words {
	case "終わり", "終わりです", "おわり", "thatsall", "stopinterpreting":
		fmt.Println("         (command: end)")
		return flow.Stop
	}
	for _, prefix := range []string{"変換", "switchto"} {
		if name, ok := strings.CutPrefix(words, prefix); ok {
			if to, ok := languages[name]; ok {
				n.trans.Update(func(o *text.TranslateOptions) { o.To = to })
				fmt.Printf("         (command: interpret into %s)\n", to)
				return nil
			}
		}
	}
	emit(c)
	return nil
}

var cli struct {
	Run    RunCmd            `cmd:"" default:"withargs" help:"Interpret speech, from the microphone or a recording, into another language, as speech."`
	Say    SayCmd            `cmd:"" help:"Translate a text and speak it: one call of the translator and the speaker."`
	Mics   MicsCmd           `cmd:"" help:"The audio inputs, numbered as --device takes them."`
	Models konghelper.Models `cmd:"" help:"The models interpreter uses and its network: what they take, download them, compile them for this machine's NPU."`
	Status konghelper.Status `cmd:"" help:"This machine's NPU: the device, how busy it is and what uses it."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	app, net := build()
	k := kong.Parse(&cli,
		kong.Name("interpreter"),
		kong.Description("Interpret speech into another language on the NPU: a sample of hwnpu's ai package."),
		kong.UsageOnError(),
		konghelper.Bind(ctx, app),
		kong.Bind(net))
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
	for i, in := range ins {
		note := ""
		switch {
		case in.Default:
			note = " (the default)"
		case in.Monitor:
			note = " (what an output plays)"
		}
		fmt.Printf("%d  %s%s\n", i+1, in.Name, note)
	}
	return nil
}

// SayCmd is interpreter say TEXT: the app's call, once.
type SayCmd struct {
	Text    []string `arg:"" name:"TEXT" help:"The text (the words are joined with spaces)."`
	Out     string   `short:"o" type:"path" placeholder:"FILE" help:"Write the speech into this WAV file; none: play it on the speaker."`
	Speed   float64  `default:"1" help:"How fast the speech is said, 0.5 to 2 (1.25: a quarter faster), its pitch kept."`
	To      string   `placeholder:"LANG" help:"The language to translate into (en, ja, es, ...); none: English for Japanese text, Japanese for any other."`
	Voice   string   `enum:"female,male" default:"female" help:"The voice."`
	Prepare bool     `help:"Download, convert and compile the models that are not ready instead of failing."`
}

func (c *SayCmd) Run(ctx context.Context, app *ai.App, n *network) error {
	if err := checkSpeed(c.Speed); err != nil {
		return err
	}
	var opts []ai.Option
	if c.Prepare {
		opts = append(opts, ai.WithAutoPrepare(nil))
	}
	rt, err := app.Open(ctx, opts...)
	if err != nil {
		return err
	}
	defer rt.Close()
	n.sayTrans.Update(func(o *text.TranslateOptions) { o.To = c.To })
	n.sayVoice.Update(func(o *speech.SpeakOptions) { o.Voice = c.Voice })
	start := time.Now()
	pcm, err := n.sayCall.Do(ctx, rt, flow.Text{Text: strings.Join(c.Text, " ")})
	if err != nil {
		return hint(err)
	}
	tts, err := speak.Get(ctx, rt) // loaded by the call: for its rate
	if err != nil {
		return err
	}
	pcm = capture.Stretch(pcm, tts.Rate(), c.Speed)
	fmt.Fprintf(os.Stderr, "%.1f s of speech in %.1f s (the models' loading included)\n",
		flow.Samples(len(pcm), tts.Rate()).Seconds(), time.Since(start).Seconds())
	return output(c.Out, pcm, tts.Rate())
}

// output writes the speech into the WAV file out, or with none plays it on the speaker and waits
// for its end.
func output(out string, pcm []float32, rate int) error {
	if out == "" {
		sp, err := capture.OpenSpeaker(rate)
		if err != nil {
			return fmt.Errorf("%w (-o FILE writes the speech into a file instead)", err)
		}
		sp.Play(pcm)
		sp.Wait()
		return nil
	}
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	if err := speech.WriteWAV(f, pcm, rate); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %s\n", out)
	return nil
}

// checkSpeed refuses a speed past what the stretch keeps intelligible.
func checkSpeed(speed float64) error {
	if speed < 0.5 || speed > 2 {
		return fmt.Errorf("--speed %g: 0.5 to 2", speed)
	}
	return nil
}

// RunCmd is interpreter run FILE.wav, and interpreter run --mic.
type RunCmd struct {
	File     string        `arg:"" optional:"" type:"existingfile" name:"FILE.wav" help:"A recording of speech (or --mic)."`
	Mic      bool          `help:"Interpret what the microphone hears, until Ctrl-C (a second Ctrl-C ends at once) or the spoken command."`
	Device   string        `placeholder:"N" help:"With --mic, the audio input: its number in \"interpreter mics\" (or its ID); none: the system's default input."`
	Seconds  time.Duration `placeholder:"TIME" help:"With --mic, stop listening after this long (30s)."`
	Out      string        `short:"o" type:"path" placeholder:"FILE" help:"Write the interpretation into this WAV file; none: play it on the speaker, each utterance as soon as it is spoken."`
	Speed    float64       `default:"1" help:"How fast the interpretation is said, 0.5 to 2 (1.25: a quarter faster), its pitch kept."`
	To       string        `placeholder:"LANG" help:"The language to interpret into (en, ja, es, ...); none: English for Japanese speech, Japanese for any other."`
	Language string        `short:"l" placeholder:"LANG" help:"The language of the speech; none: found in each utterance."`
	Voice    string        `enum:"female,male" default:"female" help:"The voice of the interpretation."`
	Live     bool          `help:"Give the recording at the pace a microphone would, and time each node against it."`
	Commands bool          `default:"true" negatable:"" help:"Hear the spoken commands: \"終わり\" (or \"that's all\") ends, \"変換、スペイン語\" (or \"switch to Spanish\") changes the language to interpret into."`
	Prepare  bool          `help:"Download, convert and compile the models that are not ready, as \"models prepare\" does, instead of failing."`
}

func (c *RunCmd) Run(ctx context.Context, app *ai.App, n *network) error {
	if err := checkSpeed(c.Speed); err != nil {
		return err
	}
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
	// loaded here, to say so before listening (the run would load them too)
	if _, err := transcribe.Get(ctx, rt); err != nil {
		return hint(err)
	}
	if _, err := translate.Get(ctx, rt); err != nil {
		return hint(err)
	}
	tts, err := speak.Get(ctx, rt)
	if err != nil {
		return hint(err)
	}
	for _, l := range rt.Loaded() {
		fmt.Fprintf(os.Stderr, "loaded %s (%s) in %.1f s\n", l.Model, l.Role, l.Took.Seconds())
	}

	// the flags are the nodes' settings
	n.hear.Update(func(o *speech.TranscribeOptions) { o.Language = c.Language })
	n.trans.Update(func(o *text.TranslateOptions) { o.To = c.To })
	n.say.Update(func(o *speech.SpeakOptions) { o.Voice = c.Voice })
	n.live, n.commands = c.Mic || c.Live, c.Commands
	n.rate, n.speed = tts.Rate(), c.Speed
	if c.Out == "" {
		if n.speaker, err = capture.OpenSpeaker(tts.Rate()); err != nil {
			return fmt.Errorf("%w (-o FILE writes the interpretation into a file instead)", err)
		}
	}

	// the source's stream, counted as it passes
	var (
		mic     flow.Stream[[]float32]
		samples int
	)
	if c.Mic {
		// The microphone (hwcamera), until the first Ctrl-C or --seconds: the source then ends and
		// the nodes after it, which run under a context of their own, work off what was heard.
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
		over := make(chan struct{}) // the run has ended (by a spoken command, say)
		defer close(over)
		go func() {
			select {
			case <-listen.Done():
				signal.Reset(os.Interrupt)
				fmt.Fprintln(os.Stderr, "the microphone is off; finishing what was heard")
			case <-over:
			}
		}()
		fmt.Fprintln(os.Stderr, "listening (Ctrl-C to stop)")
	} else {
		// the recording as a microphone's frames of 100 ms; with --live, at a microphone's pace
		mic = speech.Chunks(pcm, speech.TranscribeRate, speech.TranscribeRate/10)
		if c.Live {
			mic = flow.Live(ctx, mic)
		}
	}
	n.mic.From(flow.Map(mic, func(c flow.Chunk[[]float32]) ([]float32, error) {
		samples += len(c.Data)
		return c.Data, nil
	}))

	n.start = time.Now()
	err = rt.Run(ctx)
	took := time.Since(n.start) // the interpretation's, not the speaker's last words
	if n.speaker != nil {
		n.speaker.Wait() // what was spoken, played to its end
	}
	if err != nil {
		return err
	}
	in := flow.Samples(samples, speech.TranscribeRate)
	fmt.Fprintf(os.Stderr, "%.1f s of speech interpreted into %.1f s in %.1f s\n",
		in.Seconds(), flow.Samples(len(n.pcm), tts.Rate()).Seconds(), took.Seconds())
	if c.Out == "" {
		return nil
	}
	return output(c.Out, n.pcm, tts.Rate())
}

// hint adds how to prepare the models to the error of one that is not ready.
func hint(err error) error {
	return fmt.Errorf("%w (\"interpreter models prepare\" makes the models ready, or --prepare)", err)
}
