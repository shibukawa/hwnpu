// voice-agent is a sample app of the ai package: a small voice assistant, called by its name
// ("コダマ、今何時？", "ねえコダマ、明日の天気は？"), that answers aloud as soon as it can. It tells
// the time and the date, sets timers, reads the weather (Open-Meteo) and the news (NHK's RSS),
// and looks things up in Wikipedia; other questions the chat model answers itself.
//
//	mic → vad → transcribe → name? → shortcut (time, timers) ─────────→ speech → speaker
//	                                └→ chat model ⇄ tools → sentences ─┘
//
// The models are on the NPU: Whisper (transcribe), a chat model with tools (LFM3-1.7B), and
// LFM3-Audio (speech). The assistant is ai/agent (the conversation and its tool calls) and
// ai/agent/voice (hearing and speaking, and how the answer comes soon: the transcription and
// the model start at a pause, before the end of the utterance is certain; the answer is spoken
// a piece at a time). This program gives them what is its own: the system prompt and what it
// knows of the user (prompt.go), the tools (tools.go, timers.go), and the requests answered
// without the model (shortcut.go).
//
//	voice-agent models prepare                # download, convert and compile the models ahead
//	voice-agent                               # listen to the microphone until Ctrl-C
//	voice-agent run --device 2                # another input (its number in "voice-agent mics")
//	voice-agent run talk.wav                  # a recording, as a microphone would give it
//	voice-agent ask 明日の東京の天気は？        # one request as text, answered aloud
//	voice-agent ask --text 今日のニュース       # ... and in text only (the chat model alone)
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/agent"
	"github.com/shibukawa/hwnpu/ai/agent/voice"
	"github.com/shibukawa/hwnpu/ai/cli/konghelper"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/speech"
	"github.com/shibukawa/hwnpu/ai/text"
	"github.com/shibukawa/hwnpu/samples/internal/capture"
)

// The models: the transcriber, the chat model and the speaker. The chat model is picked as a
// machine of 8 GB would (LFM3-1.7B): a larger one answers better and later, and speed is what
// a spoken answer wants. models.yaml's aliases put another one in its place (docs/model-table.md).
var (
	transcribe = speech.Transcribe("transcribe")
	chatModel  = text.Chat("chat").As(8)
	speak      = speech.Speak("supertonic-3")
)

// names are the ways Whisper writes the default name.
var names = []string{"コダマ", "こだま", "木霊", "谺", "小玉", "児玉", "子玉", "kodama"}

var cli struct {
	Run    RunCmd            `cmd:"" default:"withargs" help:"Listen to the microphone (or a recording) and answer what is asked after the name."`
	Ask    AskCmd            `cmd:"" help:"Answer one request given as text."`
	Mics   MicsCmd           `cmd:"" help:"The audio inputs, numbered as --device takes them."`
	Models konghelper.Models `cmd:"" help:"The models voice-agent uses: what they take, download them, compile them for this machine's NPU."`
	Status konghelper.Status `cmd:"" help:"This machine's NPU: the device, how busy it is and what uses it."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	go func() { // the first Ctrl-C stops; a second ends at once
		<-ctx.Done()
		stop()
	}()
	app := ai.NewApp("voice-agent", transcribe, chatModel, speak)
	k := kong.Parse(&cli,
		kong.Name("voice-agent"),
		kong.Description("A small voice assistant on the NPU, called by its name: a sample of hwnpu's ai package."),
		kong.UsageOnError(),
		konghelper.Bind(ctx, app))
	k.FatalIfErrorf(k.Run())
}

func logf(format string, args ...any) { fmt.Fprintf(os.Stderr, format+"\n", args...) }

// indented logs what the agent, the voice and the listener do, under what was heard and said.
func indented(format string, args ...any) { fmt.Fprintf(os.Stderr, "  ("+format+")\n", args...) }

// common are the settings of the assistant, for run and ask.
type common struct {
	Place   string        `default:"東京" help:"Where the user is (a city or a prefecture): the weather's place when a request names none."`
	User    string        `placeholder:"NAME" help:"How the assistant calls the user (none: it does not)."`
	About   string        `placeholder:"TEXT" help:"What the assistant knows of the user from the start, in a sentence or two (\"家族は妻と娘。経済のニュースに関心がある\")."`
	Feed    string        `default:"https://www.nhk.or.jp/rss/news/%s.xml" placeholder:"URL" help:"The news feed (RSS), with %s for NHK's category (cat0 the main news)."`
	Voice   string        `default:"female" help:"The voice: female or male, or Supertonic's F1 ... F5, M1 ... M5."`
	Speed   float64       `default:"1.1" help:"How fast the answers are said, as a multiple of the voice's own pace: 0.7 to 1.9 (0.5 to 2 for a model stretched to it)."`
	Seed    uint64        `default:"7" help:"The seed of the voice, so that each sentence sounds the same (0: a new one each time)."`
	Reset   time.Duration `default:"5m" help:"Forget the conversation after this long without a request."`
	History int           `default:"16" help:"The most messages of earlier turns kept."`
}

// assistant is what run and ask build: the agent, its tools and its voice.
type assistant struct {
	rt     *ai.Runtime
	agent  *agent.Agent
	tools  *Tools
	timers *Timers
	voice  *voice.Voice // nil without speech
}

// open loads the models (the speaker too unless textOnly) and makes the assistant.
func (c common) open(ctx context.Context, app *ai.App, textOnly bool) (*assistant, error) {
	if c.Speed < 0.5 || c.Speed > 2 {
		return nil, fmt.Errorf("--speed %g: 0.5 to 2", c.Speed)
	}
	rt, err := konghelper.Open(ctx, app)
	if err != nil {
		return nil, err
	}
	a := &assistant{rt: rt, timers: NewTimers()}
	chat, err := chatModel.Get(ctx, rt)
	if err != nil {
		rt.Close()
		return nil, err
	}
	a.tools = &Tools{Timers: a.timers, Place: c.Place, Feed: c.Feed}
	a.agent = agent.New(chat, agent.Config{
		System:      Profile{Name: c.User, Place: c.Place, About: c.About}.prompt(),
		Tools:       a.tools.List(),
		ResetAfter:  c.Reset,
		MaxMessages: c.History,
		Log:         indented,
	})
	if !textOnly {
		tts, err := speak.Get(ctx, rt)
		if err != nil {
			rt.Close()
			return nil, err
		}
		sp, err := capture.OpenSpeaker(tts.Rate())
		if err != nil {
			rt.Close()
			return nil, err
		}
		dir := ""
		if d, err := os.UserCacheDir(); err == nil {
			dir = filepath.Join(d, "hwnpu", "voice-agent", "phrases")
		}
		o := voice.VoiceOptions{Speak: speech.SpeakOptions{Language: "ja", Voice: c.Voice, Seed: c.Seed}, Dir: dir, Log: indented}
		if speed := c.Speed; tts.Paces() {
			o.Speak.Speed = speed // Supertonic says it at that pace
		} else {
			o.Process = func(pcm []float32, rate int) []float32 { return capture.Stretch(pcm, rate, speed) }
		}
		a.voice = voice.NewVoice(tts, sp, o)
		// a timer's ring: the alarm, then what it was (synthesized when it was set)
		a.timers.Added = func(t *Timer) { go a.voice.Phrase(ctx, t.Name()+"の時間です。") }
		a.timers.Ring = func(t *Timer) {
			logf("timer: %s", t.Name())
			a.voice.Play(a.voice.Alarm())
			a.voice.SayPhrase(ctx, t.Name()+"の時間です。")
		}
	}
	for _, l := range rt.Loaded() {
		logf("loaded %s (%s) in %.1f s", l.Model, l.Role, l.Took.Seconds())
	}
	return a, nil
}

// phrases are said often: synthesized at the start (the first time; then read from the disk).
var phrases = []string{waiting, sorry}

const (
	waiting = "ちょっと待ってね。"
	sorry   = "ごめんなさい、うまく答えられませんでした。"
)

// RunCmd is voice-agent run.
type RunCmd struct {
	common
	File      string        `arg:"" optional:"" type:"existingfile" name:"FILE.wav" help:"A recording to hear in place of the microphone, at the pace it would come from one."`
	Device    string        `placeholder:"N" help:"The audio input: its number in \"voice-agent mics\" (or its ID); none: the system's default input."`
	Name      []string      `placeholder:"NAME,..." help:"The assistant's name, as Whisper may write it (default: コダマ and its spellings)."`
	Follow    time.Duration `default:"8s" help:"For this long after the assistant has spoken, a request needs no name."`
	Pause     time.Duration `default:"250ms" help:"The silence after which what was said is transcribed and answered, before the end of the utterance is certain."`
	Speculate bool          `default:"true" negatable:"" help:"Start the model on what was said at a pause (its speech waits for the end of the utterance)."`
	Shortcuts bool          `default:"true" negatable:"" help:"Answer the time, the date and timers without the model."`
	BargeIn   bool          `name:"barge-in" help:"Keep hearing while the assistant speaks, so that its name stops it (with headphones: else it hears itself, which also slows its speech)."`
	Language  string        `short:"l" default:"ja" placeholder:"LANG" help:"The language of the speech (empty: found in each utterance)."`
}

func (c *RunCmd) Run(ctx context.Context, app *ai.App) error {
	var audio flow.Stream[[]float32]
	if c.File != "" {
		raw, err := os.ReadFile(c.File)
		if err != nil {
			return err
		}
		pcm, err := speech.ReadWAV(raw, speech.TranscribeRate)
		if err != nil {
			return fmt.Errorf("%s: %w", c.File, err)
		}
		// the recording as a microphone's frames of 20 ms, at their pace, then a second of
		// silence, so that its last utterance ends
		pcm = append(pcm, make([]float32, speech.TranscribeRate)...)
		audio = flow.Live(ctx, speech.Chunks(pcm, speech.TranscribeRate, speech.TranscribeRate/50))
	}
	a, err := c.open(ctx, app, false)
	if err != nil {
		return err
	}
	defer a.rt.Close()
	stt, err := transcribe.Get(ctx, a.rt)
	if err != nil {
		return err
	}
	if err := prepare(ctx, a); err != nil {
		return err
	}
	names := names
	if len(c.Name) > 0 {
		names = c.Name
	}
	c2 := voice.Config{
		Transcriber: stt,
		Transcribe:  speech.TranscribeOptions{Language: c.Language},
		Agent:       a.agent,
		Voice:       a.voice,
		Wake:        voice.NewWake(names...),
		Pause:       c.Pause,
		Follow:      c.Follow,
		Speculate:   c.Speculate,
		BargeIn:     c.BargeIn,
		Waiting:     waiting,
		Sorry:       sorry,
		Log:         indented,
		Events: voice.Events{
			Heard: func(text string, after time.Duration, early bool) {
				at := ""
				if early {
					at = ", transcribed at the pause"
				}
				logf("heard: %s  (%.2f s after the speech%s)", text, after.Seconds(), at)
			},
			Answer: func(req, ans string, shortcut bool) {
				by := ""
				if shortcut {
					by = " (without the model)"
				}
				logf("answer%s: %s", by, ans)
			},
			FirstSound: func(after time.Duration) {
				indented("the answer's first sound %.2f s after the speech ended", after.Seconds())
			},
			Failed: func(req string, err error) { logf("answering %q: %v", req, err) },
		},
	}
	if c.Shortcuts {
		sh := &Shortcuts{Timers: a.timers, Reset: func() {
			a.agent.Reset()
			go a.agent.Prime(ctx) // the system prompt read again while the answer is said
		}}
		c2.Shortcut = sh.Func()
	}
	l := voice.NewListener(c2)
	if audio == nil {
		audio = capture.Mic(ctx, capture.MicOptions{Device: c.Device, Rate: speech.TranscribeRate, Lost: func() {
			logf("sound was lost: the assistant is more than a minute behind the microphone")
		}})
		logf("listening: call it by its name (\"%s、今何時？\"); Ctrl-C to stop", names[0])
	}
	return l.Run(ctx, audio)
}

// prepare reads the system prompt into the model's cache and makes the phrases ready, before
// the assistant listens.
func prepare(ctx context.Context, a *assistant) error {
	t0 := time.Now()
	if err := a.agent.Prime(ctx); err != nil {
		return err
	}
	if a.voice != nil {
		for _, p := range phrases {
			if _, err := a.voice.Phrase(ctx, p); err != nil {
				return err
			}
		}
	}
	logf("ready in %.1f s (the system prompt read, the phrases made)", time.Since(t0).Seconds())
	return nil
}

// AskCmd is voice-agent ask TEXT.
type AskCmd struct {
	common
	Text     []string `arg:"" name:"TEXT" help:"The request (the words are joined with spaces)."`
	TextOnly bool     `name:"text" help:"Print the answer without speaking it (the chat model alone is loaded)."`
}

func (c *AskCmd) Run(ctx context.Context, app *ai.App) error {
	a, err := c.open(ctx, app, c.TextOnly)
	if err != nil {
		return err
	}
	defer a.rt.Close()
	req := strings.Join(c.Text, " ")
	split := voice.Splitter{Out: func(s string) { fmt.Fprintf(os.Stderr, "  > %s\n", s) }}
	o := agent.TurnOptions{Text: split.Add}
	if a.voice != nil {
		split.Out = a.voice.Say
		o.Tool = func(name, args string, slow bool) {
			if slow {
				a.voice.SayPhrase(ctx, waiting)
			}
		}
	}
	if act, ok := (&Shortcuts{Timers: a.timers}).Match(req); ok {
		ans := act.Run()
		fmt.Println(ans)
		if a.voice != nil {
			a.voice.Say(ans)
			a.voice.Wait(ctx)
		}
		return nil
	}
	t0 := time.Now()
	ans, err := a.agent.Turn(ctx, req, o)
	if err != nil {
		return err
	}
	split.Flush()
	fmt.Println(ans)
	logf("answered in %.1f s", time.Since(t0).Seconds())
	if a.voice != nil {
		a.voice.Wait(ctx)
	}
	return nil
}

// MicsCmd is voice-agent mics.
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
		if in.Default {
			note = " (the default)"
		}
		fmt.Printf("%d  %s%s\n", i+1, in.Name, note)
	}
	return nil
}
