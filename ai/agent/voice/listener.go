// Package voice makes a voice assistant of an agent (ai/agent): it hears requests in a
// microphone's audio, called by a name, and speaks the answers, as soon as it can.
//
//	stt, err := transcribe.Get(ctx, rt) // speech.Transcribe("transcribe")
//	tts, err := speak.Get(ctx, rt)      // speech.Speak("speech")
//	a := agent.New(chat, agent.Config{System: "...", Tools: tools})
//	v := voice.NewVoice(tts, player, voice.VoiceOptions{Speak: speech.SpeakOptions{Language: "ja"}})
//	l := voice.NewListener(voice.Config{Transcriber: stt, Agent: a, Voice: v, Wake: voice.NewWake("コダマ", "こだま")})
//	err = l.Run(ctx, mic) // mono audio at 16 kHz, as ai/speech's VAD reads it
//
// With text, the request is over when it has been sent; with speech, it is over when the
// speaker has been silent long enough, and the answer should come soon after that. The
// Listener's ways of making it come soon:
//
//   - A pause (Pause, 250 ms) starts the transcription of what was said, before the VAD has
//     decided that the utterance ended (its Hangover, 700 ms): when it does, with no speech
//     since, that transcript is the utterance's, and the transcriber's time is not waited for.
//   - The agent starts on that transcript at once (Speculate): the model reads the request and
//     writes its answer while the VAD waits, and nothing is said and no tool runs before the
//     utterance has ended. When the speaker goes on, the turn is cancelled, and the
//     conversation is as it was.
//   - The requests a Shortcut answers (the time, timers) need no model.
//   - The answer is spoken a piece at a time (Splitter): the first piece is cut short.
//   - The phrases said often are ready-made (Voice.Phrase), and a word of waiting is said
//     before a slow tool.
//
// A request starts with the name (Wake); for Follow after the assistant has spoken, the next
// one needs no name. The name alone is answered with a chime and opens the same window.
package voice

import (
	"context"
	"errors"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/bonsaiai/ai/agent"
	"github.com/shibukawa/bonsaiai/ai/flow"
	"github.com/shibukawa/bonsaiai/ai/speech"
)

// Transcriber turns speech into text: *speech.Transcriber (Whisper), or another model.
type Transcriber interface {
	Transcribe(ctx context.Context, pcm []float32, o speech.TranscribeOptions) (speech.Transcript, error)
}

// Shortcut answers a request without the model, when it knows it: run makes its change once
// the request is certain and returns what to say. ok is false for a request it does not know.
type Shortcut func(request string) (run func() string, ok bool)

// Config is how a Listener hears and answers. The zero values of its fields take their
// defaults.
type Config struct {
	Transcriber Transcriber
	// Transcribe are the transcriber's settings (the language: "ja" saves finding it).
	Transcribe speech.TranscribeOptions
	Agent      *agent.Agent
	Voice      *Voice
	// Wake finds the name in what was heard.
	Wake *Wake
	// Shortcut answers the requests it knows without the model (nil: every request goes to
	// it).
	Shortcut Shortcut
	// VAD finds the utterances (default: speech's, with a Hangover of 700 ms).
	VAD speech.VADOptions
	// Pause is the silence after which what was said is transcribed ahead of the VAD's end
	// (default 250 ms).
	Pause time.Duration
	// Follow is how long after the assistant has spoken a request needs no name (default 8 s;
	// below 0, never).
	Follow time.Duration
	// Speculate starts the agent on the transcript of a pause.
	Speculate bool
	// BargeIn keeps hearing while the assistant speaks, for its name, which stops it. Without
	// it, the audio is not heard while the assistant speaks and for Echo after (its own voice
	// would be heard, and transcribing it takes the NPU from the speech).
	BargeIn bool
	// Echo is how long after the assistant has spoken the audio is still not heard (default
	// 300 ms).
	Echo time.Duration
	// Waiting is the phrase said before a slow tool when nothing has been said ("" for none);
	// Sorry the one said when a turn fails ("" for none).
	Waiting, Sorry string
	// Split cuts the answers into the pieces spoken (its Out is the Listener's).
	Split Splitter
	// Events are told what is heard and answered.
	Events Events
	// Log receives what the Listener does, for a look at what it took (nil for none).
	Log func(format string, args ...any)
}

// Events are told what is heard and answered. Any may be nil. They are called in the
// Listener's goroutine: they should return soon.
type Events struct {
	// Heard is an utterance's transcript, after the end of its speech; early says that it was
	// transcribed at a pause.
	Heard func(transcript string, after time.Duration, early bool)
	// Answer is the answer of a request; shortcut says that the model did not make it.
	Answer func(request, answer string, shortcut bool)
	// FirstSound is the time from the end of a request's speech to its answer's first sound.
	FirstSound func(after time.Duration)
	// Failed is a turn that failed.
	Failed func(request string, err error)
}

// Listener hears requests and has them answered.
type Listener struct {
	c Config

	following bool  // the assistant has answered: Follow counts from when it stopped speaking
	turn      *turn // the turn running (nil for none)
	utt       *utterance
	pending   int // transcriptions running
	events    chan func()
}

// NewListener is a listener of the config.
func NewListener(c Config) *Listener {
	if c.VAD.Hangover == 0 {
		c.VAD.Hangover = 700 * time.Millisecond
	}
	if c.Pause <= 0 {
		c.Pause = 250 * time.Millisecond
	}
	if c.Follow == 0 {
		c.Follow = 8 * time.Second
	}
	if c.Echo <= 0 {
		c.Echo = 300 * time.Millisecond
	}
	if c.Log == nil {
		c.Log = func(string, ...any) {}
	}
	return &Listener{c: c}
}

// utterance is what is heard of an utterance as it comes.
type utterance struct {
	pcm       []float32
	last      time.Time   // the wall clock at its last chunk of speech
	addressed bool        // it began while a request needs no name
	early     *transcript // the transcription begun at the last pause
	final     *transcript // the utterance's, once the VAD's final chunk has come
}

// transcript is a transcription of an utterance so far, or whole.
type transcript struct {
	n    int // the samples transcribed
	done bool
	text string
	err  error
	spec *turn // the turn started on it (Speculate)
}

// turn is a turn of the agent, running.
type turn struct {
	request string
	cancel  context.CancelFunc
	gate    chan struct{}
	opened  bool
}

func (t *turn) confirm() {
	if !t.opened {
		t.opened = true
		close(t.gate)
	}
}

// Run hears the audio (mono, at the VAD's rate, 16 kHz) until it ends or ctx is cancelled, and
// answers what is asked. At the end of the audio it waits for the answers to be spoken.
func (l *Listener) Run(ctx context.Context, audio flow.Stream[[]float32]) error {
	l.events = make(chan func(), 16)
	chunks := make(chan flow.Chunk[[]float32], 256)
	errc := make(chan error, 1)
	go func() {
		defer close(chunks)
		for c, err := range speech.NewVAD(l.c.VAD).Detect(ctx, l.muted(audio)) {
			if err != nil {
				errc <- err
				return
			}
			chunks <- c
		}
	}()
	tick := time.NewTicker(20 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case c, ok := <-chunks:
			if !ok {
				select {
				case err := <-errc:
					if !errors.Is(err, context.Canceled) {
						return err
					}
				default:
				}
				return l.finish(ctx)
			}
			l.chunk(ctx, c)
		case <-tick.C:
			l.pause(ctx)
		case fn := <-l.events:
			fn()
		case <-ctx.Done():
			return nil
		}
	}
}

// finish waits for the transcriptions and the turn running, and what they say, after the end
// of the audio.
func (l *Listener) finish(ctx context.Context) error {
	for l.turn != nil || l.pending > 0 {
		select {
		case fn := <-l.events:
			fn()
		case <-ctx.Done():
			return nil
		}
	}
	l.c.Voice.Wait(ctx)
	return nil
}

// muted is the audio without what is heard while the assistant speaks and for Echo after,
// unless BargeIn: the chunks are left out, so that the VAD does not take the voice for the
// user's.
func (l *Listener) muted(s flow.Stream[[]float32]) flow.Stream[[]float32] {
	if l.c.BargeIn {
		return s
	}
	return func(yield func(flow.Chunk[[]float32], error) bool) {
		for c, err := range s {
			if err == nil && l.c.Voice.Idle() < l.c.Echo {
				continue
			}
			if !yield(c, err) {
				return
			}
		}
	}
}

// chunk takes a chunk of an utterance from the VAD.
func (l *Listener) chunk(ctx context.Context, c flow.Chunk[[]float32]) {
	u := l.utt
	if u == nil {
		u = &utterance{addressed: l.following && l.c.Follow > 0 && l.c.Voice.Idle() < l.c.Follow}
		l.utt = u
	}
	u.pcm = append(u.pcm, c.Data...)
	if !c.Final {
		u.last = time.Now()
		if u.early != nil {
			// the speaker went on after the pause: what was transcribed is not the utterance
			if u.early.spec != nil {
				l.c.Log("went on speaking: the answer begun is dropped")
				l.cancel(u.early.spec)
			}
			u.early = nil
		}
		return
	}
	l.utt = nil
	if u.early != nil && u.early.n == len(u.pcm)-len(c.Data) {
		// nothing but the silence after the speech since the pause: its transcript is the
		// utterance's
		u.final = u.early
		if u.early.done {
			l.heard(ctx, u, u.early)
		} // else when it is done (transcribe)
		return
	}
	u.final = l.transcribe(ctx, u, len(u.pcm))
}

// pause starts the transcription of the utterance so far once there has been Pause of silence.
func (l *Listener) pause(ctx context.Context) {
	u := l.utt
	if u == nil || u.early != nil || u.last.IsZero() || time.Since(u.last) < l.c.Pause {
		return
	}
	u.early = l.transcribe(ctx, u, len(u.pcm))
}

// transcribe transcribes the first n samples of an utterance in a goroutine; what comes of it
// is handled in the loop.
func (l *Listener) transcribe(ctx context.Context, u *utterance, n int) *transcript {
	t := &transcript{n: n}
	pcm := u.pcm[:n:n]
	l.pending++
	go func() {
		res, err := l.c.Transcriber.Transcribe(ctx, pcm, l.c.Transcribe)
		l.events <- func() {
			l.pending--
			t.done, t.text, t.err = true, res.Text, err
			switch {
			case u.final == t:
				l.heard(ctx, u, t)
			case u.final == nil && u.early == t && l.c.Speculate:
				l.speculate(ctx, u, t)
			}
		}
	}()
	return t
}

// request is what an utterance asks: the words after the name, or all of them when it needs
// no name; ok is false for an utterance that asks nothing.
func (l *Listener) request(u *utterance, text string) (req string, named, ok bool) {
	if rest, found := l.c.Wake.Find(text); found {
		return rest, true, true
	}
	if u.addressed && !Hallucination(text) {
		return text, false, true
	}
	return "", false, false
}

// speculate starts the turn of a request transcribed at a pause, its answer held back until
// the utterance ends.
func (l *Listener) speculate(ctx context.Context, u *utterance, t *transcript) {
	if t.err != nil || l.turn != nil || l.c.Voice.Busy() {
		return
	}
	req, _, ok := l.request(u, t.text)
	if !ok || req == "" {
		return
	}
	if l.c.Shortcut != nil {
		if _, ok := l.c.Shortcut(req); ok {
			return // answered at once anyway
		}
	}
	l.c.Log("heard at a pause: %q; the agent starts on it", t.text)
	t.spec = l.start(ctx, req, u.last)
}

// heard handles the transcript of a whole utterance.
func (l *Listener) heard(ctx context.Context, u *utterance, t *transcript) {
	if t.err != nil {
		if ctx.Err() == nil {
			l.c.Log("transcribing: %v", t.err)
		}
		return
	}
	if strings.TrimSpace(t.text) == "" {
		return
	}
	if l.c.Events.Heard != nil {
		l.c.Events.Heard(t.text, time.Since(u.last), u.early == t)
	}
	req, named, ok := l.request(u, t.text)
	spec := t.spec
	if spec != nil && l.turn != spec {
		spec = nil // cancelled since
	}
	busy := l.turn != nil && l.turn != spec || l.c.Voice.Busy()
	switch {
	case !ok:
		l.drop(spec)
		return
	case busy && !named:
		return // the assistant is answering: only its name stops it
	case busy:
		l.c.Log("called while answering: stopped")
		l.c.Voice.Stop()
		if l.turn != nil && l.turn != spec {
			l.cancel(l.turn)
		}
	}
	if req == "" {
		l.drop(spec)
		l.c.Voice.Play(l.c.Voice.Chime())
		l.following = true
		return
	}
	if l.c.Shortcut != nil {
		if run, ok := l.c.Shortcut(req); ok {
			l.drop(spec)
			say := run()
			if l.c.Events.Answer != nil {
				l.c.Events.Answer(req, say, true)
			}
			l.firstSound(u.last)
			l.c.Voice.Say(say)
			l.following = true
			return
		}
	}
	l.firstSound(u.last)
	if spec != nil && spec.request == req {
		spec.confirm()
		return
	}
	l.drop(spec)
	l.start(ctx, req, u.last).confirm()
}

// firstSound tells Events.FirstSound when the next sound comes.
func (l *Listener) firstSound(end time.Time) {
	if fn := l.c.Events.FirstSound; fn != nil {
		l.c.Voice.OnHeard(func() { fn(time.Since(end)) })
	}
}

// start starts a turn of the agent, its answer held back until confirm.
func (l *Listener) start(ctx context.Context, req string, end time.Time) *turn {
	tctx, cancel := context.WithCancel(ctx)
	t := &turn{request: req, cancel: cancel, gate: make(chan struct{})}
	l.turn = t
	// the pieces said, from the agent's goroutines (its text as the model writes it, held until
	// the gate; its tool calls)
	var mu sync.Mutex
	split := l.c.Split
	split.Out = l.c.Voice.Say
	said := false
	go func() {
		ans, err := l.c.Agent.Turn(tctx, req, agent.TurnOptions{
			Gate: t.gate,
			Text: func(d string) {
				mu.Lock()
				defer mu.Unlock()
				said = true
				split.Add(d)
			},
			Tool: func(name, args string, slow bool) {
				mu.Lock()
				defer mu.Unlock()
				split.Flush()
				if slow && !said && l.c.Waiting != "" {
					l.c.Voice.SayPhrase(ctx, l.c.Waiting)
					said = true
				}
			},
		})
		if err == nil {
			mu.Lock()
			split.Flush()
			mu.Unlock()
		}
		l.events <- func() {
			if l.turn == t {
				l.turn = nil
			}
			switch {
			case err == nil:
				if l.c.Events.Answer != nil {
					l.c.Events.Answer(req, ans, false)
				}
				l.following = true
			case tctx.Err() == nil:
				if l.c.Events.Failed != nil {
					l.c.Events.Failed(req, err)
				}
				if l.c.Sorry != "" {
					l.c.Voice.SayPhrase(ctx, l.c.Sorry)
				}
			}
		}
	}()
	return t
}

// cancel gives up a turn.
func (l *Listener) cancel(t *turn) {
	t.cancel()
	if l.turn == t {
		l.turn = nil
	}
}

// drop gives up a turn begun at a pause, if there is one.
func (l *Listener) drop(t *turn) {
	if t != nil {
		l.cancel(t)
	}
}

// hallucinations are the texts Whisper writes for noise and silence (the ends of the videos it
// learned from).
var hallucinations = []string{"ご視聴ありがとうございました", "チャンネル登録", "おやすみなさい", "ありがとうございました"}

// Hallucination reports whether a transcript is one a transcriber writes for noise and silence,
// or too short to be a request.
func Hallucination(text string) bool {
	f := Fold(text)
	if len([]rune(f)) < 2 {
		return true
	}
	for _, h := range hallucinations {
		if strings.Contains(f, Fold(h)) {
			return true
		}
	}
	return false
}
