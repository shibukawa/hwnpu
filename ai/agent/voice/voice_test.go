package voice

import (
	"context"
	"math"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/shibukawa/bonsaiai/ai/agent"
	"github.com/shibukawa/bonsaiai/ai/flow"
	"github.com/shibukawa/bonsaiai/ai/speech"
	"github.com/shibukawa/bonsaiai/ai/text"
)

func TestWake(t *testing.T) {
	w := NewWake("コダマ", "こだま", "木霊", "児玉", "kodama")
	for _, c := range []struct {
		in, rest string
		ok       bool
	}{
		{"コダマ、今何時？", "今何時？", true},
		{"こだま 明日の天気は", "明日の天気は", true},
		{"ねえ、コダマ。", "", true},
		{"ヘイ木霊、タイマー3分", "タイマー3分", true},
		{"OK, Kodama. What time is it?", "What time is it?", true},
		{"コダマさん、ニュースを教えて", "ニュースを教えて", true},
		{"コダマー、起きてる？", "起きてる？", true},
		{"今日は児玉さんと会った", "", false},
		{"明日の天気は？", "", false},
	} {
		rest, ok := w.Find(c.in)
		if ok != c.ok || rest != c.rest {
			t.Errorf("Find(%q) = %q, %v; want %q, %v", c.in, rest, ok, c.rest, c.ok)
		}
	}
}

func TestSplitter(t *testing.T) {
	var got []string
	s := Splitter{Out: func(p string) { got = append(got, p) }}
	for _, tok := range []string{"東京は", "今、晴れていて", "、気温は十八度です。", "明日は", "雨の予報です", "。傘を", "持っていってね"} {
		s.Add(tok)
	}
	s.Flush()
	want := []string{"東京は今、晴れていて、", "気温は十八度です。", "明日は雨の予報です。", "傘を持っていってね"}
	if !slices.Equal(got, want) {
		t.Errorf("pieces %q, want %q", got, want)
	}
}

func TestSpeakable(t *testing.T) {
	if got := Speakable("**晴れ**です 😀 https://example.com"); got != "晴れです" {
		t.Errorf("Speakable = %q", got)
	}
	for in, want := range map[string]bool{"ご視聴ありがとうございました。": true, "あ": true, "明日の天気は？": false} {
		if got := Hallucination(in); got != want {
			t.Errorf("Hallucination(%q) = %v", in, got)
		}
	}
}

// The fakes of a Listener's models, and a player that keeps what it was given.

// words is a transcriber that writes the words of an utterance by its length: each second of
// speech is the next word.
type words struct {
	words []string
	mu    sync.Mutex
	calls int
}

func (w *words) Transcribe(ctx context.Context, pcm []float32, o speech.TranscribeOptions) (speech.Transcript, error) {
	w.mu.Lock()
	w.calls++
	w.mu.Unlock()
	time.Sleep(100 * time.Millisecond) // the transcriber's time
	// the speech's length: the loud samples
	loud := 0
	for _, x := range pcm {
		if math.Abs(float64(x)) > 0.01 {
			loud++
		}
	}
	n := min(len(w.words), int(math.Round(float64(loud)/float64(speech.TranscribeRate))))
	return speech.Transcript{Text: strings.Join(w.words[:n], "")}, nil
}

type synth struct{}

func (synth) Speak(ctx context.Context, text string, o speech.SpeakOptions) (speech.Audio, error) {
	return speech.Audio{Samples: make([]float32, 240), Rate: 24000}, nil
}
func (synth) Rate() int     { return 24000 }
func (synth) Model() string { return "fake" }

type player struct {
	mu    sync.Mutex
	clips int
}

func (p *player) Play(pcm []float32)  { p.mu.Lock(); p.clips++; p.mu.Unlock() }
func (p *player) Stop()               {}
func (p *player) Busy() bool          { return false }
func (p *player) Idle() time.Duration { return time.Hour }

// model answers every request with its text, and keeps the requests it read.
type model struct {
	mu   sync.Mutex
	read []string
}

func (m *model) Reply(ctx context.Context, msgs []text.Message, o text.ChatOptions, fn func(text.Delta) bool) (text.Answer, error) {
	req := msgs[len(msgs)-1].Text
	m.mu.Lock()
	m.read = append(m.read, req)
	m.mu.Unlock()
	a := "はい、" + req + "の答えです。"
	fn(text.Delta{Text: a})
	return text.Answer{Text: a}, nil
}

// audio is speech of the lengths given (seconds) with the silences between them, in a quiet
// room, as a microphone would give it.
func audio(ctx context.Context, parts ...float64) flow.Stream[[]float32] {
	rate := speech.TranscribeRate
	var pcm []float32
	add := func(secs float64, loud bool) {
		for i := range int(secs * float64(rate)) {
			x := 0.001 * math.Sin(float64(i)*0.7)
			if loud {
				x = 0.3 * math.Sin(2*math.Pi*220*float64(i)/float64(rate))
			}
			pcm = append(pcm, float32(x))
		}
	}
	add(0.5, false)
	for i, p := range parts {
		add(p, i%2 == 0)
	}
	add(1.2, false)
	return flow.Live(ctx, speech.Chunks(pcm, rate, rate/50))
}

type heard struct {
	text  string
	early bool
}

func listen(t *testing.T, stt *words, m *model, c Config, parts ...float64) (h []heard, answers []string) {
	t.Helper()
	ctx := context.Background()
	c.Transcriber, c.Speculate = stt, true
	c.Agent = agent.New(m, agent.Config{})
	c.Voice = NewVoice(synth{}, &player{}, VoiceOptions{})
	c.Wake = NewWake("コダマ")
	t0 := time.Now()
	c.Log = func(f string, a ...any) { t.Logf("%.2f "+f, append([]any{time.Since(t0).Seconds()}, a...)...) }
	c.Events = Events{
		Heard:  func(s string, _ time.Duration, early bool) { h = append(h, heard{s, early}) },
		Answer: func(_, a string, _ bool) { answers = append(answers, a) },
	}
	if err := NewListener(c).Run(ctx, audio(ctx, parts...)); err != nil {
		t.Fatal(err)
	}
	return h, answers
}

func TestListenAtPause(t *testing.T) {
	stt := &words{words: []string{"コダマ、", "天気"}}
	m := &model{}
	h, answers := listen(t, stt, m, Config{}, 2)
	if !slices.Equal(h, []heard{{"コダマ、天気", true}}) {
		t.Errorf("heard %v: want the transcript of the pause", h)
	}
	if !slices.Equal(answers, []string{"はい、天気の答えです。"}) || !slices.Equal(m.read, []string{"天気"}) {
		t.Errorf("answers %q, the model read %q", answers, m.read)
	}
	if stt.calls != 1 {
		t.Errorf("%d transcriptions: want the pause's alone", stt.calls)
	}
}

func TestListenGoesOn(t *testing.T) {
	// a pause of 0.4 s, past Pause but within the VAD's Hangover: the turn begun at the pause
	// is dropped, and the whole utterance answered
	stt := &words{words: []string{"コダマ、明日の", "天気", "は？"}}
	m := &model{}
	h, answers := listen(t, stt, m, Config{}, 1, 0.4, 2)
	if len(h) != 1 || h[0].text != "コダマ、明日の天気は？" {
		t.Errorf("heard %v", h)
	}
	if !slices.Equal(answers, []string{"はい、明日の天気は？の答えです。"}) {
		t.Errorf("answers %q: want the whole request's alone", answers)
	}
	if !slices.Contains(m.read, "明日の") {
		t.Errorf("the model read %q: want the request of the pause begun", m.read)
	}
}

func TestListenShortcut(t *testing.T) {
	stt := &words{words: []string{"コダマ、", "時刻"}}
	m := &model{}
	short := func(req string) (func() string, bool) {
		if req == "時刻" {
			return func() string { return "3時です。" }, true
		}
		return nil, false
	}
	_, answers := listen(t, stt, m, Config{Shortcut: short}, 2)
	if !slices.Equal(answers, []string{"3時です。"}) || len(m.read) != 0 {
		t.Errorf("answers %q, the model read %q: want the shortcut's alone", answers, m.read)
	}
}
