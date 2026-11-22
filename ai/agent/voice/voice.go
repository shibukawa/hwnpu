package voice

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/ai/speech"
)

// Synth makes speech of a text: *speech.Speaker (LFM3-Audio), or another model of speech.
type Synth interface {
	Speak(ctx context.Context, text string, o speech.SpeakOptions) (speech.Audio, error)
	// Rate is the samples a second of the speech.
	Rate() int
	// Model is the model's name (it names the phrases kept on disk).
	Model() string
}

// Player plays mono sound at the Synth's rate, clips one after another, while the program goes
// on: the speakers of an app (the samples' capture.Speaker, through oto), or a file.
type Player interface {
	// Play adds a clip after those before it, without waiting for it.
	Play(pcm []float32)
	// Stop cuts the clip that plays and drops those that wait.
	Stop()
	// Busy reports whether a clip plays or waits to.
	Busy() bool
	// Idle is how long nothing has played (0 while something does).
	Idle() time.Duration
}

// VoiceOptions are how a Voice speaks.
type VoiceOptions struct {
	// Speak are the Synth's settings: the language, the voice and its seed (a seed keeps each
	// sentence in the same voice, and makes the phrases kept on disk the same).
	Speak speech.SpeakOptions
	// Process changes the speech before it plays, such as its speed (nil: as it is made).
	Process func(pcm []float32, rate int) []float32
	// Dir is where the phrases are kept ("" for in memory only).
	Dir string
	// Log receives what is spoken and what it took (nil for none).
	Log func(format string, args ...any)
}

// Voice speaks: the sentences given to Say are synthesized one after another and played as
// each is ready, while the next is being written. The phrases said often (はい, ちょっと待って
// ね) are synthesized once and kept, so they play at once; tones are made without the model.
type Voice struct {
	synth Synth
	play  Player
	o     VoiceOptions
	jobs  chan job

	mu      sync.Mutex
	gen     int // counted up by Stop: the jobs given before it are dropped
	pending int // jobs not yet handed to the player
	phrases map[string][]float32
	heard   func()
}

// job is a sentence to synthesize, or a clip to play, in its turn.
type job struct {
	text string
	pcm  []float32
	gen  int
}

// NewVoice is a voice of the Synth through the Player.
func NewVoice(s Synth, p Player, o VoiceOptions) *Voice {
	if o.Log == nil {
		o.Log = func(string, ...any) {}
	}
	if o.Process == nil {
		o.Process = func(pcm []float32, _ int) []float32 { return pcm }
	}
	v := &Voice{synth: s, play: p, o: o, jobs: make(chan job, 64), phrases: map[string][]float32{}}
	go v.run()
	return v
}

// Rate is the samples a second of the speech.
func (v *Voice) Rate() int { return v.synth.Rate() }

func (v *Voice) run() {
	for j := range v.jobs {
		pcm := j.pcm
		if pcm == nil && v.current(j.gen) {
			t0 := time.Now()
			a, err := v.synth.Speak(context.Background(), j.text, v.o.Speak)
			if err != nil {
				v.o.Log("speaking %q: %v", j.text, err)
			} else {
				pcm = v.o.Process(a.Samples, a.Rate)
				v.o.Log("spoke %q: %.1f s of speech in %.1f s", j.text, float64(len(pcm))/float64(a.Rate), time.Since(t0).Seconds())
			}
		}
		v.mu.Lock()
		v.pending--
		ok := j.gen == v.gen && len(pcm) > 0
		heard := v.heard
		if ok {
			v.heard = nil
		}
		v.mu.Unlock()
		if ok {
			if heard != nil {
				heard()
			}
			v.play.Play(pcm)
		}
	}
}

func (v *Voice) current(gen int) bool {
	v.mu.Lock()
	defer v.mu.Unlock()
	return gen == v.gen
}

func (v *Voice) add(j job) {
	v.mu.Lock()
	j.gen = v.gen
	v.pending++
	v.mu.Unlock()
	v.jobs <- j
}

// Say speaks a sentence after what was given before it.
func (v *Voice) Say(sentence string) {
	if s := Speakable(sentence); s != "" {
		v.add(job{text: s})
	}
}

// Play plays a clip (at Rate) after what was given before it.
func (v *Voice) Play(pcm []float32) { v.add(job{pcm: pcm}) }

// OnHeard has fn called when the next sound is handed to the player (for the time an answer
// takes).
func (v *Voice) OnHeard(fn func()) {
	v.mu.Lock()
	v.heard = fn
	v.mu.Unlock()
}

// Stop cuts what is said and drops what waits to be.
func (v *Voice) Stop() {
	v.mu.Lock()
	v.gen++
	v.heard = nil
	v.mu.Unlock()
	v.play.Stop()
}

// Busy reports whether something is said or waits to be.
func (v *Voice) Busy() bool {
	v.mu.Lock()
	p := v.pending
	v.mu.Unlock()
	return p > 0 || v.play.Busy()
}

// Idle is how long nothing has been said (0 while something is).
func (v *Voice) Idle() time.Duration {
	if v.Busy() {
		return 0
	}
	return v.play.Idle()
}

// Wait waits until what was given has been said, or ctx ends.
func (v *Voice) Wait(ctx context.Context) {
	for v.Busy() && ctx.Err() == nil {
		time.Sleep(20 * time.Millisecond)
	}
}

// Phrase is the speech of a phrase said often: from memory, from Dir, or synthesized now and
// kept.
func (v *Voice) Phrase(ctx context.Context, text string) ([]float32, error) {
	v.mu.Lock()
	pcm, ok := v.phrases[text]
	v.mu.Unlock()
	if ok {
		return pcm, nil
	}
	// the file is named by what makes the speech: the model, the text, the voice and its seed
	o := v.o.Speak
	key := sha256.Sum256([]byte(strings.Join([]string{v.synth.Model(), text, o.Language, o.Voice, strconv.FormatUint(o.Seed, 10), strconv.FormatFloat(o.Speed, 'g', -1, 64)}, "\x00")))
	path := ""
	if v.o.Dir != "" {
		path = filepath.Join(v.o.Dir, hex.EncodeToString(key[:8])+".wav")
		if raw, err := os.ReadFile(path); err == nil {
			if pcm, err := speech.ReadWAV(raw, v.synth.Rate()); err == nil {
				return v.keep(text, pcm), nil
			}
		}
	}
	a, err := v.synth.Speak(ctx, text, o)
	if err != nil {
		return nil, err
	}
	if path != "" {
		var b bytes.Buffer
		if speech.WriteWAV(&b, a.Samples, a.Rate) == nil && os.MkdirAll(v.o.Dir, 0o755) == nil {
			os.WriteFile(path, b.Bytes(), 0o644)
		}
	}
	return v.keep(text, a.Samples), nil
}

func (v *Voice) keep(text string, pcm []float32) []float32 {
	pcm = v.o.Process(pcm, v.synth.Rate())
	v.mu.Lock()
	v.phrases[text] = pcm
	v.mu.Unlock()
	return pcm
}

// SayPhrase plays a phrase said often (Phrase), after what was given before it.
func (v *Voice) SayPhrase(ctx context.Context, text string) {
	pcm, err := v.Phrase(ctx, text)
	if err != nil {
		v.o.Log("speaking %q: %v", text, err)
		return
	}
	v.Play(pcm)
}

// Tone is a tone of the notes given (Hz, 0 for a rest), each of d, softened at its ends.
func (v *Voice) Tone(d time.Duration, notes ...float64) []float32 {
	rate := v.synth.Rate()
	n := int(d.Seconds() * float64(rate))
	fade := rate / 100 // 10 ms
	var out []float32
	for _, f := range notes {
		for i := range n {
			g := 0.25 * math.Min(1, math.Min(float64(i)/float64(fade), float64(n-i)/float64(fade)))
			out = append(out, float32(g*math.Sin(2*math.Pi*f*float64(i)/float64(rate))))
		}
	}
	return out
}

// Chime is the sound of an assistant that heard its name alone: it listens.
func (v *Voice) Chime() []float32 { return v.Tone(90*time.Millisecond, 880, 1320) }

// Alarm is the ring of a timer: three double beeps.
func (v *Voice) Alarm() []float32 {
	var out []float32
	for range 3 {
		out = append(out, v.Tone(120*time.Millisecond, 1760, 0, 1760)...)
		out = append(out, make([]float32, v.synth.Rate()/4)...)
	}
	return out
}

var (
	markup = regexp.MustCompile("[*#`_>|~\\[\\]]|https?://\\S+")
	spaces = regexp.MustCompile(`\s+`)
)

// Speakable is a sentence as it is spoken: without the marks of Markdown, URLs and emoji that a
// model writes now and then.
func Speakable(s string) string {
	s = markup.ReplaceAllString(s, "")
	s = strings.Map(func(r rune) rune {
		if r >= 0x1F000 || (r >= 0x2600 && r <= 0x27BF) {
			return -1 // emoji
		}
		return r
	}, s)
	return strings.TrimSpace(spaces.ReplaceAllString(s, " "))
}
