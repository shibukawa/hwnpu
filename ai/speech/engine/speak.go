package engine

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"io"
	"math/rand/v2"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/speech"
	"github.com/shibukawa/hwnpu/ai/speech/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/iron"
	isp "github.com/shibukawa/hwnpu/internal/speech"
	"github.com/shibukawa/hwnpu/internal/supertonic"
)

// prepareSpeak is the speech role's Prepare.
func prepareSpeak(e airole.PrepareEnv) (f airole.Files, err error) {
	if isSupertonic(e.Model) || isSupertonic(e.Base) {
		return prepareSupertonic(e)
	}
	var c isp.Config
	cpu, ps := e.NPU.CPU(e.Model), e.Model.Programs
	switch {
	case first(ps["iron"]) != "":
		dir := first(ps["iron"])
		if !isp.IsIronExport(dir) {
			return f, fmt.Errorf("%s is not an export of tools/amd/iron_lfm3speech.py", dir)
		}
		c = isp.IronConfig(dir)
	case len(ps) > 0:
		// converted by hand: all of them, used as they are
		c = isp.Config{Text: ps["text"], IronText: first(ps["iron-text"]), Depth: first(ps["depth"]), Codec: first(ps["codec"]), Dir: first(ps["checkpoint"])}
		if c.Dir == "" || len(c.Text) == 0 && c.IronText == "" || c.Depth == "" || c.Codec == "" {
			return f, errors.New("programs converted by hand take all of checkpoint, text (or iron-text), depth and codec")
		}
	case e.NPU.AMD() && !cpu:
		// AMD's NPU runs the export of tools/amd/iron_lfm3speech.py
		if c, err = isp.PrepareAMD(e.Ctx, e.Base, e.Log); err != nil {
			return f, err
		}
		iron.ShareExport(c.Dir, logf(e.Log))
	default:
		backend := e.NPU.Format
		if cpu {
			backend = "" // Core ML packages, what the CPU interpreter runs
		}
		if c, err = isp.Prepare(e.Ctx, isp.Setup{Dir: e.Dir, Repo: e.Model.Source(), Format: e.Model.Settings["format"], Backend: backend, Log: e.Log}); err != nil {
			return f, err
		}
	}
	f.Set("text", c.Text...)
	f.Set("iron-text", c.IronText)
	f.Set("depth", c.Depth)
	f.Set("codec", c.Codec)
	f.Set("dir", c.Dir)
	f.Set("weights", c.Weights)
	f.Set("tokenizer", c.Tokenizer)
	return f, nil
}

// loadSpeak is the speech role's Load.
func loadSpeak(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	if f.Path("supertonic") != "" {
		return loadSupertonic(ctx, f, e)
	}
	s, err := isp.Load(ctx, isp.Config{Text: f.Paths["text"], IronText: f.Path("iron-text"), Depth: f.Path("depth"),
		Codec: f.Path("codec"), Dir: f.Path("dir"), Tokenizer: f.Path("tokenizer"), Weights: f.Path("weights"),
		CacheDir: e.Cache, Device: e.NPU.Device, CPU: e.NPU.CPU(e.Def), Log: e.Log})
	if err != nil {
		return nil, err
	}
	return bridge.Wrap["speak"](&speaker{s: s, model: e.Model}), nil
}

// speaker is a speech model of the engine (speech.Speaker's). It runs one request at a time;
// others wait.
type speaker struct {
	mu    sync.Mutex
	s     *isp.Synth // LFM3-Audio's, or
	st    *supertonic.Synth
	dev   *hwnpu.Device // Supertonic's NPU (nil: the CPU)
	model string
	// styles are Supertonic's voices read, by name
	styles map[string]supertonic.Style
}

func (s *speaker) Model() string { return s.model }
func (s *speaker) Paces() bool   { return s.st != nil }

func (s *speaker) Rate() int {
	if s.st != nil {
		return supertonic.SampleRate
	}
	return s.s.SampleRate()
}

func (s *speaker) Stages() (text, depth, codec string) {
	if s.st != nil {
		return "the text", "the flow", "the vocoder"
	}
	return "the text model", "the depthformer", "the codec decoder"
}

func (s *speaker) OnNPU() bool {
	if s.st != nil {
		n, ok := s.st.Stages.(interface{ OnNPU() bool })
		return ok && n.OnNPU()
	}
	return true
}

// request is LFM3-Audio's request of a text spoken with o.
func request(o speech.SpeakOptions, text string) isp.Request {
	lang, voice := o.Language, o.Voice
	if lang == "" {
		lang = "en"
	}
	if voice == "" {
		voice = "female"
	}
	return isp.Request{Text: text, Language: lang, Voice: voice, Seed: o.Seed, MaxFrames: o.MaxFrames}
}

func (s *speaker) Speak(ctx context.Context, text string, o speech.SpeakOptions) (speech.Audio, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if o.Seed == 0 {
		o.Seed = rand.Uint64() // drawn here so that it can be told (Stats.Seed)
	}
	t0 := time.Now()
	if s.st != nil {
		return s.speakSupertonic(ctx, text, o, t0)
	}
	pcm, frames, err := s.s.Speak(ctx, request(o, text))
	if err != nil {
		return speech.Audio{}, err
	}
	t := s.s.Times
	return speech.Audio{Samples: pcm, Rate: s.s.SampleRate(), Stats: speech.SpeakStats{Frames: len(frames), Seed: o.Seed,
		Took: time.Since(t0), Text: t.Text, Depth: t.Depth, Codec: t.Codec}}, nil
}

func (s *speaker) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.st != nil {
		err := s.st.Close()
		if s.dev != nil {
			s.dev.Close()
		}
		return err
	}
	return s.s.Close()
}

// supertonicPiece is the most letters Supertonic is given at once: a longer text is spoken in
// pieces of whole sentences, so that each fits a program of the NPU (about 9 s of speech).
const supertonicPiece = 60

// speakSupertonic speaks a text with Supertonic: in pieces of sentences, joined by a short
// silence, as the reference does with its chunks.
func (s *speaker) speakSupertonic(ctx context.Context, text string, o speech.SpeakOptions, t0 time.Time) (speech.Audio, error) {
	voice := o.Voice
	st, ok := s.styles[voice]
	if !ok {
		var err error
		if st, err = s.st.Voice(voice); err != nil {
			return speech.Audio{}, fmt.Errorf("speech: the voice %q: %w (Supertonic's are female, male, F1 ... F5, M1 ... M5)", voice, err)
		}
		s.styles[voice] = st
	}
	var pieces []string
	cur := ""
	for _, sent := range flow.SplitSentences(text) {
		if cur != "" && len([]rune(cur+sent)) > supertonicPiece {
			pieces = append(pieces, cur)
			cur = ""
		}
		cur += sent
	}
	if strings.TrimSpace(cur) != "" {
		pieces = append(pieces, cur)
	}
	speed := supertonic.DefaultSpeed * cmp.Or(o.Speed, 1)
	if speed < supertonic.MinSpeed || speed > supertonic.MaxSpeed {
		return speech.Audio{}, fmt.Errorf("speech: the speed %g: Supertonic says from %.2f to %.2f", o.Speed, supertonic.MinSpeed/supertonic.DefaultSpeed, supertonic.MaxSpeed/supertonic.DefaultSpeed)
	}
	var pcm []float32
	var stats speech.SpeakStats
	n := 0
	var say func(p string) error
	say = func(p string) error {
		out, _, err := s.st.Speak(ctx, supertonic.Request{Text: p, Lang: supertonicLang(o.Language), Style: st, Seed: o.Seed + uint64(n), Speed: speed})
		if errors.Is(err, supertonic.ErrTooLong) {
			// longer than the NPU's programs hold: in two, at the pause nearest its middle
			a, b, ok := halve(p)
			if !ok {
				return err
			}
			if err := say(a); err != nil {
				return err
			}
			return say(b)
		}
		if err != nil {
			return err
		}
		if n > 0 {
			pcm = append(pcm, make([]float32, supertonic.SampleRate*15/100)...)
		}
		n++
		pcm = append(pcm, out...)
		t := s.st.Times
		stats.Text += t.Duration + t.Encode
		stats.Depth += t.Estimate
		stats.Codec += t.Vocode
		return nil
	}
	for _, p := range pieces {
		if err := say(strings.TrimSpace(p)); err != nil {
			return speech.Audio{}, err
		}
	}
	stats.Frames, stats.Seed, stats.Took = (len(pcm)+supertonic.Chunk-1)/supertonic.Chunk, o.Seed, time.Since(t0)
	return speech.Audio{Samples: pcm, Rate: supertonic.SampleRate, Stats: stats}, nil
}

// halve cuts a text in two at the pause (a comma, a space) nearest its middle, or at its
// middle without one; ok is false for a text too short to cut.
func halve(text string) (a, b string, ok bool) {
	r := []rune(text)
	if len(r) < 4 {
		return "", "", false
	}
	mid, at := len(r)/2, -1
	for i, c := range r[1 : len(r)-1] {
		if strings.ContainsRune("、，,；;：: 　・…", c) && (at < 0 || abs(i+1-mid) < abs(at-mid)) {
			at = i + 1
		}
	}
	if at < 0 {
		at = mid
	} else {
		at++ // the pause stays with the first half
	}
	return strings.TrimSpace(string(r[:at])), strings.TrimSpace(string(r[at:])), true
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// supertonicLang is a language as Supertonic's token names it: a code (ja, en-gb as en), or
// na for one it does not speak.
func supertonicLang(lang string) string {
	l := strings.ToLower(strings.TrimSpace(lang))
	if i := strings.IndexAny(l, "-_"); i > 0 {
		l = l[:i]
	}
	names := map[string]string{"english": "en", "japanese": "ja", "korean": "ko", "spanish": "es", "french": "fr", "german": "de", "italian": "it", "portuguese": "pt", "russian": "ru", "": "en"}
	if c, ok := names[l]; ok {
		l = c
	}
	for _, c := range supertonic.Languages {
		if c == l {
			return l
		}
	}
	return "na"
}

// supertonicFamily is the family of Supertonic's models in the definitions.
const supertonicFamily = "supertonic"

// isSupertonic reports whether a model is Supertonic's: of its family, or of its repository
// (a model an app derived from the definitions keeps the repository, not the family).
func isSupertonic(m catalog.Model) bool {
	return m.Family == supertonicFamily || strings.HasPrefix(strings.TrimPrefix(m.Source(), "hf:"), "Supertone/")
}

// prepareSupertonic downloads Supertonic 3 and, for the NPU, converts the programs of its flow
// and its vocoder (a model of device cpu runs on ONNX Runtime alone). On AMD's NPU the graphs
// are supertonic.PrepareAMD's export (no programs).
func prepareSupertonic(e airole.PrepareEnv) (f airole.Files, err error) {
	backend := e.NPU.Format
	amd := e.NPU.AMD() && !e.NPU.CPU(e.Model)
	if e.NPU.CPU(e.Model) || e.NPU.AMD() {
		backend = "" // AMD's export, or the CPU: ONNX Runtime reads the graphs
	}
	if backend == llm.BackendCoreAI {
		// Core AI's lowering has no 1-D convolution (the flow's and the vocoder's); Core ML's
		// programs run all on the Neural Engine (measured on M2: 80 times real time)
		backend = llm.BackendCoreML
	}
	c, err := supertonic.Prepare(e.Ctx, supertonic.Setup{Repo: e.Model.Source(), Dir: e.Dir, Backend: backend, Log: e.Log})
	if err != nil {
		return f, err
	}
	if amd {
		dir, err := supertonic.PrepareAMD(e.Ctx, e.Base, e.Log)
		if err != nil {
			return f, err
		}
		iron.ShareExport(dir, logf(e.Log))
		f.Set("amd", dir)
	}
	f.Set("supertonic", c.Dir)
	f.Set("durations", c.Durations...)
	f.Set("encoders", c.Encoders...)
	f.Set("estimators", c.Estimators...)
	f.Set("vocoders", c.Vocoders...)
	return f, nil
}

func loadSupertonic(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	c := supertonic.Config{Dir: f.Path("supertonic"), Durations: f.Paths["durations"], Encoders: f.Paths["encoders"], Estimators: f.Paths["estimators"], Vocoders: f.Paths["vocoders"], AMD: f.Path("amd")}
	if c.AMD != "" {
		dev, err := hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD))
		if err != nil {
			return nil, err
		}
		c.Device = dev
	} else if len(c.Estimators) > 0 && !e.NPU.CPU(e.Def) {
		name := e.NPU.Device
		if name == llm.BackendCoreAI || name == "" && e.NPU.Format == llm.BackendCoreAI {
			// Core AI's device lowers Core ML's programs with its own lowering, which has no
			// 1-D convolution (prepareSupertonic): Core ML runs them
			name = llm.BackendCoreML
		}
		dev, _, err := llm.OpenDevice(ctx, name, e.Cache)
		if err != nil {
			return nil, err
		}
		c.Device = dev
	}
	st, err := supertonic.Load(ctx, c)
	if err != nil {
		if c.Device != nil {
			c.Device.Close()
		}
		return nil, err
	}
	return bridge.Wrap["speak"](&speaker{st: st, dev: c.Device, model: e.Model, styles: map[string]supertonic.Style{}}), nil
}
