package speech

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

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/ai"
	"github.com/shibukawa/bonsaiai/ai/flow"
	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/airpc"
	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/llm/iron"
	"github.com/shibukawa/bonsaiai/internal/speech"
	"github.com/shibukawa/bonsaiai/internal/supertonic"
)

// Speak is the use of a model of text to speech (LFM3-Audio, Supertonic 3) for an app: a model,
// a family or a use of the definitions ("speech", "supertonic").
func Speak(name string) ai.Use[*Speaker] {
	return ai.Define[*Speaker](speakRole, name)
}

// The settings and programs a speech model's definition may give (ai.Definition):
//
//	settings: format     the text model's weight format (default: the model's, 8 bit tables)
//	programs: iron       on AMD's NPU, an export of tools/amd/iron_lfm3speech.py (its directory)
//	          text       the text model converted by hand (-cache and -hidden-out 1), its packages
//	                     of different -seq; or iron-text, on AMD's NPU its export of
//	                     iron_export_lm.py lfm3tts
//	          depth      the depthformer (-arch lfm3-depthformer)
//	          codec      the codec decoder (-arch lfm3-codec)
//	          checkpoint the checkpoint's directory of those (config.json, model.safetensors,
//	                     tokenizer.json, audio_tokenizer/config.json)
//
// A model of device cpu runs on bonsaiai's CPU interpreter (slow).
var speakRole = &airole.Role{
	Name:   "speak",
	Remote: func(h *airpc.Handle) io.Closer { return &Speaker{remote: h} },
	Kinds:  []chat.Kind{chat.KindSpeech},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		if isSupertonic(e.Model) || isSupertonic(e.Base) {
			return prepareSupertonic(e)
		}
		var c speech.Config
		cpu, ps := e.NPU.CPU(e.Model), e.Model.Programs
		switch {
		case first(ps["iron"]) != "":
			dir := first(ps["iron"])
			if !speech.IsIronExport(dir) {
				return f, fmt.Errorf("%s is not an export of tools/amd/iron_lfm3speech.py", dir)
			}
			c = speech.IronConfig(dir)
		case len(ps) > 0:
			// converted by hand: all of them, used as they are
			c = speech.Config{Text: ps["text"], IronText: first(ps["iron-text"]), Depth: first(ps["depth"]), Codec: first(ps["codec"]), Dir: first(ps["checkpoint"])}
			if c.Dir == "" || len(c.Text) == 0 && c.IronText == "" || c.Depth == "" || c.Codec == "" {
				return f, errors.New("programs converted by hand take all of checkpoint, text (or iron-text), depth and codec")
			}
		case e.NPU.AMD() && !cpu:
			// AMD's NPU runs the export of tools/amd/iron_lfm3speech.py
			if c, err = speech.PrepareAMD(e.Ctx, e.Base, e.Log); err != nil {
				return f, err
			}
			iron.ShareExport(c.Dir, logf(e.Log))
		default:
			backend := e.NPU.Format
			if cpu {
				backend = "" // Core ML packages, what the CPU interpreter runs
			}
			if c, err = speech.Prepare(e.Ctx, speech.Setup{Dir: e.Dir, Repo: e.Model.Source(), Format: e.Model.Settings["format"], Backend: backend, Log: e.Log}); err != nil {
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
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		if f.Path("supertonic") != "" {
			return loadSupertonic(ctx, f, e)
		}
		s, err := speech.Load(ctx, speech.Config{Text: f.Paths["text"], IronText: f.Path("iron-text"), Depth: f.Path("depth"),
			Codec: f.Path("codec"), Dir: f.Path("dir"), Tokenizer: f.Path("tokenizer"), Weights: f.Path("weights"),
			CacheDir: e.Cache, Device: e.NPU.Device, CPU: e.NPU.CPU(e.Def), Log: e.Log})
		if err != nil {
			return nil, err
		}
		return &Speaker{s: s, model: e.Model}, nil
	},
	// 1: Supertonic's programs of its text's graphs (durations, encoders), with no ONNX Runtime
	// 2: Supertonic on AMD's NPU (amd: supertonic.PrepareAMD's export)
	// 3: its text's graphs in the export too (no programs)
	// 4: the tokenizer as its binary file (chat.PrepareTokenizer) in the files, as "tokenizer": a
	// store prepared before named none, and Load then read the checkpoint's tokenizer.json, which a
	// checkpoint fetched since is without
	Revision: 4,
}

// SpeakOptions are how a text is spoken (LFM3-Audio's settings).
type SpeakOptions struct {
	// Language is a code of the model card (en, en-gb, ja, zh, ...) or a language's name;
	// "" is a chunk's Text.Lang in SpeakStream, else en.
	Language string
	// Voice is female or male ("": female).
	Voice string
	// Seed makes the speech repeatable (0: a new one each call, told in Audio.Stats; SpeakStream
	// draws one for the whole stream, so that its voice stays the same).
	Seed uint64
	// MaxFrames is the most frames of 80 ms (0: until the model ends the speech).
	MaxFrames int
	// Speed is how fast the text is said, as a multiple of the model's own pace (0: 1), by a
	// model that paces its speech (Paces: Supertonic, from 0.67 to 1.9); the others leave it.
	Speed float64
}

// Paces reports whether the model says a text at SpeakOptions.Speed itself (Supertonic); the
// speech of another is to be stretched for it.
func (s *Speaker) Paces() bool {
	if s.remote != nil {
		return airpc.Value[bool](s.remote, "Paces")
	}
	return s.st != nil
}

// Audio is mono audio.
type Audio struct {
	Samples []float32 // in [-1, 1]
	Rate    int       // samples a second
	// Stats are what speaking it took (Speak's; the zero value elsewhere).
	Stats SpeakStats
}

// SpeakStats are what speaking a text took.
type SpeakStats struct {
	// Frames are the audio frames of 80 ms made, and Seed the seed they were drawn with (the one
	// drawn when SpeakOptions had none: given again, it speaks the same).
	Frames int
	Seed   uint64
	// Took is the whole time; Text, Depth and Codec the time in the text model (the prompt and
	// the frames' steps), the depthformer and the codec decoder.
	Took, Text, Depth, Codec time.Duration
}

// Speaker speaks texts (LFM3-Audio). It runs one request at a time; others wait.
type Speaker struct {
	remote *airpc.Handle // the model in bonsaiaid (ai's BONSAIAI_DAEMON), nil when it is here
	mu     sync.Mutex
	s      *speech.Synth // LFM3-Audio's, or
	st     *supertonic.Synth
	dev    *bonsaiai.Device // Supertonic's NPU (nil: the CPU)
	model  string
	// styles are Supertonic's voices read, by name
	styles map[string]supertonic.Style
}

// Model is the name of the model in the definitions.
func (s *Speaker) Model() string {
	if s.remote != nil {
		return airpc.Value[string](s.remote, "Model")
	}
	return s.model
}

// Rate is the samples a second of the speech (LFM3-Audio's 24000, Supertonic's 44100).
func (s *Speaker) Rate() int {
	if s.remote != nil {
		return airpc.Value[int](s.remote, "Rate")
	}
	if s.st != nil {
		return supertonic.SampleRate
	}
	return s.s.SampleRate()
}

// Stages names what SpeakStats' Text, Depth and Codec times are of: LFM3-Audio's text model,
// depthformer and codec decoder, or Supertonic's text graphs, flow and vocoder.
func (s *Speaker) Stages() (text, depth, codec string) {
	if s.remote != nil {
		out, err := s.remote.Call(context.Background(), "Stages")
		if err != nil || len(out) != 3 {
			return "", "", ""
		}
		return out[0].String(), out[1].String(), out[2].String()
	}
	if s.st != nil {
		return "the text", "the flow", "the vocoder"
	}
	return "the text model", "the depthformer", "the codec decoder"
}

// OnNPU reports whether the speech is made on the NPU (Supertonic's flow and vocoder; false on
// the CPU).
func (s *Speaker) OnNPU() bool {
	if s.remote != nil {
		return airpc.Value[bool](s.remote, "OnNPU")
	}
	if s.st != nil {
		n, ok := s.st.Stages.(interface{ OnNPU() bool })
		return ok && n.OnNPU()
	}
	return true
}

func (o SpeakOptions) request(text string) speech.Request {
	lang, voice := o.Language, o.Voice
	if lang == "" {
		lang = "en"
	}
	if voice == "" {
		voice = "female"
	}
	return speech.Request{Text: text, Language: lang, Voice: voice, Seed: o.Seed, MaxFrames: o.MaxFrames}
}

// Speak speaks a text.
func (s *Speaker) Speak(ctx context.Context, text string, o SpeakOptions) (Audio, error) {
	if strings.TrimSpace(text) == "" {
		return Audio{}, errors.New("speech: nothing to speak")
	}
	if s.remote != nil {
		return airpc.Call1[Audio](s.remote, ctx, "Speak", text, o)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if o.Seed == 0 {
		o.Seed = rand.Uint64() // drawn here so that it can be told (Stats.Seed)
	}
	t0 := time.Now()
	if s.st != nil {
		return s.speakSupertonic(ctx, text, o, t0)
	}
	pcm, frames, err := s.s.Speak(ctx, o.request(text))
	if err != nil {
		return Audio{}, err
	}
	t := s.s.Times
	return Audio{Samples: pcm, Rate: s.s.SampleRate(), Stats: SpeakStats{Frames: len(frames), Seed: o.Seed,
		Took: time.Since(t0), Text: t.Text, Depth: t.Depth, Codec: t.Codec}}, nil
}

// SpeakStream speaks each final chunk of a stream of text (flow.Sentences, or a translation's)
// as it comes: a final chunk of audio each, timed on the output's own clock (its samples from
// the start, so that flow.Pace hands them to a player one after another), with one seed for the
// whole stream. Provisional chunks are left out. Speech that comes out as it is made, before
// the text's whole audio is, is to come.
func (s *Speaker) SpeakStream(ctx context.Context, in flow.Stream[flow.Text], o SpeakOptions) flow.Stream[[]float32] {
	return s.speakStream(ctx, in, func() SpeakOptions { return o })
}

// SpeakNode is the node of a speaker in a network or a call (flow, ai.App.From, ai.NewCall): texts in, their
// speech out, as SpeakStream gives it. Its settings may be changed while the network runs
// (Update): the next text is spoken with them.
func SpeakNode(u ai.Use[*Speaker], o SpeakOptions) *ai.ModelNode[flow.Text, []float32, SpeakOptions] {
	return ai.NewModelNode(u, o, func(ctx context.Context, s *Speaker, in flow.Stream[flow.Text], options func() SpeakOptions) flow.Stream[[]float32] {
		return s.speakStream(ctx, in, options)
	})
}

// speakStream is SpeakStream with the options asked for at each text.
func (s *Speaker) speakStream(ctx context.Context, in flow.Stream[flow.Text], options func() SpeakOptions) flow.Stream[[]float32] {
	return func(yield func(flow.Chunk[[]float32], error) bool) {
		seed := rand.Uint64() // one for the stream when the options have none
		pos := 0
		for c, err := range in {
			if err != nil {
				yield(flow.Chunk[[]float32]{}, err)
				return
			}
			if !c.Final || strings.TrimSpace(c.Data.Text) == "" {
				continue
			}
			oo := options()
			if oo.Seed == 0 {
				oo.Seed = seed
			}
			if oo.Language == "" {
				oo.Language = c.Data.Lang
			}
			a, err := s.Speak(ctx, c.Data.Text, oo)
			if err != nil {
				yield(flow.Chunk[[]float32]{}, err)
				return
			}
			out := flow.Chunk[[]float32]{Data: a.Samples, Start: flow.Samples(pos, a.Rate), Final: true}
			pos += len(a.Samples)
			out.Len = flow.Samples(pos, a.Rate) - out.Start
			if !yield(out, nil) {
				return
			}
		}
	}
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (s *Speaker) Close() error {
	if s.remote != nil {
		return s.remote.Release() // the Runtime in bonsaiaid unloads it
	}
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
func (s *Speaker) speakSupertonic(ctx context.Context, text string, o SpeakOptions, t0 time.Time) (Audio, error) {
	voice := o.Voice
	st, ok := s.styles[voice]
	if !ok {
		var err error
		if st, err = s.st.Voice(voice); err != nil {
			return Audio{}, fmt.Errorf("speech: the voice %q: %w (Supertonic's are female, male, F1 ... F5, M1 ... M5)", voice, err)
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
		return Audio{}, fmt.Errorf("speech: the speed %g: Supertonic says from %.2f to %.2f", o.Speed, supertonic.MinSpeed/supertonic.DefaultSpeed, supertonic.MaxSpeed/supertonic.DefaultSpeed)
	}
	var pcm []float32
	var stats SpeakStats
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
			return Audio{}, err
		}
	}
	stats.Frames, stats.Seed, stats.Took = (len(pcm)+supertonic.Chunk-1)/supertonic.Chunk, o.Seed, time.Since(t0)
	return Audio{Samples: pcm, Rate: supertonic.SampleRate, Stats: stats}, nil
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

func init() { airole.Register(speakRole) }

// supertonicFamily is the family of Supertonic's models in the definitions.
const supertonicFamily = "supertonic"

// isSupertonic reports whether a model is Supertonic's: of its family, or of its repository
// (a model an app derived from the definitions keeps the repository, not the family).
func isSupertonic(m chat.Model) bool {
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
		dev, err := bonsaiai.Open(ctx, bonsaiai.WithVendor(bonsaiai.VendorAMD))
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
	return &Speaker{st: st, dev: c.Device, model: e.Model, styles: map[string]supertonic.Style{}}, nil
}
