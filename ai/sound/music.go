package sound

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"slices"
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
	"github.com/shibukawa/bonsaiai/internal/llm/lfm3audio"
	"github.com/shibukawa/bonsaiai/internal/music"
	"github.com/shibukawa/bonsaiai/internal/notes"
)

// The music's audio: 48 kHz stereo, its samples interleaved (left, right), 25 frames of the
// model a second.
const (
	MusicRate      = music.SampleRate
	MusicFrameRate = music.FrameRate
	// MusicGain is what magenta-rt multiplies the samples by when it writes them as 16 bit
	// integers: bonsaiai-audio writes and plays the music at it.
	MusicGain = 0.5
	// StyleCodes are the codes of a style (MusicCoCa's 12).
	StyleCodes = music.StyleCodes
)

// Music is the use of a model that makes music (Magenta RealTime 2) for an app: a model, a
// family or a use of the definitions ("music", "magenta-realtime-2").
func Music(name string) ai.Use[*MusicGenerator] {
	return ai.Define[*MusicGenerator](musicRole, name)
}

// The settings a music model's definition may give (ai.Definition):
//
//	settings: format   the language model's weight format: lut8 (the default), lut6 or fp16
//
// A model of device cpu runs on bonsaiai's CPU interpreter (slow). On AMD's NPU SpectroStream's
// decoder runs as the graph of tools/amd/iron_magenta.py (exported on the first preparation),
// the language model and MusicCoCa on the CPU interpreter.
//
// Every part is prepared: MusicCoCa's text and audio sides (styles from texts and from audio)
// and SpectroStream's encoder (continuing a clip), loaded at their first use.
var musicRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &MusicGenerator{remote: h} },
	Name:   "music",
	Kinds:  []chat.Kind{chat.KindMusic},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		cpu := e.NPU.CPU(e.Model)
		amd := e.NPU.AMD() && !cpu
		backend := llm.BackendCoreML // what the CPU interpreter runs, the language model's on AMD's NPU
		if !cpu && !amd {
			if backend, err = programFormat(e.NPU); err != nil {
				return f, err
			}
		}
		p, err := music.Prepare(e.Ctx, music.Setup{Dir: e.Dir, Repo: e.Model.Source(), Format: e.Model.Settings["format"],
			Backend: backend, Log: e.Log, Text: true, AudioStyle: true, Encoder: true})
		if err != nil {
			return f, err
		}
		if amd {
			// the export is the model's of the definitions: a weight format of the language
			// model does not change the decoder
			if p.IronDecoder, err = music.PrepareAMD(chat.WithModel(e.Ctx, e.Base), e.Base, e.Log); err != nil {
				return f, err
			}
		}
		for name, path := range musicPaths(&p) {
			f.Set(name, *path)
		}
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		var p music.Paths
		for name, path := range musicPaths(&p) {
			*path = f.Path(name)
		}
		m := &MusicGenerator{p: p, model: e.Model, cpu: e.NPU.CPU(e.Def), allow: e.AllowOffNPU, log: e.Log}
		m.amd = !m.cpu && p.IronDecoder != ""
		var err error
		switch {
		case m.amd:
			if m.dev, err = bonsaiai.Open(ctx, bonsaiai.WithVendor(bonsaiai.VendorAMD)); err != nil {
				return nil, err
			}
		case !m.cpu:
			if m.dev, err = openNPU(ctx, e.NPU, e.Cache); err != nil {
				return nil, err
			}
			m.side = m.dev
		}
		q := p
		q.Encoder = "" // loaded when a clip is continued
		if m.amd {
			m.g, err = music.OpenAMD(ctx, m.dev, q)
		} else {
			m.g, err = music.Open(ctx, m.dev, q)
		}
		if err != nil {
			m.closeDev()
			return nil, err
		}
		if err := m.check(m.g.Placement()); err != nil {
			m.Close()
			return nil, err
		}
		return m, nil
	},
}

// musicPaths are the fields of p by the names a preparation keeps them under.
func musicPaths(p *music.Paths) map[string]*string {
	return map[string]*string{"dir": &p.Dir, "checkpoint": &p.Checkpoint, "host": &p.Host, "temporal": &p.Temporal, "depth": &p.Depth,
		"decoder": &p.Decoder, "encoder-weights": &p.EncoderWeights, "encoder": &p.Encoder,
		"text-encoder": &p.TextEncoder, "music-encoder": &p.MusicEncoder, "audio-preprocessor": &p.AudioPreprocessor,
		"mapper": &p.Mapper, "quantizer": &p.Quantizer, "spm": &p.SPM, "text-program": &p.TextProgram,
		"text-table": &p.TextTable, "music-program": &p.MusicProgram, "music-patches": &p.MusicPatches,
		"iron-decoder": &p.IronDecoder}
}

// MusicGenerator makes music (Magenta RealTime 2): styles from texts and audio (Style), then
// the music of a style, as it is made (Generate). It runs one request at a time; others wait.
type MusicGenerator struct {
	remote *airpc.Handle // the model in bonsaiaid (ai's BONSAIAI_DAEMON), nil when it is here
	mu     sync.Mutex
	g      *music.Generator
	p      music.Paths
	dev    *bonsaiai.Device // the generator's (AMD's NPU: its decoder's)
	side   *bonsaiai.Device // MusicCoCa's and SpectroStream's encoder's: dev, nil on AMD's NPU (the CPU)
	st     *music.Styler    // loaded at the first style
	model  string
	// cpu: the model's device is cpu; amd: the decoder runs on AMD's NPU, the rest on the CPU;
	// allow: programs placed off the NPU are run (ai.AllowOffNPU)
	cpu, amd, allow bool
	log             io.Writer
}

// Model is the name of the model in the definitions.
func (m *MusicGenerator) Model() string {
	if m.remote != nil {
		return airpc.Value[string](m.remote, "Model")
	}
	return m.model
}

// OnCPU reports whether the model runs on bonsaiai's CPU interpreter (its definition's device cpu).
func (m *MusicGenerator) OnCPU() bool {
	if m.remote != nil {
		return airpc.Value[bool](m.remote, "OnCPU")
	}
	return m.cpu
}

// Where says where the programs run, for a log: "all on the NPU", "SpectroStream's decoder on
// the NPU, the language model on the CPU" (AMD's NPU), "on the CPU interpreter", ...
func (m *MusicGenerator) Where() string {
	if m.remote != nil {
		return airpc.Value[string](m.remote, "Where")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	g := m.g
	switch {
	case m.cpu:
		return "on the CPU interpreter"
	case m.amd:
		return "SpectroStream's decoder on the NPU, the language model on the CPU"
	case g.OnNPU() && len(g.Elsewhere()) == 0:
		return "all on the NPU"
	case g.OnNPU():
		return "all on the NPU but " + strings.Join(g.Elsewhere(), ", ")
	case !slices.ContainsFunc(g.Placement(), func(l string) bool { return !strings.Contains(l, "placement not known") }):
		return "placement not known"
	}
	return "NOT all on the NPU"
}

// check refuses programs placed off the NPU (Placement's lines) unless allowed; the CPU's
// programs are where they are meant to be (device cpu, and beside AMD's NPU).
func (m *MusicGenerator) check(off []string) error {
	if m.cpu || m.amd {
		return nil
	}
	return checkNPU(m.log, m.allow, off)
}

// Clip is audio of any rate and channels, as a WAV file holds it (DecodeWAV).
type Clip struct {
	// Channels are the samples of each channel, in [-1, 1].
	Channels [][]float32
	Rate     int
}

// Seconds is how long the clip lasts.
func (c Clip) Seconds() float64 {
	if len(c.Channels) == 0 || c.Rate <= 0 {
		return 0
	}
	return float64(len(c.Channels[0])) / float64(c.Rate)
}

// DecodeWAV decodes a WAV file (PCM of 8, 16, 24 or 32 bits, or 32 bit float) as it is.
func DecodeWAV(data []byte) (Clip, error) {
	ch, rate, err := lfm3audio.DecodeWAV(data)
	return Clip{Channels: ch, Rate: rate}, err
}

// mono is the clip's channels averaged.
func (c Clip) mono() []float32 {
	out := make([]float32, len(c.Channels[0]))
	for i := range out {
		var sum float32
		for _, ch := range c.Channels {
			sum += ch[i]
		}
		out[i] = sum / float32(len(c.Channels))
	}
	return out
}

// Style is a part of a style: a text ("disco funk") or audio in that style, and its weight
// among the parts.
type Style struct {
	Text  string
	Audio Clip
	// Weight is the part's share of the mix (0: 1).
	Weight float64
}

// StyleResult is a style made: its codes, which MusicOptions.Codes takes.
type StyleResult struct {
	Codes []int
	// Took is how long making it took, the first time with loading MusicCoCa.
	Took time.Duration
}

// Style is the style of the parts mixed by their weights (MusicCoCa's embeddings, averaged,
// then its 12 codes). MusicCoCa is loaded at the first call and kept.
func (m *MusicGenerator) Style(ctx context.Context, parts ...Style) (StyleResult, error) {
	if m.remote != nil {
		return airpc.Call1[StyleResult](m.remote, ctx, "Style", parts)
	}
	if len(parts) == 0 {
		return StyleResult{}, errors.New("sound: a style of no parts")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	t0 := time.Now()
	if m.st == nil {
		st, err := music.OpenStyler(ctx, m.side, m.p)
		if err != nil {
			return StyleResult{}, err
		}
		if !m.amd { // MusicCoCa runs on the CPU beside AMD's NPU
			if err := m.check(st.Placement()); err != nil {
				st.Close()
				return StyleResult{}, err
			}
		}
		m.st = st
	}
	var embs [][]float32
	var weights []float64
	for _, s := range parts {
		w := s.Weight
		switch {
		case w == 0:
			w = 1
		case w < 0:
			return StyleResult{}, fmt.Errorf("sound: a style's weight of %v: more than 0", w)
		}
		var e []float32
		var err error
		switch {
		case len(s.Audio.Channels) > 0:
			// mono, then 16 kHz as magenta-rt's Waveform does (resampy)
			e, err = m.st.Audio(music.Resample(s.Audio.mono(), s.Audio.Rate, music.ClipRate))
		case strings.TrimSpace(s.Text) != "":
			e, err = m.st.Text(s.Text)
		default:
			err = errors.New("sound: a style's part of neither text nor audio")
		}
		if err != nil {
			return StyleResult{}, err
		}
		embs, weights = append(embs, e), append(weights, w)
	}
	codes, err := m.st.Tokens(music.Mix(embs, weights))
	if err != nil {
		return StyleResult{}, err
	}
	return StyleResult{Codes: codes, Took: time.Since(t0)}, nil
}

// Drums says whether the music has drums.
type Drums int

const (
	DrumsAuto Drums = iota // the model decides
	DrumsOn
	DrumsOff
)

// Guidance are the classifier-free guidance scales (-1 to 7): how strongly the style, the
// notes and the drums setting guide the music.
type Guidance struct{ Style, Notes, Drums float64 }

// DefaultGuidance is magenta-rt's: 3, 5 and 1.
var DefaultGuidance = Guidance(music.DefaultGuidance)

// MusicOptions are a piece of music's settings.
type MusicOptions struct {
	// Codes are the style's (StyleResult.Codes, StyleCodes of them from 0 to 1023); nil for none,
	// when the clip alone leads (Continue).
	Codes []int
	// Duration is how long the music made lasts; 0 is the notes' length and half a second with
	// Notes, else 10 seconds.
	Duration time.Duration
	// Notes are notes the music plays (ReadMIDI; the drums' channel 10 is left out). Onsets
	// tells their starts (otherwise the model decides where a held note starts again);
	// NoteWidth says that the pitches within that many semitones of a sounding note are silent
	// (127: all).
	Notes     []MIDINote
	Onsets    bool
	NoteWidth int
	// Continue is a clip the music goes on from (its last 20 seconds count): the stream gives
	// the clip, then what follows it; with NewOnly, what follows alone.
	Continue Clip
	NewOnly  bool
	Drums    Drums
	// Temperature (0: 1.3) and TopK (0: 40) are how the tokens are sampled.
	Temperature float64
	TopK        int
	// Guidance is nil for DefaultGuidance.
	Guidance *Guidance
	// Fade fades the last of the music out (0: it stops where the generation does).
	Fade time.Duration
	// Seed makes the music repeatable (0: a new one, told in MusicStats.Seed).
	Seed uint64
	// Progress, when not nil, is told of the clip being read and of each second generated, in
	// the goroutine that reads the stream.
	Progress func(MusicProgress)
}

// MusicStage is what a piece's generation is doing.
type MusicStage int

const (
	// MusicReadingClip: SpectroStream's encoder reads the clip to continue (ClipSeconds of it,
	// the last 20 seconds at most), and the language model goes over its frames.
	MusicReadingClip MusicStage = iota
	// MusicClipRead: the clip is read (MusicStats' ClipFrames and Clip).
	MusicClipRead
	// MusicGenerating: Frames of the Of frames to generate are made, after each second.
	MusicGenerating
)

// MusicProgress is how far a piece's generation is.
type MusicProgress struct {
	Stage       MusicStage
	ClipSeconds float64
	Frames, Of  int
	// Elapsed is the time since the frames began.
	Elapsed time.Duration
}

// MusicStats are what making a piece took. Seed is set when Generate returns; the rest as the
// stream goes, whole when it ends.
type MusicStats struct {
	// Seed is the seed the tokens were sampled with (the one drawn when MusicOptions had none:
	// given again, it makes the same music).
	Seed uint64
	// Frames are the frames generated (25 a second, one more than the duration's: the decoder's
	// lookahead holds the last one back); Samples the stereo samples the stream gave.
	Frames, Samples int
	// ClipFrames are the clip's frames read and Clip how long reading them took.
	ClipFrames int
	Clip       time.Duration
	// Took is the time from the first frame to the end of the music; Decode the decoder's
	// part of it (it runs beside the language model).
	Took, Decode time.Duration
}

// Seconds is how long the music generated lasts (the clip left out).
func (s MusicStats) Seconds() float64 { return float64(s.Frames) / MusicFrameRate }

// RealTime is how many times real time the music was made.
func (s MusicStats) RealTime() float64 { return s.Seconds() / s.Took.Seconds() }

// options checks o and fills in the defaults: the style's codes (-1 for none) and the frames to
// generate.
func (o *MusicOptions) check() (codes []int, frames int, err error) {
	switch {
	case o.Codes == nil:
		codes = make([]int, StyleCodes)
		for i := range codes {
			codes[i] = -1 // no style: the clip alone leads
		}
	case len(o.Codes) != StyleCodes:
		return nil, 0, fmt.Errorf("sound: a style of %d codes, not %d", len(o.Codes), StyleCodes)
	default:
		for _, c := range o.Codes {
			if c < 0 || c >= music.CodebookSize {
				return nil, 0, fmt.Errorf("sound: a style's code %d: from 0 to %d", c, music.CodebookSize-1)
			}
		}
		codes = o.Codes
	}
	if o.Duration < 0 {
		return nil, 0, fmt.Errorf("sound: a duration of %v", o.Duration)
	}
	if o.Drums < DrumsAuto || o.Drums > DrumsOff {
		return nil, 0, fmt.Errorf("sound: drums %d: DrumsAuto, DrumsOn or DrumsOff", o.Drums)
	}
	// one frame more: the decoder's lookahead holds the last one back
	switch {
	case o.Duration > 0:
		frames = int(o.Duration.Seconds()*music.FrameRate) + 1
	case len(o.Notes) > 0:
		end := 0.0
		for _, n := range o.Notes {
			end = math.Max(end, n.End)
		}
		frames = int(math.Ceil(end*music.FrameRate)) + music.FrameRate/2 + 1
	default:
		frames = 10*music.FrameRate + 1
	}
	if o.Temperature == 0 {
		o.Temperature = 1.3
	}
	if o.TopK == 0 {
		o.TopK = 40
	}
	if o.Guidance == nil {
		g := DefaultGuidance
		o.Guidance = &g
	}
	return codes, frames, nil
}

// Generate makes a piece of music and gives it as it is made: 48 kHz stereo (interleaved) a
// second at a time, the stream's clock in samples, the last chunk final. It is faster than real
// time on the NPU; a player paces it (flow.Pace) and plays it at MusicGain. The stats are those
// of the piece, whole when the stream ends. The stream generates once, when it is read.
func (m *MusicGenerator) Generate(ctx context.Context, o MusicOptions) (flow.Stream[[]float32], *MusicStats) {
	if m.remote != nil {
		// the stats are whole when the stream ends, as here (airpc sends them with its end)
		s, st, err := airpc.Call2[flow.Stream[[]float32], *MusicStats](m.remote, ctx, "Generate", o)
		if err != nil {
			return func(yield func(flow.Chunk[[]float32], error) bool) { yield(flow.Chunk[[]float32]{}, err) }, &MusicStats{Seed: o.Seed}
		}
		return s, st
	}
	if o.Seed == 0 {
		o.Seed = rand.Uint64() // drawn here so that it can be told (MusicStats.Seed)
	}
	stats := &MusicStats{Seed: o.Seed}
	return func(yield func(flow.Chunk[[]float32], error) bool) {
		if err := m.generate(ctx, o, stats, yield); err != nil {
			yield(flow.Chunk[[]float32]{}, err)
		}
	}, stats
}

// piece is where a piece's samples go: the stream, the last Fade of them held back to fade out.
type piece struct {
	yield func(flow.Chunk[[]float32], error) bool
	fade  int // interleaved samples held back
	hold  []float32
	pos   int // stereo samples given
	stop  bool
}

// give passes interleaved samples on, but for the last fade of them; false when the reader
// stopped.
func (p *piece) give(pcm []float32) bool {
	p.hold = append(p.hold, pcm...)
	if n := len(p.hold) - p.fade; n > 0 {
		if !p.emit(slices.Clone(p.hold[:n]), false) {
			return false
		}
		p.hold = append(p.hold[:0], p.hold[n:]...)
	}
	return true
}

// end fades out what is held back and gives it as the final chunk.
func (p *piece) end() bool {
	fadeOut(p.hold, 2, len(p.hold)/2)
	ok := p.emit(p.hold, true)
	p.hold = nil
	return ok
}

func (p *piece) emit(pcm []float32, final bool) bool {
	if p.stop {
		return false
	}
	c := flow.Chunk[[]float32]{Data: pcm, Start: flow.Samples(p.pos, MusicRate), Final: final}
	p.pos += len(pcm) / 2
	c.Len = flow.Samples(p.pos, MusicRate) - c.Start
	if !p.yield(c, nil) {
		p.stop = true
		return false
	}
	return true
}

// generate is Generate's stream: an error is the stream's last.
func (m *MusicGenerator) generate(ctx context.Context, o MusicOptions, stats *MusicStats, yield func(flow.Chunk[[]float32], error) bool) error {
	codes, frames, err := o.check()
	if err != nil {
		return err
	}
	drum := map[Drums]int{DrumsAuto: music.DrumsMasked, DrumsOn: music.DrumsOn, DrumsOff: music.DrumsOff}[o.Drums]
	var roll [][]int
	if len(o.Notes) > 0 {
		ns := make([]notes.MIDINote, len(o.Notes))
		for i, n := range o.Notes {
			ns[i] = notes.MIDINote(n)
		}
		roll = music.PianoRoll(ns, 0, frames, music.RollOptions{Onsets: o.Onsets, Width: o.NoteWidth})
	}
	tell := func(pr MusicProgress) {
		if o.Progress != nil {
			o.Progress(pr)
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	g := m.g
	r := rand.New(rand.NewPCG(o.Seed, o.Seed^0x9e3779b97f4a7c15))
	base := music.NewConditioning(codes, music.Guidance(*o.Guidance), drum)
	s := g.NewSession()
	out := &piece{yield: yield, fade: 2 * int(o.Fade.Seconds()*music.SampleRate)}
	if c := o.Continue; len(c.Channels) > 0 {
		if err := g.OpenEncoder(ctx, m.side, m.p); err != nil {
			return err
		}
		if err := m.check(g.Placement()); err != nil {
			return err
		}
		ch := make([][]float32, len(c.Channels))
		for i := range ch {
			ch[i] = music.Resample(c.Channels[i], c.Rate, music.SampleRate)
		}
		stereo := make([]float32, 2*len(ch[0]))
		for i := range ch[0] {
			stereo[2*i], stereo[2*i+1] = ch[0][i], ch[len(ch)-1][i]
		}
		// the last 20 seconds count, in whole frames
		keep := min(len(ch[0]), 500*music.SamplesPerFrame) / music.SamplesPerFrame * music.SamplesPerFrame
		clip := stereo[2*(len(ch[0])-keep):]
		tell(MusicProgress{Stage: MusicReadingClip, ClipSeconds: float64(keep) / music.SampleRate})
		t0 := time.Now()
		cc, err := g.EncodeAudio(clip)
		if err != nil {
			return err
		}
		// prefilled with the style masked, as magenta-rt's runtime does
		if err := s.Prefill(cc, base.WithoutStyle()); err != nil {
			return err
		}
		stats.ClipFrames, stats.Clip = len(cc), time.Since(t0)
		tell(MusicProgress{Stage: MusicClipRead, ClipSeconds: float64(keep) / music.SampleRate})
		if !o.NewOnly && !out.give(stereo[:2*(len(ch[0])-keep)+len(clip)]) {
			return nil
		}
	}
	pick := music.TopK(o.TopK, o.Temperature, r)
	// the decoder turns each second into samples while the language model writes the next one
	type chunk struct {
		pcm []float32
		err error
	}
	want, results, done := make(chan bool, 1024), make(chan chunk, 1024), make(chan struct{})
	decode0 := g.Timing.Decode
	go func() {
		defer close(done)
		defer close(results)
		for final := range want {
			pcm, err := s.Audio(final)
			results <- chunk{pcm, err}
		}
	}()
	defer func() {
		close(want)
		<-done
	}()
	pending := 0
	// drain passes on what the decoder has made: all of it when block; false with the reader
	// stopped
	drain := func(block bool) (bool, error) {
		for pending > 0 {
			var c chunk
			if block {
				c = <-results
			} else {
				select {
				case c = <-results:
				default:
					return true, nil
				}
			}
			pending--
			if c.err != nil {
				return false, c.err
			}
			stats.Samples += len(c.pcm) / 2
			if !out.give(c.pcm) {
				return false, nil
			}
		}
		return true, nil
	}
	t0 := time.Now()
	defer func() {
		stats.Took, stats.Decode = time.Since(t0), g.Timing.Decode-decode0
	}()
	for f := range frames {
		if err := ctx.Err(); err != nil {
			return err
		}
		c := base
		if roll != nil {
			c = c.WithNotes(roll[f])
		}
		if _, err := s.Step(c, pick); err != nil {
			return err
		}
		stats.Frames = f + 1
		if (f+1)%music.FrameRate == 0 {
			tell(MusicProgress{Stage: MusicGenerating, Frames: f + 1, Of: frames, Elapsed: time.Since(t0)})
			// decoded and given a second at a time: a player plays it as it comes
			want <- false
			pending++
			if ok, err := drain(false); !ok {
				return err
			}
		}
	}
	want <- true
	pending++
	if ok, err := drain(true); !ok {
		return err
	}
	out.end()
	return nil
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (m *MusicGenerator) Close() error {
	if m.remote != nil {
		return m.remote.Release() // the Runtime in bonsaiaid unloads it
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.st != nil {
		m.st.Close()
		m.st = nil
	}
	var err error
	if m.g != nil {
		err = m.g.Close()
		m.g = nil
	}
	m.closeDev()
	return err
}

func (m *MusicGenerator) closeDev() {
	if m.dev != nil {
		m.dev.Close()
		m.dev = nil
	}
}

func init() {
	airole.Register(musicRole)
}
