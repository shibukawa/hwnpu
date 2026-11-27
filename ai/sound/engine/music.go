package engine

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

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/sound"
	"github.com/shibukawa/hwnpu/ai/sound/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/music"
	"github.com/shibukawa/hwnpu/internal/notes"
)

// musicGenerator is a music model of the engine (sound.MusicGenerator's). It runs one request
// at a time; others wait.
type musicGenerator struct {
	mu   sync.Mutex
	g    *music.Generator
	p    music.Paths
	dev  *hwnpu.Device // the generator's (AMD's NPU: its decoder's)
	side *hwnpu.Device // MusicCoCa's and SpectroStream's encoder's: dev, nil on AMD's NPU (the CPU)
	st   *music.Styler // loaded at the first style
	// model is the name of the model in the definitions
	model string
	// cpu: the model's device is cpu; amd: the decoder runs on AMD's NPU, the rest on the CPU;
	// allow: programs placed off the NPU are run (ai.AllowOffNPU)
	cpu, amd, allow bool
	log             io.Writer
}

func (m *musicGenerator) Model() string { return m.model }
func (m *musicGenerator) OnCPU() bool   { return m.cpu }

// prepareMusic prepares a music model (sound.Music's settings): every part, MusicCoCa's text and
// audio sides and SpectroStream's encoder with the rest.
func prepareMusic(e airole.PrepareEnv) (f airole.Files, err error) {
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
}

func loadMusic(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	var p music.Paths
	for name, path := range musicPaths(&p) {
		*path = f.Path(name)
	}
	m := &musicGenerator{p: p, model: e.Model, cpu: e.NPU.CPU(e.Def), allow: e.AllowOffNPU, log: e.Log}
	m.amd = !m.cpu && p.IronDecoder != ""
	var err error
	switch {
	case m.amd:
		if m.dev, err = hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD)); err != nil {
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
	return bridge.Wrap["music"](m), nil
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

func (m *musicGenerator) Where() string {
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
func (m *musicGenerator) check(off []string) error {
	if m.cpu || m.amd {
		return nil
	}
	return checkNPU(m.log, m.allow, off)
}

// mono is the clip's channels averaged.
func mono(c sound.Clip) []float32 {
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

// Style loads MusicCoCa at the first call and keeps it.
func (m *musicGenerator) Style(ctx context.Context, parts ...sound.Style) (sound.StyleResult, error) {
	if len(parts) == 0 {
		return sound.StyleResult{}, errors.New("sound: a style of no parts")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	t0 := time.Now()
	if m.st == nil {
		st, err := music.OpenStyler(ctx, m.side, m.p)
		if err != nil {
			return sound.StyleResult{}, err
		}
		if !m.amd { // MusicCoCa runs on the CPU beside AMD's NPU
			if err := m.check(st.Placement()); err != nil {
				st.Close()
				return sound.StyleResult{}, err
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
			return sound.StyleResult{}, fmt.Errorf("sound: a style's weight of %v: more than 0", w)
		}
		var e []float32
		var err error
		switch {
		case len(s.Audio.Channels) > 0:
			// mono, then 16 kHz as magenta-rt's Waveform does (resampy)
			e, err = m.st.Audio(music.Resample(mono(s.Audio), s.Audio.Rate, music.ClipRate))
		case strings.TrimSpace(s.Text) != "":
			e, err = m.st.Text(s.Text)
		default:
			err = errors.New("sound: a style's part of neither text nor audio")
		}
		if err != nil {
			return sound.StyleResult{}, err
		}
		embs, weights = append(embs, e), append(weights, w)
	}
	codes, err := m.st.Tokens(music.Mix(embs, weights))
	if err != nil {
		return sound.StyleResult{}, err
	}
	return sound.StyleResult{Codes: codes, Took: time.Since(t0)}, nil
}

// checkMusic checks o and fills in the defaults: the style's codes (-1 for none) and the frames
// to generate.
func checkMusic(o *sound.MusicOptions) (codes []int, frames int, err error) {
	switch {
	case o.Codes == nil:
		codes = make([]int, sound.StyleCodes)
		for i := range codes {
			codes[i] = -1 // no style: the clip alone leads
		}
	case len(o.Codes) != sound.StyleCodes:
		return nil, 0, fmt.Errorf("sound: a style of %d codes, not %d", len(o.Codes), sound.StyleCodes)
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
	if o.Drums < sound.DrumsAuto || o.Drums > sound.DrumsOff {
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
		g := sound.DefaultGuidance
		o.Guidance = &g
	}
	return codes, frames, nil
}

// Generate gives the stream of the piece, which generates once, when it is read, and its stats,
// whole when the stream ends.
func (m *musicGenerator) Generate(ctx context.Context, o sound.MusicOptions) (flow.Stream[[]float32], *sound.MusicStats) {
	if o.Seed == 0 {
		o.Seed = rand.Uint64() // drawn here so that it can be told (sound.MusicStats.Seed)
	}
	stats := &sound.MusicStats{Seed: o.Seed}
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
	c := flow.Chunk[[]float32]{Data: pcm, Start: flow.Samples(p.pos, sound.MusicRate), Final: final}
	p.pos += len(pcm) / 2
	c.Len = flow.Samples(p.pos, sound.MusicRate) - c.Start
	if !p.yield(c, nil) {
		p.stop = true
		return false
	}
	return true
}

// generate is Generate's stream: an error is the stream's last.
func (m *musicGenerator) generate(ctx context.Context, o sound.MusicOptions, stats *sound.MusicStats, yield func(flow.Chunk[[]float32], error) bool) error {
	codes, frames, err := checkMusic(&o)
	if err != nil {
		return err
	}
	drum := map[sound.Drums]int{sound.DrumsAuto: music.DrumsMasked, sound.DrumsOn: music.DrumsOn, sound.DrumsOff: music.DrumsOff}[o.Drums]
	var roll [][]int
	if len(o.Notes) > 0 {
		ns := make([]notes.MIDINote, len(o.Notes))
		for i, n := range o.Notes {
			ns[i] = notes.MIDINote(n)
		}
		roll = music.PianoRoll(ns, 0, frames, music.RollOptions{Onsets: o.Onsets, Width: o.NoteWidth})
	}
	tell := func(pr sound.MusicProgress) {
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
		tell(sound.MusicProgress{Stage: sound.MusicReadingClip, ClipSeconds: float64(keep) / music.SampleRate})
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
		tell(sound.MusicProgress{Stage: sound.MusicClipRead, ClipSeconds: float64(keep) / music.SampleRate})
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
			tell(sound.MusicProgress{Stage: sound.MusicGenerating, Frames: f + 1, Of: frames, Elapsed: time.Since(t0)})
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

func (m *musicGenerator) Close() error {
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

func (m *musicGenerator) closeDev() {
	if m.dev != nil {
		m.dev.Close()
		m.dev = nil
	}
}
