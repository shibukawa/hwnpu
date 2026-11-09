// hwnpu-music generates music with Magenta RealTime 2 small (google/magenta-realtime-2) on the
// NPU, as a 48 kHz stereo WAV: in the style of texts or of audio, following a MIDI file's notes,
// going on from a clip, as it is generated. With a sound effects model (-m sfx, or
// moss-soundeffect: MOSS-SoundEffect v2.0, Apache-2.0) it makes a sound effect from a
// description instead (sfx.go).
//
//	hwnpu-music "disco funk"                                  # plays through the speakers
//	hwnpu-music -o out.wav -duration 10s "disco funk"
//	hwnpu-music -o out.wav "jazz piano@2" "bossa nova@1"     # a mix of styles, weighted
//	hwnpu-music -o out.wav song.wav                           # the style of a recording
//	hwnpu-music -o out.wav -continue intro.wav "disco funk"   # the clip, then where it goes
//	hwnpu-music -o out.wav -notes melody.mid "string quartet" # playing the MIDI file's notes
//	hwnpu-music -duration 2m "lo-fi"                          # without -o: through the speakers as it comes
//	hwnpu-music -play -o out.wav "lo-fi"                      # both
//	hwnpu-music -o - -duration 2m "lo-fi" | ffplay -          # a WAV to stdout as it comes
//	hwnpu-music -m sfx "A dog barking loudly in a park."      # a sound effect
//	hwnpu-music -m moss-soundeffect:8gb -o rain.wav "Heavy rain on a tin roof"  # as on an 8 GB machine
//
// The model (about 1.7 GB; the audio side of the style 0.4 GB more) is downloaded and converted
// the first time. The weights are CC-BY 4.0: credit "Magenta RealTime 2 by Google DeepMind" where
// the music is used.
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/music"
	"github.com/shibukawa/hwnpu/internal/notes"
	"github.com/shibukawa/hwnpu/internal/sfx"
)

type options struct {
	use, dir, cacheDir, format, out string
	cpu, newOnly, offNPU, play      bool
	outSet, playSet                 bool
	decoderOn                       string
	duration, fade                  time.Duration
	durationSet                     bool
	codes, cont, notes, drums       string
	onsets                          bool
	width                           int
	temperature                     float64
	topK                            int
	guidance                        music.Guidance
	seed                            uint64
	fadeSet                         bool
	// sound effects
	steps                 int
	cfg, shift            float64
	textFormat, ditFormat string
}

func main() {
	var o options
	flag.StringVar(&o.use, "use", "music", "the model: a name, a use or a family of -list (sfx for sound effects; FAMILY:8gb picks the one for an 8 GB machine)")
	flag.StringVar(&o.use, "m", "music", "the same as -use")
	flag.StringVar(&o.dir, "dir", defaultDir(), "where the models are kept, each in a directory of its name")
	flag.StringVar(&o.out, "o", "", "the WAV file to write (-: stdout); without -o and -play, the music plays")
	flag.DurationVar(&o.fade, "fade", 2*time.Second, "fade the music out over its last part (0: stop where the generation does)")
	flag.BoolVar(&o.play, "play", false, "play the music through the speakers as it is generated (the default without -o)")
	flag.DurationVar(&o.duration, "duration", 10*time.Second, "how long the music is: 10s, 2m, 1m30s (with -notes, the MIDI file's length by default)")
	flag.StringVar(&o.codes, "codes", "", "MusicCoCa's 12 codes of the style, comma separated, instead of texts or audio")
	flag.StringVar(&o.cont, "continue", "", "a WAV file to go on from: the music is the clip, then what follows it (its last 20 seconds count)")
	flag.BoolVar(&o.newOnly, "new-only", false, "with -continue, write what follows the clip alone")
	flag.StringVar(&o.notes, "notes", "", "a MIDI file whose notes the music plays (the drums' channel 10 left out)")
	flag.BoolVar(&o.onsets, "onsets", false, "with -notes, tell the notes' starts (otherwise the model decides where a held note starts again)")
	flag.IntVar(&o.width, "note-width", 0, "with -notes, also say that the pitches within this many semitones of a sounding note are silent (127: all)")
	flag.StringVar(&o.drums, "drums", "auto", "drums: auto (the model decides), on or off")
	flag.Float64Var(&o.temperature, "temperature", 1.3, "sampling temperature")
	flag.IntVar(&o.topK, "top-k", 40, "sample from the k most likely tokens")
	flag.Float64Var(&o.guidance.Style, "cfg-style", music.DefaultGuidance.Style, "how strongly the style guides (-1 to 7)")
	flag.Float64Var(&o.guidance.Notes, "cfg-notes", music.DefaultGuidance.Notes, "how strongly the notes guide (-1 to 7)")
	flag.Float64Var(&o.guidance.Drums, "cfg-drums", music.DefaultGuidance.Drums, "how strongly the drums setting guides (-1 to 7)")
	flag.Uint64Var(&o.seed, "seed", 0, "random seed (0: from the time)")
	flag.IntVar(&o.steps, "steps", sfx.DefaultSteps, "sound effects: flow-matching steps (each two calls of the DiT; fewer is faster and rougher)")
	flag.Float64Var(&o.cfg, "cfg", sfx.DefaultCFG, "sound effects: how strongly the description guides (classifier-free guidance)")
	flag.Float64Var(&o.shift, "shift", sfx.DefaultShift, "sound effects: the sigma schedule's shift")
	flag.StringVar(&o.textFormat, "text-format", "", "sound effects: the text encoder's weight format, fp16 or lut8 (default: the model's, by its machine)")
	flag.StringVar(&o.ditFormat, "dit-format", "", "sound effects: the DiT's weight format, lut8 (default) or fp16")
	flag.StringVar(&o.format, "format", "", "the language model's weight format (lut8 by default; fp16, lut6)")
	flag.BoolVar(&o.cpu, "cpu", false, "run the programs on the CPU interpreter instead of the NPU")
	flag.StringVar(&o.decoderOn, "decoder-on", "npu", "where SpectroStream's decoder and encoder run: npu, or gpu or cpu to compare")
	flag.BoolVar(&o.offNPU, "allow-off-npu", false, "go on when Core ML puts part of a program off the NPU (on the GPU or the CPU; that has made a Mac sluggish and failed)")
	flag.StringVar(&o.cacheDir, "cache", "", "compile cache (default: hwnpu in the user cache directory)")
	list := flag.Bool("list", false, "list the music and sound effects models of the definitions and exit")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `usage: hwnpu-music [flags] [STYLE...]
       hwnpu-music -m sfx [flags] DESCRIPTION

Music as a 48 kHz stereo WAV. Each STYLE is a text or a WAV file (its style), with an optional
weight after @ ("jazz@2"); several are mixed. The weights are CC-BY 4.0: credit "Magenta
RealTime 2 by Google DeepMind" where the music is used.

With a sound effects model (-m sfx, moss-soundeffect, moss-soundeffect:8gb), a sound effect of
up to 30 seconds from an English or Chinese description (MOSS-SoundEffect v2.0, Apache-2.0);
-steps, -cfg, -shift, -text-format and -dit-format are its.

`)
		flag.PrintDefaults()
	}
	flag.Parse()
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "o" {
			o.outSet = true
		}
		if f.Name == "play" {
			o.playSet = true
		}
		if f.Name == "duration" {
			o.durationSet = true
		}
		if f.Name == "fade" {
			o.fadeSet = true
		}
	})
	if *list {
		chat.List(os.Stdout, o.dir, chat.KindMusic, chat.KindSFX)
		return
	}
	if flag.NArg() == 0 && o.codes == "" && o.cont == "" {
		flag.Usage()
		os.Exit(2)
	}
	if !o.outSet && !o.playSet {
		o.play = true // neither a file nor the speakers asked for: the speakers
	}
	if o.cacheDir == "" {
		if d, err := os.UserCacheDir(); err == nil {
			o.cacheDir = filepath.Join(d, "hwnpu")
		}
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := run(ctx, o, flag.Args()); err != nil {
		fmt.Fprintln(os.Stderr, "hwnpu-music:", err)
		os.Exit(1)
	}
}

func defaultDir() string {
	d, err := os.UserCacheDir()
	if err != nil {
		return "models"
	}
	return filepath.Join(d, "hwnpu", "models")
}

// style is a STYLE argument: a text or a WAV file, and its weight.
type style struct {
	text, wav string
	weight    float64
}

func parseStyles(args []string) ([]style, error) {
	var out []style
	for _, a := range args {
		s := style{text: a, weight: 1}
		if i := strings.LastIndex(a, "@"); i > 0 {
			if w, err := strconv.ParseFloat(a[i+1:], 64); err == nil {
				if w <= 0 {
					return nil, fmt.Errorf("%q: the weight must be more than 0", a)
				}
				s.text, s.weight = a[:i], w
			}
		}
		if strings.HasSuffix(strings.ToLower(s.text), ".wav") {
			if _, err := os.Stat(s.text); err == nil {
				s.wav, s.text = s.text, ""
			}
		}
		out = append(out, s)
	}
	return out, nil
}

func run(ctx context.Context, o options, args []string) error {
	styles, err := parseStyles(args)
	if err != nil {
		return err
	}
	drums := map[string]int{"auto": music.DrumsMasked, "on": music.DrumsOn, "off": music.DrumsOff}
	drum, ok := drums[o.drums]
	if !ok {
		return fmt.Errorf("-drums %q: auto, on or off", o.drums)
	}
	// what goes on in gray (with a spinner or a bar on a terminal), where the music went as it is
	lw := console.Log(os.Stderr)
	logf := func(format string, args ...any) { fmt.Fprintf(lw, format+"\n", args...) }
	spin := func(label string) func(result string) {
		stop := console.Spin(lw, label)
		return func(result string) {
			stop()
			fmt.Fprintf(lw, " %s\n", result)
		}
	}
	m, err := chat.ResolveForKinds(o.use, lw, chat.KindMusic, chat.KindSFX)
	if err != nil {
		return err
	}
	if m.Kind == chat.KindSFX {
		return runSFX(ctx, o, m, args, lw)
	}
	backend := llm.BackendCoreML
	var dev *hwnpu.Device
	if !o.cpu {
		var device string
		if dev, device, err = llm.OpenDevice(ctx, "", o.cacheDir); err != nil {
			return err
		}
		defer dev.Close()
		if backend = llm.ProgramBackend(device); backend != llm.BackendCoreML {
			return fmt.Errorf("%s: only Apple's Neural Engine runs Magenta RealTime 2 so far (-cpu runs it on the CPU)", device)
		}
	}
	setup := music.Setup{Dir: filepath.Join(o.dir, m.DirName()), Repo: m.Source(), Format: o.format, Backend: backend, Log: lw, Encoder: o.cont != ""}
	if o.codes == "" {
		for _, s := range styles {
			setup.Text = setup.Text || s.text != ""
			setup.AudioStyle = setup.AudioStyle || s.wav != ""
		}
	}
	p, err := music.Prepare(ctx, setup)
	if err != nil {
		return err
	}

	// the style
	var codes []int
	switch {
	case o.codes != "":
		for _, s := range strings.Split(o.codes, ",") {
			c, err := strconv.Atoi(strings.TrimSpace(s))
			if err != nil || c < 0 || c >= 1024 {
				return fmt.Errorf("-codes %q: 12 codes from 0 to 1023", o.codes)
			}
			codes = append(codes, c)
		}
		if len(codes) != music.StyleCodes {
			return fmt.Errorf("-codes has %d codes, not %d", len(codes), music.StyleCodes)
		}
	case len(styles) > 0:
		t0 := time.Now()
		done := spin("reading the style")
		st, err := music.OpenStyler(ctx, dev, p)
		if err != nil {
			done("failed")
			return err
		}
		if err := checkNPU(lw, o, st.Placement()); err != nil {
			done("failed")
			st.Close()
			return err
		}
		var embs [][]float32
		var weights []float64
		var names []string
		for _, s := range styles {
			var e []float32
			if s.wav != "" {
				raw, err := os.ReadFile(s.wav)
				if err != nil {
					st.Close()
					return err
				}
				ch, rate, err := lfm3audio.DecodeWAV(raw)
				if err != nil {
					st.Close()
					return fmt.Errorf("%s: %w", s.wav, err)
				}
				// mono, then 16 kHz as magenta-rt's Waveform does (resampy)
				mono := make([]float32, len(ch[0]))
				for i := range mono {
					var sum float32
					for _, c := range ch {
						sum += c[i]
					}
					mono[i] = sum / float32(len(ch))
				}
				x := music.Resample(mono, rate, music.ClipRate)
				e, err = st.Audio(x)
				if err != nil {
					st.Close()
					return err
				}
				names = append(names, fmt.Sprintf("%s (%.0f s)", s.wav, float64(len(x))/music.ClipRate))
			} else {
				if e, err = st.Text(s.text); err != nil {
					st.Close()
					return err
				}
				names = append(names, strconv.Quote(s.text))
			}
			embs, weights = append(embs, e), append(weights, s.weight)
		}
		codes, err = st.Tokens(music.Mix(embs, weights))
		st.Close()
		if err != nil {
			done("failed")
			return err
		}
		done(fmt.Sprintf("in %s", console.Duration(time.Since(t0))))
		logf("style %s: %v (-codes takes them)", strings.Join(names, " + "), codes)
	default:
		codes = make([]int, music.StyleCodes)
		for i := range codes {
			codes[i] = -1 // no style: the clip alone leads
		}
	}

	// the notes
	var roll [][]int
	// one frame more: the decoder's lookahead holds the last one back
	if o.duration <= 0 {
		return fmt.Errorf("-duration %v: more than 0", o.duration)
	}
	frames := int(o.duration.Seconds()*music.FrameRate) + 1
	if o.notes != "" {
		raw, err := os.ReadFile(o.notes)
		if err != nil {
			return err
		}
		ns, err := notes.ReadMIDI(raw)
		if err != nil {
			return fmt.Errorf("%s: %w", o.notes, err)
		}
		end := 0.0
		for _, n := range ns {
			end = math.Max(end, n.End)
		}
		if !o.durationSet {
			frames = int(math.Ceil(end*music.FrameRate)) + music.FrameRate/2 + 1
		}
		roll = music.PianoRoll(ns, 0, frames, music.RollOptions{Onsets: o.onsets, Width: o.width})
		logf("notes: %d of %s (%.1f s)", len(ns), o.notes, end)
	}

	t0 := time.Now()
	units := map[string]string{"npu": "", "gpu": "gpu", "cpu": "cpu"}
	ss, ok := units[o.decoderOn]
	if !ok {
		return fmt.Errorf("-decoder-on %q: npu, gpu or cpu", o.decoderOn)
	}
	done := spin("loading " + m.Name)
	g, err := music.OpenOn(ctx, dev, p, ss)
	if err != nil {
		done("failed")
		return err
	}
	defer g.Close()
	where := "on the CPU interpreter"
	switch {
	case o.cpu:
	case g.OnNPU() && len(g.Elsewhere()) == 0:
		where = "all on the NPU"
	case g.OnNPU():
		where = "all on the NPU but " + strings.Join(g.Elsewhere(), ", ")
	default:
		where = "NOT all on the NPU"
	}
	done(fmt.Sprintf("in %s, %s", console.Duration(time.Since(t0)), where))
	if err := checkNPU(lw, o, g.Placement()); err != nil {
		return err
	}
	if o.seed == 0 {
		o.seed = uint64(time.Now().UnixNano())
	}
	r := rand.New(rand.NewPCG(o.seed, o.seed^0x9e3779b97f4a7c15))
	base := music.NewConditioning(codes, o.guidance, drum)
	s := g.NewSession()

	// the output, a file and the speakers: the clip first, unless -new-only
	var out sink
	if o.out != "" {
		if out.w, err = newWAVWriter(o.out); err != nil {
			return err
		}
	}
	if o.play {
		if out.sp, err = newSpeaker(); err != nil {
			return fmt.Errorf("the speakers: %w", err)
		}
	}
	out.fade = 2 * int(o.fade.Seconds()*music.SampleRate)
	w := &out
	if o.cont != "" {
		raw, err := os.ReadFile(o.cont)
		if err != nil {
			return err
		}
		ch, rate, err := lfm3audio.DecodeWAV(raw)
		if err != nil {
			return fmt.Errorf("%s: %w", o.cont, err)
		}
		for i := range ch {
			ch[i] = music.Resample(ch[i], rate, music.SampleRate)
		}
		stereo := make([]float32, 2*len(ch[0]))
		for i := range ch[0] {
			stereo[2*i], stereo[2*i+1] = ch[0][i], ch[len(ch)-1][i]
		}
		// the last 20 seconds count, in whole frames
		keep := min(len(ch[0]), 500*music.SamplesPerFrame) / music.SamplesPerFrame * music.SamplesPerFrame
		clip := stereo[2*(len(ch[0])-keep):]
		t0 := time.Now()
		done := spin(fmt.Sprintf("reading the last %.1f s of %s", float64(keep)/music.SampleRate, o.cont))
		cc, err := g.EncodeAudio(clip)
		if err != nil {
			done("failed")
			return err
		}
		// prefilled with the style masked, as magenta-rt's runtime does
		if err := s.Prefill(cc, base.WithoutStyle()); err != nil {
			done("failed")
			return err
		}
		done(fmt.Sprintf("%d frames in %s", len(cc), console.Duration(time.Since(t0))))
		if !o.newOnly {
			if err := w.write(stereo[:2*(len(ch[0])-keep)+len(clip)]); err != nil {
				return err
			}
		}
	}
	pick := music.TopK(o.topK, o.temperature, r)
	// the decoder (on the GPU) turns each second into samples while the language model (on the
	// NPU) writes the next one
	type chunk struct {
		pcm []float32
		err error
	}
	want, results := make(chan bool, 1024), make(chan chunk, 1024)
	go func() {
		for final := range want {
			pcm, err := s.Audio(final)
			results <- chunk{pcm, err}
		}
		close(results)
	}()
	pending := 0
	drain := func(block bool) error {
		for pending > 0 {
			var c chunk
			if block {
				c = <-results
			} else {
				select {
				case c = <-results:
				default:
					return nil
				}
			}
			pending--
			if c.err != nil {
				return c.err
			}
			if err := w.write(c.pcm); err != nil {
				return err
			}
		}
		return nil
	}
	defer close(want)
	bar := console.NewBar(lw, "generating")
	t0 = time.Now()
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
		if (f+1)%music.FrameRate == 0 {
			el := time.Since(t0)
			bar.Set(float64(f+1)/float64(frames), fmt.Sprintf("%s of %s, %.1f times real time", console.Duration(time.Duration(f+1)*time.Second/music.FrameRate), console.Duration(time.Duration(frames)*time.Second/music.FrameRate), float64(f+1)/music.FrameRate/el.Seconds()))
			// decoded and written a second at a time: the file holds what is done if
			// something fails, and -o - plays as it comes
			want <- false
			pending++
			if err := drain(false); err != nil {
				return err
			}
		}
	}
	want <- true
	pending++
	if err := drain(true); err != nil {
		bar.Done("failed")
		return err
	}
	if err := out.end(); err != nil {
		bar.Done("failed")
		return err
	}
	el := time.Since(t0)
	bar.Done(fmt.Sprintf("%s in %s (%.1f times real time; the decoder %s of it)", console.Duration(time.Duration(frames)*time.Second/music.FrameRate), console.Duration(el), float64(frames)/music.FrameRate/el.Seconds(), console.Duration(g.Timing.Decode)))
	if out.sp != nil {
		done := spin("playing the rest")
		if err := out.sp.finish(ctx); err != nil {
			done("stopped")
			return err
		}
		done("done")
	}
	if out.w != nil {
		if err := out.w.close(); err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%s: %.1f s of 48 kHz stereo\n", o.out, float64(out.w.frames)/music.SampleRate)
	}
	logf("the weights are CC-BY 4.0: credit \"Magenta RealTime 2 by Google DeepMind\" where you use the music")
	return nil
}

// sink is where the music goes: a WAV file, the speakers, or both. It holds back the last fade
// of samples, which end fades out, so that the music does not stop dead where the generation does.
type sink struct {
	w    *wavWriter
	sp   *speaker
	fade int // interleaved samples held back
	hold []float32
}

func (s *sink) write(pcm []float32) error {
	s.hold = append(s.hold, pcm...)
	if n := len(s.hold) - s.fade; n > 0 {
		if err := s.emit(s.hold[:n]); err != nil {
			return err
		}
		s.hold = append(s.hold[:0], s.hold[n:]...)
	}
	return nil
}

// end fades out what is held back (a raised cosine to silence) and writes it.
func (s *sink) end() error {
	n := len(s.hold) / 2
	for i := range n {
		g := float32(0.5 * (1 + math.Cos(math.Pi*float64(i)/float64(n))))
		s.hold[2*i] *= g
		s.hold[2*i+1] *= g
	}
	err := s.emit(s.hold)
	s.hold = nil
	return err
}

func (s *sink) emit(pcm []float32) error {
	if s.sp != nil {
		s.sp.write(pcm)
	}
	if s.w != nil {
		return s.w.write(pcm)
	}
	return nil
}

// checkNPU reports the programs not all on the NPU, and stops unless -allow-off-npu.
func checkNPU(lw io.Writer, o options, off []string) error {
	if o.cpu || len(off) == 0 {
		return nil
	}
	for _, l := range off {
		fmt.Fprintln(lw, "  NOT all on the NPU:", l)
	}
	if o.offNPU {
		return nil
	}
	return fmt.Errorf("%d programs are not all on the NPU, which has made a Mac sluggish and failed; -allow-off-npu goes on anyway", len(off))
}

// wavWriter writes a 16-bit stereo WAV as samples come, its sizes set at the end when the file
// can seek (a stream to stdout says it goes on), and after each write so that a run that fails
// leaves a WAV of what was done.
type wavWriter struct {
	w      io.Writer
	f      *os.File
	frames int
	gain   float64 // 0.5 for music (the reference's int16 conversion), 1 for sound effects
}

func newWAVWriter(path string) (*wavWriter, error) {
	w := &wavWriter{gain: 0.5}
	if path == "-" {
		w.w = os.Stdout
	} else {
		f, err := os.Create(path)
		if err != nil {
			return nil, err
		}
		w.w, w.f = f, f
	}
	h := make([]byte, 44)
	copy(h, "RIFF")
	binary.LittleEndian.PutUint32(h[4:], 0xffffffff)
	copy(h[8:], "WAVEfmt ")
	binary.LittleEndian.PutUint32(h[16:], 16)
	binary.LittleEndian.PutUint16(h[20:], 1)
	binary.LittleEndian.PutUint16(h[22:], 2)
	binary.LittleEndian.PutUint32(h[24:], music.SampleRate)
	binary.LittleEndian.PutUint32(h[28:], 4*music.SampleRate)
	binary.LittleEndian.PutUint16(h[32:], 4)
	binary.LittleEndian.PutUint16(h[34:], 16)
	copy(h[36:], "data")
	binary.LittleEndian.PutUint32(h[40:], 0xffffffff)
	_, err := w.w.Write(h)
	return w, err
}

// write writes interleaved stereo samples at the writer's gain.
func (w *wavWriter) write(pcm []float32) error {
	b := make([]byte, 2*len(pcm))
	for i, v := range pcm {
		x := math.Round(math.Max(-1, math.Min(1, w.gain*float64(v))) * 32767)
		binary.LittleEndian.PutUint16(b[2*i:], uint16(int16(x)))
	}
	w.frames += len(pcm) / 2
	if _, err := w.w.Write(b); err != nil {
		return err
	}
	return w.sizes()
}

func (w *wavWriter) sizes() error {
	if w.f == nil {
		return nil
	}
	size := uint32(4 * w.frames)
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], 36+size)
	if _, err := w.f.WriteAt(b[:], 4); err != nil {
		return err
	}
	binary.LittleEndian.PutUint32(b[:], size)
	_, err := w.f.WriteAt(b[:], 40)
	return err
}

func (w *wavWriter) close() error {
	if w.f == nil {
		return nil
	}
	if err := w.sizes(); err != nil {
		w.f.Close()
		return err
	}
	return w.f.Close()
}
