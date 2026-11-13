// hwnpu-audio runs audio models on an NPU: music and sound effects made, speech spoken and
// transcribed, and the pitch and the notes of recordings.
//
//	hwnpu-audio generate "disco funk"                                    # music, played through the speakers
//	hwnpu-audio generate -o out.wav --duration 10s "disco funk"
//	hwnpu-audio generate -o out.wav "jazz piano@2" "bossa nova@1"       # a mix of styles, weighted
//	hwnpu-audio generate -o out.wav song.wav                             # the style of a recording
//	hwnpu-audio generate -o out.wav --continue intro.wav "disco funk"    # the clip, then where it goes
//	hwnpu-audio generate -o out.wav --notes melody.mid "string quartet"  # playing the MIDI file's notes
//	hwnpu-audio generate --play -o out.wav "lo-fi"                       # both
//	hwnpu-audio generate -o - --duration 2m "lo-fi" | ffplay -           # a WAV to stdout as it comes
//	hwnpu-audio generate -m sfx "A dog barking loudly in a park."       # a sound effect
//	hwnpu-audio generate -m moss-soundeffect:8gb -o rain.wav "Heavy rain on a tin roof"
//	hwnpu-audio tts --lang ja -o out.wav こんにちは                       # speech (LFM3-Audio-2B)
//	hwnpu-audio transcribe speech.wav                                    # text (Whisper)
//	hwnpu-audio transcribe -l ja --timestamps a.wav b.wav
//	hwnpu-audio pitch voice.wav                                          # CSV: the pitch every 16 ms (SwiftF0)
//	hwnpu-audio notes --midi song.mid song.wav                           # CSV and MIDI: the notes (Basic Pitch)
//	hwnpu-audio list [music|sfx|speech|transcribe|pitch|notes]
//
// --model takes a name, a family or a use of models.yaml (music, magenta-realtime-2; sfx,
// moss-soundeffect; speech; transcribe, whisper:large-v3-turbo; pitch; notes): a family alone
// picks for the machine's memory as hwnpu-chat's --model, a tag of memory (moss-soundeffect:8gb, 16G)
// for that machine's. Each model is downloaded and converted the first time. Without -o, generate
// plays the sound.
//
// Music is Magenta RealTime 2 small (google/magenta-realtime-2): 48 kHz stereo in the style of
// texts or of audio, following a MIDI file's notes, going on from a clip, as it is generated.
// The model (about 1.7 GB; the audio side of the style 0.4 GB more) is downloaded and converted
// the first time. Its weights are CC-BY 4.0: credit "Magenta RealTime 2 by Google DeepMind" where
// the music is used.
//
// Sound effects are MOSS-SoundEffect v2.0 (OpenMOSS-Team/MOSS-SoundEffect-v2.0, Apache-2.0): up
// to 30 seconds of 48 kHz mono from an English or Chinese description (sfx.go).
//
// tts (TTSCmd), transcribe (TranscribeCmd), pitch (PitchCmd) and notes (NotesCmd) are described
// at their commands.
package main

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"github.com/shibukawa/hwnpu/internal/cache"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/console"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/music"
	"github.com/shibukawa/hwnpu/internal/notes"
)

type cli struct {
	Generate   GenerateCmd   `cmd:"" help:"Make music from styles (Magenta RealTime 2), or a sound effect from a description (-m sfx: MOSS-SoundEffect v2.0)."`
	TTS        TTSCmd        `cmd:"" name:"tts" help:"Speak a text into a WAV file (LFM3-Audio-2B; Apple's Neural Engine)."`
	Transcribe TranscribeCmd `cmd:"" help:"Turn speech in WAV files into text (Whisper)."`
	Pitch      PitchCmd      `cmd:"" help:"The pitch (F0) of a voice or an instrument every 16 ms, as CSV (SwiftF0)."`
	Notes      NotesCmd      `cmd:"" help:"The notes of music, as CSV and MIDI (Basic Pitch)."`
	List       ListCmd       `cmd:"" help:"List the audio models, whether they are downloaded and what each use and family picks for a machine's memory."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	var c cli
	k := kong.Parse(&c,
		kong.Name("hwnpu-audio"),
		kong.Description("Audio models on an NPU: music, sound effects, speech, transcription, pitch and notes."),
		kong.UsageOnError(),
		kong.Vars{
			"models_usage": chat.ModelsUsage,
			"onset":        fmt.Sprint(notes.DefaultOptions.OnsetThreshold),
			"frame":        fmt.Sprint(notes.DefaultOptions.FrameThreshold),
			"min_note":     fmt.Sprint(notes.DefaultOptions.MinNoteMS),
			"melodia":      fmt.Sprint(notes.DefaultOptions.Melodia),
		},
		kong.BindTo(ctx, (*context.Context)(nil)))
	console.MeasureEnergy(console.Log(os.Stderr)) // HWNPU_ENERGY: the idle power, then each phase's energy in the last lines
	k.FatalIfErrorf(k.Run())
	console.Done(console.Log(os.Stderr)) // the times, when the command said console.Running; HWNPU_PROFILE's profile
}

// GenerateCmd is "hwnpu-audio generate": music, or with a sound effects model a sound effect.
type GenerateCmd struct {
	Args     []string       `arg:"" optional:"" name:"STYLE|DESCRIPTION" help:"Music: texts or WAV files of the style, each with an optional weight after @ (\"jazz@2\"), mixed. Sound effects: the description."`
	Model    string         `short:"m" default:"music" placeholder:"NAME" help:"The model: a name, a family or a use of models.yaml (music; sfx, moss-soundeffect; a family alone picks for the machine's memory, or for the memory a tag names: moss-soundeffect:8gb)."`
	Out      string         `short:"o" placeholder:"FILE" help:"The WAV to write (-: stdout); without it the sound plays through the speakers."`
	Play     bool           `help:"Play through the speakers (music: as it is generated) besides writing --out."`
	Duration *time.Duration `placeholder:"10s" help:"How long: 10s, 2m, 1m30s (default 10s; music with --notes: the MIDI file's length; sound effects up to 30s)."`
	Fade     *time.Duration `placeholder:"2s" help:"Fade out over the last part (music: 2s by default, 0 to stop where the generation does; sound effects: none by default)."`
	Seed     uint64         `placeholder:"N" help:"The random seed (0: a new one)."`
	Dir      string         `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's --dir)."`
	Cache    string         `placeholder:"DIR" help:"Compile cache (default: hwnpu in the user cache directory)."`
	CPU      bool           `name:"cpu" help:"Run the programs with the CPU interpreter instead of the NPU (slow)."`
	OffNPU   bool           `name:"allow-off-npu" help:"Go on when Core ML puts part of a program off the NPU (on the GPU or the CPU; that has made a Mac sluggish and failed)."`

	Codes       string  `group:"Music" placeholder:"C,..." help:"MusicCoCa's 12 codes of the style, comma separated, instead of texts or audio."`
	Continue    string  `group:"Music" type:"existingfile" placeholder:"WAV" help:"A WAV to go on from: the music is the clip, then what follows it (its last 20 seconds count)."`
	NewOnly     bool    `group:"Music" help:"With --continue, write what follows the clip alone."`
	Notes       string  `group:"Music" type:"existingfile" placeholder:"MIDI" help:"A MIDI file whose notes the music plays (the drums' channel 10 left out)."`
	Onsets      bool    `group:"Music" help:"With --notes, tell the notes' starts (otherwise the model decides where a held note starts again)."`
	NoteWidth   int     `group:"Music" placeholder:"N" help:"With --notes, also say that the pitches within this many semitones of a sounding note are silent (127: all)."`
	Drums       string  `group:"Music" enum:"auto,on,off" default:"auto" help:"Drums: auto (the model decides), on or off."`
	Temperature float64 `group:"Music" default:"1.3" help:"Sampling temperature."`
	TopK        int     `group:"Music" name:"top-k" default:"40" help:"Sample from the k most likely tokens."`
	CFGStyle    float64 `group:"Music" name:"cfg-style" default:"3" help:"How strongly the style guides (-1 to 7)."`
	CFGNotes    float64 `group:"Music" name:"cfg-notes" default:"5" help:"How strongly the notes guide (-1 to 7)."`
	CFGDrums    float64 `group:"Music" name:"cfg-drums" default:"1" help:"How strongly the drums setting guides (-1 to 7)."`
	Format      string  `group:"Music" placeholder:"FORMAT" help:"The language model's weight format (lut8 by default; fp16, lut6)."`
	DecoderOn   string  `group:"Music" name:"decoder-on" enum:"npu,gpu,cpu" default:"npu" help:"Where SpectroStream's decoder and encoder run: npu, or gpu or cpu to compare."`

	Steps      int     `group:"Sound effects" default:"100" help:"Flow-matching steps (each two calls of the DiT; fewer is faster and rougher)."`
	CFG        float64 `group:"Sound effects" name:"cfg" default:"4" help:"How strongly the description guides (classifier-free guidance)."`
	Shift      float64 `group:"Sound effects" default:"5" help:"The sigma schedule's shift."`
	TextFormat string  `group:"Sound effects" placeholder:"FORMAT" help:"The text encoder's weight format, fp16 or lut8 (default: the model's, by its machine)."`
	DitFormat  string  `group:"Sound effects" placeholder:"FORMAT" help:"The DiT's weight format, lut8 (default) or fp16."`
}

// options are GenerateCmd's values as the generation reads them.
type options struct {
	use, dir, cacheDir, format, out string
	cpu, newOnly, offNPU, play      bool
	decoderOn                       string
	duration, fade                  time.Duration
	durationSet, fadeSet            bool
	codes, cont, notes, drums       string
	onsets                          bool
	width                           int
	temperature                     float64
	topK                            int
	guidance                        music.Guidance
	seed                            uint64
	// sound effects
	steps                 int
	cfg, shift            float64
	textFormat, ditFormat string
}

// Run makes the music or the sound effect.
func (c *GenerateCmd) Run(ctx context.Context) error {
	o := options{use: c.Model, dir: c.Dir, cacheDir: c.Cache, format: c.Format, out: c.Out,
		cpu: c.CPU, newOnly: c.NewOnly, offNPU: c.OffNPU, play: c.Play || c.Out == "", decoderOn: c.DecoderOn,
		duration: 10 * time.Second, fade: 2 * time.Second, codes: c.Codes, cont: c.Continue, notes: c.Notes, drums: c.Drums,
		onsets: c.Onsets, width: c.NoteWidth, temperature: c.Temperature, topK: c.TopK,
		guidance: music.Guidance{Style: c.CFGStyle, Notes: c.CFGNotes, Drums: c.CFGDrums}, seed: c.Seed,
		steps: c.Steps, cfg: c.CFG, shift: c.Shift, textFormat: c.TextFormat, ditFormat: c.DitFormat}
	if c.Duration != nil {
		o.duration, o.durationSet = *c.Duration, true
	}
	if c.Fade != nil {
		o.fade, o.fadeSet = *c.Fade, true
	}
	if o.dir == "" {
		o.dir = chat.ModelsDir()
	}
	if o.cacheDir == "" {
		o.cacheDir = cache.Root()
	}
	return run(ctx, o, c.Args)
}

// ListCmd is "hwnpu-audio list": the audio models of the definitions
// (models.yaml), whether they are downloaded, and what each use and family picks for each
// machine's memory.
type ListCmd struct {
	Kind string `arg:"" optional:"" enum:"music,sfx,speech,transcribe,pitch,notes," default:"" help:"Only music, sfx, speech, transcribe, pitch or notes models."`
	Dir  string `type:"path" placeholder:"DIR" help:"Where the models are kept (as hwnpu-chat's --dir)."`
}

// Run lists the models.
func (c *ListCmd) Run() error {
	if err := chat.LoadUserModels("", os.Stderr); err != nil {
		return err
	}
	dir := c.Dir
	if dir == "" {
		dir = chat.ModelsDir()
	}
	kinds := []chat.Kind{chat.KindMusic, chat.KindSFX, chat.KindSpeech, chat.KindTranscribe, chat.KindPitch, chat.KindNotes}
	if c.Kind != "" {
		kinds = []chat.Kind{chat.Kind(c.Kind)}
	}
	chat.List(os.Stdout, dir, kinds...)
	return nil
}

// resolve is the model of the definitions --model means (music or sound effects); when there is
// none, what there is is listed before the error.
func resolve(name string, lw io.Writer) (chat.Model, error) {
	if err := chat.LoadUserModels("", lw); err != nil {
		return chat.Model{}, err
	}
	m, err := chat.ResolveForKinds(name, lw, chat.KindMusic, chat.KindSFX)
	if e := (*chat.NoModelError)(nil); errors.As(err, &e) {
		err = fmt.Errorf("no music or sound effects model, family or use %q (hwnpu-audio list shows them)", name)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "no model %q; the music and sound effects models of the definitions:\n\n", name)
		chat.List(os.Stderr, chat.ModelsDir(), chat.KindMusic, chat.KindSFX)
		fmt.Fprintln(os.Stderr)
	}
	return m, err
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
		return fmt.Errorf("--drums %q: auto, on or off", o.drums)
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
	m, err := resolve(o.use, lw)
	if err != nil {
		return err
	}
	if m.Kind == chat.KindSFX {
		return runSFX(ctx, o, m, args, lw)
	}
	if len(args) == 0 && o.codes == "" && o.cont == "" {
		return errors.New("music wants a style: texts or WAV files, --codes or --continue")
	}
	backend := llm.BackendCoreML
	var dev *hwnpu.Device
	if !o.cpu {
		var device string
		if dev, device, err = llm.OpenDevice(ctx, "", o.cacheDir); err != nil {
			return err
		}
		defer dev.Close()
		if backend = llm.ProgramBackend(device); backend == "" {
			return fmt.Errorf("%s runs neither Core ML packages nor OpenVINO IR (--cpu runs Magenta RealTime 2 on the CPU)", device)
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
				return fmt.Errorf("--codes %q: 12 codes from 0 to 1023", o.codes)
			}
			codes = append(codes, c)
		}
		if len(codes) != music.StyleCodes {
			return fmt.Errorf("--codes has %d codes, not %d", len(codes), music.StyleCodes)
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
		console.Running()
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
		logf("style %s: %v (--codes takes them)", strings.Join(names, " + "), codes)
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
		return fmt.Errorf("--duration %v: more than 0", o.duration)
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

	console.Preparing()
	t0 := time.Now()
	units := map[string]string{"npu": "", "gpu": "gpu", "cpu": "cpu"}
	ss, ok := units[o.decoderOn]
	if !ok {
		return fmt.Errorf("--decoder-on %q: npu, gpu or cpu", o.decoderOn)
	}
	if ss != "" && backend != llm.BackendCoreML {
		return fmt.Errorf("--decoder-on %s: Core ML's units, not Intel's NPU", o.decoderOn)
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
	case !slices.ContainsFunc(g.Placement(), func(l string) bool { return !strings.Contains(l, "placement not known") }):
		where = "placement not known"
	default:
		where = "NOT all on the NPU"
	}
	done(fmt.Sprintf("in %s, %s", console.Duration(time.Since(t0)), where))
	if err := checkNPU(lw, o, g.Placement()); err != nil {
		return err
	}
	console.Running()
	if o.seed == 0 {
		o.seed = rand.Uint64()
	}
	logf("seed %d", o.seed)
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

// checkNPU reports the programs not all on the NPU, and stops unless --allow-off-npu. A program
// whose placement Core ML could not tell (MLComputePlan fails or does not finish while the Neural
// Engine is busy compiling) is reported and does not stop it.
func checkNPU(lw io.Writer, o options, off []string) error {
	if o.cpu {
		return nil
	}
	n := 0
	for _, l := range off {
		if strings.Contains(l, "placement not known") {
			fmt.Fprintln(lw, "  going on:", l)
			continue
		}
		fmt.Fprintln(lw, "  NOT all on the NPU:", l)
		n++
	}
	if n == 0 || o.offNPU {
		return nil
	}
	return fmt.Errorf("%d programs are not all on the NPU, which has made a Mac sluggish and failed; --allow-off-npu goes on anyway", n)
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
