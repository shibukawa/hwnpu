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
// The commands run the models of ai/sound (music, sound effects, pitch, notes) and ai/speech
// (speech, transcription) as an app would (app.go); what is theirs alone is reading and
// writing the files, playing the sound, and the log.
//
// On a Mac hwnpu-audio runs on Apple's Neural Engine through Core AI (macOS 27) alone: each
// model is converted for Core AI (exports NAME.coreai beside the checkpoint, as hwnpu-chat's and
// hwnpu-image's, llm.CoreAIExport) and kept compiled in Core AI's own cache. Core ML is not used on the Neural Engine; --cpu runs Core ML exports with the CPU
// interpreter, as do pitch and notes on the CPU without ONNX Runtime.
//
// tts (TTSCmd), transcribe (TranscribeCmd), pitch (PitchCmd) and notes (NotesCmd) are described
// at their commands.
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"time"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/ai/sound"
	"github.com/shibukawa/hwnpu/internal/cachecmd"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/console"
	shared "github.com/shibukawa/hwnpu/internal/store"
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
	ctx, stop := console.Interrupt() // Ctrl-C ends the command, and the process when the command does not return
	defer stop()
	var c cli
	k := kong.Parse(&c,
		kong.Name("hwnpu-audio"),
		kong.Description("Audio models on an NPU: music, sound effects, speech, transcription, pitch and notes."),
		kong.UsageOnError(),
		kong.Vars{
			"models_usage": catalog.ModelsUsage,
			"onset":        fmt.Sprint(sound.DefaultNotesOptions.Onset),
			"frame":        fmt.Sprint(sound.DefaultNotesOptions.Frame),
			"min_note":     fmt.Sprint(sound.DefaultNotesOptions.MinNoteMS),
			"melodia":      fmt.Sprint(!sound.DefaultNotesOptions.NoMelodia),
		},
		kong.BindTo(ctx, (*context.Context)(nil)))
	console.MeasureEnergy(console.Log(os.Stderr)) // HWNPU_ENERGY: the idle power, then each phase's energy in the last lines
	console.RecordRun()                           // for hwnpu issue: the arguments, the log and how it ended
	err := console.EndInterrupted(k.Run())
	if err == nil {
		console.Done(console.Log(os.Stderr)) // the times, when the command said console.Running; HWNPU_PROFILE's profile
	}
	shared.Wait(console.Log(os.Stderr)) // what was made here goes in the sources written back to (after the times)
	k.FatalIfErrorf(err)
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
	CPU      bool           `name:"cpu" help:"Run the programs with the CPU interpreter instead of the NPU (slow)."`
	OffNPU   bool           `name:"allow-off-npu" help:"Go on when part of a program runs off the NPU (on the GPU or the CPU; that has made a Mac sluggish and failed)."`

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

	Steps      int     `group:"Sound effects" default:"100" help:"Flow-matching steps (each two calls of the DiT; fewer is faster and rougher)."`
	CFG        float64 `group:"Sound effects" name:"cfg" default:"4" help:"How strongly the description guides (classifier-free guidance)."`
	Shift      float64 `group:"Sound effects" default:"5" help:"The sigma schedule's shift."`
	TextFormat string  `group:"Sound effects" placeholder:"FORMAT" help:"The text encoder's weight format, fp16 or lut8 (default: the model's, by its machine)."`
	DitFormat  string  `group:"Sound effects" placeholder:"FORMAT" help:"The DiT's weight format, lut8 (default) or fp16."`
}

// ListCmd is "hwnpu-audio list": the audio models of the definitions
// (models.yaml), whether they are downloaded, and what each use and family picks for each
// machine's memory.
type ListCmd struct {
	Kind string `arg:"" optional:"" enum:"music,sfx,speech,transcribe,pitch,notes," default:"" help:"Only music, sfx, speech, transcribe, pitch or notes models."`
}

// Run lists the models of the definitions the commands' store reads (ai.Store.Catalog), as
// hwnpu-chat list shows them.
func (c *ListCmd) Run(ctx context.Context) error {
	kinds := []string{}
	for _, k := range []catalog.Kind{catalog.KindMusic, catalog.KindSFX, catalog.KindSpeech, catalog.KindTranscribe, catalog.KindPitch, catalog.KindNotes} {
		kinds = append(kinds, string(k))
	}
	if c.Kind != "" {
		kinds = []string{c.Kind}
	}
	return cachecmd.List(ctx, os.Stdout, nil, kinds...)
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
	binary.LittleEndian.PutUint32(h[24:], sound.MusicRate)
	binary.LittleEndian.PutUint32(h[28:], 4*sound.MusicRate)
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
