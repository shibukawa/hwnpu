package sound

import (
	"context"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/llm"
	"github.com/shibukawa/bonsaiai/internal/notes"
)

// first is the first of ps, "" for none.
func first(ps []string) string {
	if len(ps) == 0 {
		return ""
	}
	return ps[0]
}

// logf writes lines to w (nil for nowhere).
func logf(w io.Writer) func(string, ...any) {
	return func(format string, args ...any) {
		if w != nil {
			fmt.Fprintf(w, format+"\n", args...)
		}
	}
}

// programFormat is the program format the music and the sound effects are converted for on the
// NPU n: Core AI's on a Mac, OpenVINO IR on Intel's NPU. Others run neither (AMD's runs exports
// of IRON instead, which the roles make apart).
func programFormat(n airole.NPU) (string, error) {
	switch n.Format {
	case llm.BackendCoreAI, llm.BackendIntel:
		return n.Format, nil
	case "":
		return "", fmt.Errorf("no NPU here that runs Core AI's programs or OpenVINO IR (a model of device cpu runs on the CPU)")
	}
	return "", fmt.Errorf("the NPU here runs neither Core AI's programs nor OpenVINO IR (%s; a model of device cpu runs on the CPU)", n.Format)
}

// openNPU opens the NPU n as the music and the sound effects run on it (not AMD's), with the
// compile cache in cacheDir.
func openNPU(ctx context.Context, n airole.NPU, cacheDir string) (*bonsaiai.Device, error) {
	if _, err := programFormat(n); err != nil {
		return nil, err
	}
	dev, format, err := llm.OpenDevice(ctx, n.Device, cacheDir)
	if err != nil {
		if n.Format == llm.BackendCoreAI {
			return nil, fmt.Errorf("the models run on Apple's Neural Engine through Core AI, which needs macOS 27 (a model of device cpu runs on the CPU): %w", err)
		}
		return nil, err
	}
	if n.Format != llm.BackendCoreAI && format != llm.BackendIntel {
		dev.Close()
		return nil, fmt.Errorf("%s runs neither Core AI's programs nor OpenVINO IR (a model of device cpu runs on the CPU)", dev.Info().Backend)
	}
	return dev, nil
}

// OffNPUError is the error of a model part of whose programs the NPU's compiler placed off the
// NPU (on the GPU or the CPU), which has made a Mac sluggish and failed: ai.AllowOffNPU loads it
// anyway.
type OffNPUError struct {
	// Programs are the programs not all on the NPU, with what runs elsewhere.
	Programs []string
}

func (e *OffNPUError) Error() string {
	return fmt.Sprintf("sound: %d programs are not all on the NPU, which has made a Mac sluggish and failed; ai.AllowOffNPU (bonsaiai-audio --allow-off-npu) goes on anyway", len(e.Programs))
}

// checkNPU logs the programs not all on the NPU (off: Placement's lines) and refuses them with
// an *OffNPUError unless allow. A program whose placement could not be told is logged and does
// not count.
func checkNPU(log io.Writer, allow bool, off []string) error {
	var bad []string
	for _, l := range off {
		if strings.Contains(l, "placement not known") {
			logf(log)("  going on: %s", l)
			continue
		}
		logf(log)("  NOT all on the NPU: %s", l)
		bad = append(bad, l)
	}
	if len(bad) == 0 || allow {
		return nil
	}
	return &OffNPUError{Programs: bad}
}

// fadeOut fades the last n frames of interleaved audio of channels channels out (all of it
// when it is shorter): a raised cosine from full to silence.
func fadeOut(pcm []float32, channels, n int) {
	frames := len(pcm) / channels
	n = min(n, frames)
	tail := pcm[(frames-n)*channels:]
	for i := range n {
		g := float32(0.5 * (1 + math.Cos(math.Pi*float64(i)/float64(n))))
		for c := range channels {
			tail[i*channels+c] *= g
		}
	}
}

// MIDINote is a note of a MIDI file: its start and end in seconds, its pitch, its velocity (1
// to 127) and its channel (0 to 15; 9 is General MIDI's drums).
type MIDINote struct {
	Start, End float64
	Pitch      int
	Velocity   int
	Channel    int
}

// ReadMIDI reads the notes of a standard MIDI file (format 0 or 1), sorted by start: the notes
// music plays (MusicOptions.Notes).
func ReadMIDI(data []byte) ([]MIDINote, error) {
	ns, err := notes.ReadMIDI(data)
	if err != nil {
		return nil, err
	}
	out := make([]MIDINote, len(ns))
	for i, n := range ns {
		out[i] = MIDINote(n)
	}
	return out, nil
}

// Note is a note found in music (NoteTranscriber): its start and end in seconds, its MIDI
// pitch, its amplitude (0 to 1) and its pitch bends, in thirds of a semitone, evenly over the
// note.
type Note struct {
	Start     float64 `json:"start"`
	End       float64 `json:"end"`
	Pitch     int     `json:"pitch"`
	Amplitude float64 `json:"amplitude"`
	Bends     []int   `json:"bends,omitempty"`
}

// Velocity is the note's MIDI velocity, from its amplitude.
func (n Note) Velocity() int { return int(math.RoundToEven(127 * n.Amplitude)) }

// WriteMIDI writes notes as a standard MIDI file, with their pitch bends.
func WriteMIDI(w io.Writer, ns []Note) error {
	in := make([]notes.Note, len(ns))
	for i, n := range ns {
		in[i] = notes.Note(n)
	}
	return notes.WriteMIDI(w, in)
}
