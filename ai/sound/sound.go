package sound

import (
	"fmt"
	"io"
	"math"

	"github.com/shibukawa/hwnpu/internal/soundwire"
)

// OffNPUError is the error of a model part of whose programs the NPU's compiler placed off the
// NPU (on the GPU or the CPU), which has made a Mac sluggish and failed: ai.AllowOffNPU loads it
// anyway.
type OffNPUError struct {
	// Programs are the programs not all on the NPU, with what runs elsewhere.
	Programs []string
}

func (e *OffNPUError) Error() string {
	return fmt.Sprintf("sound: %d programs are not all on the NPU, which has made a Mac sluggish and failed; ai.AllowOffNPU (hwnpu-audio --allow-off-npu) goes on anyway", len(e.Programs))
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
	ns, err := soundwire.ReadMIDI(data)
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
	in := make([]soundwire.Note, len(ns))
	for i, n := range ns {
		in[i] = soundwire.Note(n)
	}
	return soundwire.WriteMIDI(w, in)
}
