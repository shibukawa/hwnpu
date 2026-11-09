package music

import (
	"math"

	"github.com/shibukawa/hwnpu/internal/notes"
)

// The conditioning's layout: MusicCoCa's codes, the piano roll's 128 pitches, the drums, and the
// classifier-free guidance scales binned.
const (
	condNotes = StyleCodes
	condDrums = condNotes + 128
	condCFG   = condDrums + 1
)

// Note states of the piano roll (PIANOROLL_WITH_ONSETS): masked (the model decides), off, on
// after its start, on at its start, on (the model decides whether it starts).
const (
	NoteMasked = -1
	NoteOff    = 0
	NoteHeld   = 1
	NoteOnset  = 2
	NoteOn     = 3
)

// Drums states (DRUM_PIANOROLL): masked (the model decides), none, drums.
const (
	DrumsMasked = -1
	DrumsOff    = 0
	DrumsOn     = 1
)

// Guidance are the classifier-free guidance scales (-1 to 7) the conditioning carries as
// tokens: of the style, of the notes, of the drums.
type Guidance struct{ Style, Notes, Drums float64 }

// DefaultGuidance are magenta-rt's C++ runtime's defaults.
var DefaultGuidance = Guidance{Style: 3, Notes: 5, Drums: 1}

// NewConditioning is a conditioning of MusicCoCa codes (12) and guidance scales, the notes masked
// and the drums as given, as MagentaRT2System._build_conditioning makes it.
func NewConditioning(musiccoca []int, g Guidance, drums int) Conditioning {
	var c Conditioning
	for i := range c {
		c[i] = -1
	}
	copy(c[:StyleCodes], musiccoca)
	c[condDrums] = drums
	bin := func(v, step float64, maxBin int) int {
		v = math.Max(-1, math.Min(7, v))
		return max(0, min(maxBin, int(math.RoundToEven((v+1)/step))))
	}
	c[condCFG], c[condCFG+1] = bin(g.Style, 0.2, 40), bin(g.Notes, 0.2, 40)
	c[condCFG+2] = bin(g.Drums, 1, 8)
	return c
}

// WithNotes is c with the piano roll's states (NoteMasked, ...; 128 of them).
func (c Conditioning) WithNotes(states []int) Conditioning {
	copy(c[condNotes:condDrums], states)
	return c
}

// WithoutStyle is c with MusicCoCa's codes masked.
func (c Conditioning) WithoutStyle() Conditioning {
	for i := range StyleCodes {
		c[i] = -1
	}
	return c
}

// RollOptions say how notes become the piano roll's states, as the C++ runtime's
// populate_condition_tokens: Onsets tells a note's start (NoteOnset, then NoteHeld) instead of
// NoteOn throughout; Width unmasks the pitches within it of a sounding note as NoteOff (0: the
// sounding ones alone; 127 or more: all, NoteOff where silent).
type RollOptions struct {
	Onsets bool
	Width  int
}

// PianoRoll is the piano roll of the notes (the drums' channel 9 left out) for frames, 25 a
// second from the time from on: a note sounds in a frame when it overlaps it, and starts in it
// when its start is in it. A frame without a sounding note is all masked unless Width is 127 or
// more.
func PianoRoll(ns []notes.MIDINote, from float64, frames int, o RollOptions) [][]int {
	roll := make([][]int, frames)
	for f := range roll {
		t0, t1 := from+float64(f)/FrameRate, from+float64(f+1)/FrameRate
		state := make([]int, 128)
		for _, n := range ns {
			if n.Channel == 9 || n.Pitch < 0 || n.Pitch > 127 || n.End <= t0 || n.Start >= t1 {
				continue
			}
			s := NoteOn
			if o.Onsets {
				s = NoteHeld
				if n.Start >= t0 {
					s = NoteOnset
				}
			}
			if state[n.Pitch] != NoteOnset {
				state[n.Pitch] = s
			}
		}
		row := make([]int, 128)
		sounding := false
		for i, s := range state {
			if s != NoteOff {
				sounding = true
			}
			if o.Width >= 127 {
				row[i] = s
			} else {
				row[i] = NoteMasked
			}
		}
		if o.Width < 127 && sounding {
			for i, s := range state {
				if s == NoteOff {
					continue
				}
				for j := max(0, i-o.Width); j <= min(127, i+o.Width); j++ {
					if state[j] == NoteOff {
						row[j] = NoteOff
					}
				}
				row[i] = s
			}
		}
		roll[f] = row
	}
	return roll
}
