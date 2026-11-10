package music

import (
	"math"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/internal/notes"
)

func TestPianoRoll(t *testing.T) {
	ns := []notes.MIDINote{
		{Start: 0, End: 0.1, Pitch: 60},                // frames 0-2
		{Start: 0.05, End: 0.07, Pitch: 64},            // within frame 1
		{Start: 0, End: 1, Pitch: 36, Channel: 9},      // drums: left out
		{Start: 0.12, End: 0.2, Pitch: 67, Channel: 1}, // frames 3-4
	}
	roll := PianoRoll(ns, 0, 6, RollOptions{})
	at := func(f int) map[int]int {
		m := map[int]int{}
		for p, s := range roll[f] {
			if s != NoteMasked {
				m[p] = s
			}
		}
		return m
	}
	for f, want := range []map[int]int{{60: NoteOn}, {60: NoteOn, 64: NoteOn}, {60: NoteOn}, {67: NoteOn}, {67: NoteOn}, {}} {
		if got := at(f); !mapsEqual(got, want) {
			t.Errorf("frame %d: %v, want %v", f, got, want)
		}
	}
	roll = PianoRoll(ns, 0, 6, RollOptions{Onsets: true, Width: 1})
	if got := at(0); !mapsEqual(got, map[int]int{59: NoteOff, 60: NoteOnset, 61: NoteOff}) {
		t.Errorf("onsets, width 1, frame 0: %v", got)
	}
	if got := at(2); !mapsEqual(got, map[int]int{59: NoteOff, 60: NoteHeld, 61: NoteOff}) {
		t.Errorf("onsets, width 1, frame 2: %v", got)
	}
	roll = PianoRoll(ns, 0, 6, RollOptions{Width: 127})
	if n := slices.Index(roll[5], NoteMasked); n >= 0 {
		t.Errorf("width 127: pitch %d masked", n)
	}
}

func mapsEqual(a, b map[int]int) bool {
	if len(a) != len(b) {
		return false
	}
	for k, v := range a {
		if w, ok := b[k]; !ok || w != v {
			return false
		}
	}
	return true
}

func TestMix(t *testing.T) {
	a, b := make([]float32, StyleDim), make([]float32, StyleDim)
	a[0], b[1] = 28, 1
	if got := Mix([][]float32{a}, []float64{3}); got[0] != 28 || got[1] != 0 {
		t.Errorf("one embedding: %v", got[:2])
	}
	got := Mix([][]float32{a, b}, []float64{1, 1})
	// the directions halfway, at the mean length 14.5
	if d := math.Abs(float64(got[0]) - 14.5/math.Sqrt2); d > 1e-4 || math.Abs(float64(got[1]-got[0])) > 1e-4 {
		t.Errorf("two embeddings: %v", got[:2])
	}
}
