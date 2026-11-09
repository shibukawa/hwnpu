package notes

import (
	"math"
	"slices"
)

// Options are model_output_to_notes's settings (basic_pitch.inference.predict's defaults by
// DefaultOptions).
type Options struct {
	// OnsetThreshold is the least onset activation of a note's start; FrameThreshold the least
	// note activation that keeps a note on.
	OnsetThreshold, FrameThreshold float64
	// MinNoteMS is the shortest note.
	MinNoteMS float64
	// MinHz and MaxHz, when not 0, drop the notes outside them.
	MinHz, MaxHz float64
	// Melodia also takes notes from the strongest activations left without an onset.
	Melodia bool
}

// DefaultOptions are predict's defaults.
var DefaultOptions = Options{OnsetThreshold: 0.5, FrameThreshold: 0.3, MinNoteMS: 127.7, Melodia: true}

// Note is a note: its start and end in seconds, its MIDI pitch, its amplitude (0 to 1, the mean
// note activation) and its pitch bends, in thirds of a semitone, evenly over the note.
type Note struct {
	Start     float64 `json:"start"`
	End       float64 `json:"end"`
	Pitch     int     `json:"pitch"`
	Amplitude float64 `json:"amplitude"`
	Bends     []int   `json:"bends,omitempty"`
}

const (
	energyTolerance = 11
	maxFreqIdx      = Pitches - 1
	alignmentOffset = 0.0018
)

func hzToMIDI(f float64) float64 { return 12*(math.Log2(f)-math.Log2(440)) + 69 }

// Notes makes the notes of the activations, as model_output_to_notes does with pitch bends.
func Notes(a Activations, o Options) []Note {
	n := a.Frames
	frames := slices.Clone(a.Note)
	onsets := slices.Clone(a.Onset)
	// constrain_frequency
	lo, hi := 0, Pitches
	if o.MinHz > 0 {
		lo = int(math.RoundToEven(hzToMIDI(o.MinHz) - MIDIOffset))
	}
	if o.MaxHz > 0 {
		hi = int(math.RoundToEven(hzToMIDI(o.MaxHz) - MIDIOffset))
	}
	for t := range n {
		for f := range Pitches {
			if f < lo || f >= hi {
				frames[t*Pitches+f], onsets[t*Pitches+f] = 0, 0
			}
		}
	}
	onsets = inferOnsets(onsets, frames, n)
	minLen := int(math.RoundToEven(o.MinNoteMS / 1000 * (float64(SampleRate) / Hop)))
	at := func(m []float64, t, f int) float64 { return m[t*Pitches+f] }

	// the onsets' peaks in time (scipy.signal.argrelmax over axis 0) of at least the threshold,
	// latest first, and the higher pitch first within a frame
	type idx struct{ t, f int }
	var peaks []idx
	for t := 1; t < n-1; t++ {
		for f := range Pitches {
			v := at(onsets, t, f)
			if v > at(onsets, t-1, f) && v > at(onsets, t+1, f) && v >= o.OnsetThreshold {
				peaks = append(peaks, idx{t, f})
			}
		}
	}
	slices.Reverse(peaks)
	remaining := slices.Clone(frames)
	type event struct {
		start, end, pitch int
		amp               float64
	}
	var events []event
	mean := func(s, e, f int) float64 {
		var sum float64
		for t := s; t < e; t++ {
			sum += at(frames, t, f)
		}
		return sum / float64(e-s)
	}
	clearNote := func(t, f int) {
		remaining[t*Pitches+f] = 0
		if f < maxFreqIdx {
			remaining[t*Pitches+f+1] = 0
		}
		if f > 0 {
			remaining[t*Pitches+f-1] = 0
		}
	}
	for _, p := range peaks {
		start, f := p.t, p.f
		if start >= n-1 {
			continue
		}
		i, k := start+1, 0
		for i < n-1 && k < energyTolerance {
			if remaining[i*Pitches+f] < o.FrameThreshold {
				k++
			} else {
				k = 0
			}
			i++
		}
		i -= k
		if i-start <= minLen {
			continue
		}
		for t := start; t < i; t++ {
			clearNote(t, f)
		}
		events = append(events, event{start, i, f + MIDIOffset, mean(start, i, f)})
	}
	if o.Melodia {
		for {
			best := 0
			for j, v := range remaining {
				if v > remaining[best] {
					best = j
				}
			}
			if remaining[best] <= o.FrameThreshold {
				break
			}
			mid, f := best/Pitches, best%Pitches
			remaining[best] = 0
			i, k := mid+1, 0
			for i < n-1 && k < energyTolerance {
				if remaining[i*Pitches+f] < o.FrameThreshold {
					k++
				} else {
					k = 0
				}
				clearNote(i, f)
				i++
			}
			end := i - 1 - k
			i, k = mid-1, 0
			for i > 0 && k < energyTolerance {
				if remaining[i*Pitches+f] < o.FrameThreshold {
					k++
				} else {
					k = 0
				}
				clearNote(i, f)
				i--
			}
			start := i + 1 + k
			if end-start <= minLen {
				continue
			}
			events = append(events, event{start, end, f + MIDIOffset, mean(start, end, f)})
		}
	}
	times := frameTimes(n)
	var out []Note
	for _, e := range events {
		out = append(out, Note{Start: times[e.start], End: times[e.end], Pitch: e.pitch, Amplitude: e.amp,
			Bends: pitchBends(a.Contour, e.start, e.end, e.pitch)})
	}
	return out
}

// inferOnsets is get_infered_onsets: the onsets, or where larger the rises of the note
// activations over 1 and 2 frames (the smaller of the two), scaled to the onsets' maximum.
func inferOnsets(onsets, frames []float64, n int) []float64 {
	diff := make([]float64, len(frames))
	prev := func(t, f, d int) float64 {
		if t-d < 0 {
			return 0
		}
		return frames[(t-d)*Pitches+f]
	}
	hiDiff, hiOnset := 0.0, 0.0
	for t := range n {
		for f := range Pitches {
			v := frames[t*Pitches+f]
			d := math.Max(math.Min(v-prev(t, f, 1), v-prev(t, f, 2)), 0)
			if t < 2 {
				d = 0
			}
			diff[t*Pitches+f] = d
			hiDiff = math.Max(hiDiff, d)
			hiOnset = math.Max(hiOnset, onsets[t*Pitches+f])
		}
	}
	out := make([]float64, len(onsets))
	for i := range out {
		d := 0.0
		if hiDiff > 0 {
			d = hiOnset * diff[i] / hiDiff
		}
		out[i] = math.Max(onsets[i], d)
	}
	return out
}

// pitchBends is get_pitch_bends of one note: in each frame, the contour bin within 25 thirds of a
// semitone of the note's, weighted by a Gaussian of 5, that is the largest, relative to the note's.
func pitchBends(contour []float64, start, end, pitch int) []int {
	const tol = 25
	const length = 2*tol + 1
	gauss := make([]float64, length)
	for i := range gauss {
		x := float64(i-tol) / 5
		gauss[i] = math.Exp(-0.5 * x * x)
	}
	fi := 3 * (pitch - MIDIOffset)
	fs, fe := max(fi-tol, 0), min(ContourBins, fi+tol+1)
	g0 := max(0, tol-fi)
	shift := tol - max(0, tol-fi)
	bends := []int{}
	for t := start; t < end; t++ {
		best, bv := 0, math.Inf(-1)
		for j := fs; j < fe; j++ {
			v := contour[t*ContourBins+j] * gauss[g0+j-fs]
			if v > bv {
				best, bv = j-fs, v
			}
		}
		bends = append(bends, best-shift)
	}
	return bends
}

// frameTimes is model_frames_to_time: the frames' times, less the windows' small misalignment.
func frameTimes(n int) []float64 {
	offset := float64(Hop)/SampleRate*(2*(SampleRate/Hop)-float64(WindowSamples)/Hop) + alignmentOffset
	out := make([]float64, n)
	for i := range out {
		out[i] = float64(i)*Hop/SampleRate - offset*math.Floor(float64(i)/float64(2*(SampleRate/Hop)))
	}
	return out
}
