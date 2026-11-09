package notes

import (
	"bytes"
	"encoding/binary"
	"io"
	"math"
	"slices"
	"sort"
)

// WriteMIDI writes the notes as a standard MIDI file as note_events_to_midi does: format 1 at
// 120 beats a minute and 220 ticks a beat (pretty_midi's), one Electric Piano 1 track, a
// velocity of 127 times the amplitude, and the pitch bends (2 semitones full scale) of the notes
// that overlap no other note.
func WriteMIDI(w io.Writer, notes []Note) error {
	const ticksPerBeat, tempo = 220, 500000 // µs a beat
	tick := func(s float64) int { return int(math.RoundToEven(s * 1e6 / tempo * ticksPerBeat)) }
	sorted := slices.Clone(notes)
	sort.SliceStable(sorted, func(i, j int) bool {
		a, b := sorted[i], sorted[j]
		if a.Start != b.Start {
			return a.Start < b.Start
		}
		if a.End != b.End {
			return a.End < b.End
		}
		return a.Pitch < b.Pitch
	})
	bends := make([]bool, len(sorted)) // whether a note keeps its bends
	for i := range bends {
		bends[i] = len(sorted[i].Bends) > 0
	}
	for i := range sorted {
		for j := i + 1; j < len(sorted) && sorted[j].Start < sorted[i].End; j++ {
			bends[i], bends[j] = false, false
		}
	}
	type ev struct {
		tick, order int
		data        []byte
	}
	var evs []ev
	for i, n := range sorted {
		vel := byte(min(127, max(0, int(math.RoundToEven(127*n.Amplitude)))))
		evs = append(evs, ev{tick(n.Start), 1, []byte{0x90, byte(n.Pitch), vel}}, ev{tick(n.End), 0, []byte{0x80, byte(n.Pitch), 0}})
		if !bends[i] {
			continue
		}
		for k, b := range n.Bends {
			t := n.Start
			if len(n.Bends) > 1 {
				t += (n.End - n.Start) * float64(k) / float64(len(n.Bends)-1)
			}
			v := min(8191, max(-8192, int(math.RoundToEven(float64(b)*4096/3))))
			u := v + 8192
			evs = append(evs, ev{tick(t), 2, []byte{0xE0, byte(u & 0x7f), byte(u >> 7)}})
		}
	}
	sort.SliceStable(evs, func(i, j int) bool {
		if evs[i].tick != evs[j].tick {
			return evs[i].tick < evs[j].tick
		}
		return evs[i].order < evs[j].order
	})
	var track bytes.Buffer
	varint := func(b *bytes.Buffer, v int) {
		var s [5]byte
		n := 0
		s[n] = byte(v & 0x7f)
		for v >>= 7; v > 0; v >>= 7 {
			n++
			s[n] = byte(v&0x7f) | 0x80
		}
		for ; n >= 0; n-- {
			b.WriteByte(s[n])
		}
	}
	varint(&track, 0)
	track.Write([]byte{0xC0, 4}) // Electric Piano 1
	last := 0
	for _, e := range evs {
		varint(&track, e.tick-last)
		track.Write(e.data)
		last = e.tick
	}
	varint(&track, 0)
	track.Write([]byte{0xFF, 0x2F, 0})
	var tempoTrack bytes.Buffer
	tempoTrack.Write([]byte{0, 0xFF, 0x51, 3, byte(tempo >> 16 & 0xff), byte(tempo >> 8 & 0xff), byte(tempo & 0xff)})
	tempoTrack.Write([]byte{0, 0xFF, 0x2F, 0})
	var out bytes.Buffer
	out.WriteString("MThd")
	binary.Write(&out, binary.BigEndian, []uint32{6})
	binary.Write(&out, binary.BigEndian, []uint16{1, 2, ticksPerBeat})
	for _, t := range []*bytes.Buffer{&tempoTrack, &track} {
		out.WriteString("MTrk")
		binary.Write(&out, binary.BigEndian, uint32(t.Len()))
		out.Write(t.Bytes())
	}
	_, err := w.Write(out.Bytes())
	return err
}
