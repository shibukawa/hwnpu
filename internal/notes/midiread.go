package notes

import (
	"encoding/binary"
	"errors"
	"fmt"
	"sort"
)

// MIDINote is a note of a MIDI file: its start and end in seconds, pitch, velocity (1 to 127)
// and channel (0 to 15; 9 is General MIDI's drums).
type MIDINote struct {
	Start, End float64
	Pitch      int
	Velocity   int
	Channel    int
}

// ReadMIDI reads the notes of a standard MIDI file (format 0 or 1, ticks a beat; the tempo
// changes of any track apply to all), sorted by start. A note on with velocity 0 ends a note; a
// note left on ends with the file.
func ReadMIDI(data []byte) ([]MIDINote, error) {
	if len(data) < 14 || string(data[:4]) != "MThd" {
		return nil, errors.New("midi: not a standard MIDI file")
	}
	hl := int(binary.BigEndian.Uint32(data[4:]))
	if hl < 6 || 8+hl > len(data) {
		return nil, errors.New("midi: short header")
	}
	format, tracks, division := binary.BigEndian.Uint16(data[8:]), int(binary.BigEndian.Uint16(data[10:])), int(binary.BigEndian.Uint16(data[12:]))
	if format > 1 {
		return nil, fmt.Errorf("midi: format %d is not supported", format)
	}
	if division&0x8000 != 0 || division == 0 {
		return nil, errors.New("midi: SMPTE time is not supported")
	}
	type event struct {
		tick              int64
		kind              int // 0 tempo, 1 note on, 2 note off
		value, pitch, vel int
		ch                int
	}
	var events []event
	p := 8 + hl
	for t := 0; t < tracks && p+8 <= len(data); t++ {
		if string(data[p:p+4]) != "MTrk" {
			return nil, fmt.Errorf("midi: track %d: no MTrk", t)
		}
		n := int(binary.BigEndian.Uint32(data[p+4:]))
		body := data[p+8 : min(p+8+n, len(data))]
		p += 8 + n
		var tick int64
		var status byte
		i := 0
		vlq := func() (int64, error) {
			var v int64
			for k := 0; k < 4; k++ {
				if i >= len(body) {
					return 0, errors.New("midi: truncated")
				}
				b := body[i]
				i++
				v = v<<7 | int64(b&0x7f)
				if b&0x80 == 0 {
					return v, nil
				}
			}
			return 0, errors.New("midi: bad variable length")
		}
		for i < len(body) {
			d, err := vlq()
			if err != nil {
				return nil, err
			}
			tick += d
			if i >= len(body) {
				break
			}
			b := body[i]
			if b&0x80 != 0 {
				status = b
				i++
			} else if status == 0 {
				return nil, errors.New("midi: running status without a status")
			}
			switch {
			case status == 0xff: // meta
				if i >= len(body) {
					return nil, errors.New("midi: truncated meta")
				}
				typ := body[i]
				i++
				l, err := vlq()
				if err != nil {
					return nil, err
				}
				if i+int(l) > len(body) {
					return nil, errors.New("midi: truncated meta")
				}
				if typ == 0x51 && l == 3 {
					us := int(body[i])<<16 | int(body[i+1])<<8 | int(body[i+2])
					events = append(events, event{tick: tick, kind: 0, value: us})
				}
				i += int(l)
				status = 0
				if typ == 0x2f {
					i = len(body)
				}
			case status == 0xf0 || status == 0xf7: // sysex
				l, err := vlq()
				if err != nil {
					return nil, err
				}
				i += int(l)
				status = 0
			default:
				hi, ch := status&0xf0, int(status&0x0f)
				size := 2
				if hi == 0xc0 || hi == 0xd0 {
					size = 1
				}
				if i+size > len(body) {
					return nil, errors.New("midi: truncated event")
				}
				a := int(body[i])
				var v int
				if size == 2 {
					v = int(body[i+1])
				}
				i += size
				switch {
				case hi == 0x90 && v > 0:
					events = append(events, event{tick: tick, kind: 1, pitch: a, vel: v, ch: ch})
				case hi == 0x80 || hi == 0x90:
					events = append(events, event{tick: tick, kind: 2, pitch: a, ch: ch})
				}
			}
		}
	}
	// in time order, tempo changes first, then note offs before note ons at the same tick
	sort.SliceStable(events, func(a, b int) bool {
		if events[a].tick != events[b].tick {
			return events[a].tick < events[b].tick
		}
		ka, kb := events[a].kind, events[b].kind
		order := map[int]int{0: 0, 2: 1, 1: 2}
		return order[ka] < order[kb]
	})
	tempo := 500000.0
	var lastTick int64
	var now float64
	open := map[[2]int][]int{} // (channel, pitch) -> indexes of open notes
	var out []MIDINote
	for _, e := range events {
		now += float64(e.tick-lastTick) * tempo / 1e6 / float64(division)
		lastTick = e.tick
		switch e.kind {
		case 0:
			tempo = float64(e.value)
		case 1:
			k := [2]int{e.ch, e.pitch}
			open[k] = append(open[k], len(out))
			out = append(out, MIDINote{Start: now, End: -1, Pitch: e.pitch, Velocity: e.vel, Channel: e.ch})
		case 2:
			k := [2]int{e.ch, e.pitch}
			if l := open[k]; len(l) > 0 {
				out[l[0]].End = now
				open[k] = l[1:]
			}
		}
	}
	for i := range out {
		if out[i].End < 0 {
			out[i].End = now
		}
	}
	sort.SliceStable(out, func(a, b int) bool { return out[a].Start < out[b].Start })
	return out, nil
}
