package sound

import (
	"bytes"
	"math"
	"slices"
	"testing"
	"time"
)

func TestUses(t *testing.T) {
	for _, c := range []struct {
		role, kind string
		kinds      []string
	}{
		{Music("music").Role(), "music", Music("music").Kinds()},
		{Effect("sfx").Role(), "sfx", Effect("sfx").Kinds()},
		{Pitch("pitch").Role(), "pitch", Pitch("pitch").Kinds()},
		{Notes("notes").Role(), "notes", Notes("notes").Kinds()},
	} {
		if !slices.Equal(c.kinds, []string{c.kind}) {
			t.Errorf("the role %s runs %v, want %s", c.role, c.kinds, c.kind)
		}
	}
}

func TestMusicStats(t *testing.T) {
	s := MusicStats{Frames: 50, Took: time.Second}
	if s.Seconds() != 2 || s.RealTime() != 2 {
		t.Errorf("%v s, %v times real time", s.Seconds(), s.RealTime())
	}
}

func TestClip(t *testing.T) {
	c := Clip{Channels: [][]float32{{1, 0, 0.5}, {0, 0, 0.5}}, Rate: 2}
	if c.Seconds() != 1.5 || (Clip{}).Seconds() != 0 {
		t.Errorf("%v s", c.Seconds())
	}
}

// TestMIDI checks that notes written as MIDI read back.
func TestMIDI(t *testing.T) {
	ns := []Note{{Start: 0, End: 0.5, Pitch: 60, Amplitude: 0.5}, {Start: 0.5, End: 1, Pitch: 64, Amplitude: 1}}
	var b bytes.Buffer
	if err := WriteMIDI(&b, ns); err != nil {
		t.Fatal(err)
	}
	got, err := ReadMIDI(b.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Pitch != 60 || got[1].Pitch != 64 || math.Abs(got[1].End-1) > 0.01 {
		t.Errorf("read back: %+v", got)
	}
	if ns[0].Velocity() != 64 || ns[1].Velocity() != 127 {
		t.Errorf("velocities %d, %d", ns[0].Velocity(), ns[1].Velocity())
	}
}
