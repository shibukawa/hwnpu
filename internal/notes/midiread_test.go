package notes

import (
	"bytes"
	"math"
	"testing"
)

// What WriteMIDI writes, ReadMIDI reads back.
func TestReadMIDI(t *testing.T) {
	in := []Note{{Start: 0, End: 0.5, Pitch: 60, Amplitude: 1}, {Start: 0.25, End: 1.5, Pitch: 64, Amplitude: 0.5}, {Start: 1, End: 2, Pitch: 67, Amplitude: 0.8}}
	var b bytes.Buffer
	if err := WriteMIDI(&b, in); err != nil {
		t.Fatal(err)
	}
	got, err := ReadMIDI(b.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(in) {
		t.Fatalf("%d notes, want %d: %+v", len(got), len(in), got)
	}
	for i, n := range in {
		g := got[i]
		if g.Pitch != n.Pitch || math.Abs(g.Start-n.Start) > 0.003 || math.Abs(g.End-n.End) > 0.003 {
			t.Errorf("note %d: %+v, want %+v", i, g, n)
		}
	}
}
