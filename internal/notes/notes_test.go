package notes

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/onnx"
)

func readF32(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]float32, len(raw)/4)
	if err := binary.Read(bytes.NewReader(raw), binary.LittleEndian, out); err != nil {
		t.Fatal(err)
	}
	return out
}

func relErr(got, want []float32) float64 {
	var se, sw float64
	for i := range want {
		d := float64(got[i]) - float64(want[i])
		se += d * d
		sw += float64(want[i]) * float64(want[i])
	}
	return math.Sqrt(se / sw)
}

type refNote struct {
	Start, End float64
	Pitch      int
	Amplitude  float64
	Bends      []int
}

func (r *refNote) UnmarshalJSON(b []byte) error {
	var v []json.RawMessage
	if err := json.Unmarshal(b, &v); err != nil {
		return err
	}
	for i, p := range []any{&r.Start, &r.End, &r.Pitch, &r.Amplitude, &r.Bends} {
		if err := json.Unmarshal(v[i], p); err != nil {
			return err
		}
	}
	return nil
}

// The note making, given the reference's activations, makes the reference's notes exactly.
func TestNotesFromReference(t *testing.T) {
	ref := os.Getenv("HWNPU_BASIC_PITCH_REF")
	if ref == "" {
		t.Skip("HWNPU_BASIC_PITCH_REF is not set")
	}
	var meta struct {
		Frames int       `json:"frames"`
		Notes  []refNote `json:"notes"`
	}
	raw, err := os.ReadFile(ref + ".json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	out := readF32(t, ref+".output.bin")
	a := activations(out, meta.Frames)
	got := Notes(a, DefaultOptions)
	compareNotes(t, got, meta.Notes, 0, 0)
}

func activations(out []float32, frames int) Activations {
	a := Activations{Frames: frames}
	for f := range frames {
		row := out[f*outputs:]
		for j := range Pitches {
			a.Note = append(a.Note, float64(row[j]))
			a.Onset = append(a.Onset, float64(row[Pitches+j]))
		}
		for j := range ContourBins {
			a.Contour = append(a.Contour, float64(row[2*Pitches+j]))
		}
	}
	return a
}

// compareNotes checks the notes' pitches and times (within tol seconds) and, with bendTol < 0, not
// the bends.
func compareNotes(t *testing.T, got []Note, want []refNote, tol float64, ampTol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%d notes, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		g, w := got[i], want[i]
		if g.Pitch != w.Pitch || math.Abs(g.Start-w.Start) > tol+1e-9 || math.Abs(g.End-w.End) > tol+1e-9 || math.Abs(g.Amplitude-w.Amplitude) > ampTol+1e-6 {
			t.Errorf("note %d: %v %.3f-%.3f %.3f, want %v %.3f-%.3f %.3f", i, g.Pitch, g.Start, g.End, g.Amplitude, w.Pitch, w.Start, w.End, w.Amplitude)
		}
		if tol == 0 && len(g.Bends) != len(w.Bends) {
			t.Errorf("note %d: %d bends, want %d", i, len(g.Bends), len(w.Bends))
		}
	}
}

// With the real model (HWNPU_BASIC_PITCH = nmp.onnx) and a reference of ref_basic_pitch.py, the
// host's front end gives the graph's normalized log CQT, the program on the CPU interpreter in
// fp16 its activations, and the whole the reference's notes.
func TestMatchesONNX(t *testing.T) {
	path, ref := os.Getenv("HWNPU_BASIC_PITCH"), os.Getenv("HWNPU_BASIC_PITCH_REF")
	if path == "" || ref == "" {
		t.Skip("HWNPU_BASIC_PITCH and HWNPU_BASIC_PITCH_REF are not set")
	}
	g, err := onnx.Load(path)
	if err != nil {
		t.Fatal(err)
	}
	p, _, err := newProgram(g)
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(t.TempDir(), "bp.mlpackage")
	if _, err := llm.WritePart(dir, p, nil, llm.Options{Format: "fp16"}, nil); err != nil {
		t.Fatal(err)
	}
	tr, err := Open(context.Background(), nil, path, dir)
	if err != nil {
		t.Fatal(err)
	}
	defer tr.Close()
	audio := readF32(t, ref+".audio.bin")
	window := make([]float32, WindowSamples)
	copy(window[overlapSamples/2:], audio)
	spec, err := tr.Spectrum(window)
	if err != nil {
		t.Fatal(err)
	}
	if e := relErr(spec, readF32(t, ref+".cut.bin")); e > 5e-4 { // the graph computes it in float32, the host in float64
		t.Errorf("normalized log CQT: relative error %g", e)
	} else {
		t.Logf("normalized log CQT: relative error %.3g", e)
	}
	y, err := tr.window(context.Background(), spec)
	if err != nil {
		t.Fatal(err)
	}
	if e := relErr(y, readF32(t, ref+".window.bin")); e > 5e-3 {
		t.Errorf("activations: relative error %g", e)
	} else {
		t.Logf("activations: relative error %.3g", e)
	}
	a, err := tr.Run(context.Background(), audio)
	if err != nil {
		t.Fatal(err)
	}
	var meta struct {
		Frames int       `json:"frames"`
		Notes  []refNote `json:"notes"`
	}
	raw, err := os.ReadFile(ref + ".json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		t.Fatal(err)
	}
	if a.Frames != meta.Frames {
		t.Fatalf("%d frames, want %d", a.Frames, meta.Frames)
	}
	got := Notes(a, DefaultOptions)
	t.Logf("%d notes", len(got))
	// fp16 moves an activation by a little, a note's end by a frame at most
	compareNotes(t, got, meta.Notes, 2*float64(Hop)/SampleRate, 0.01)
	var buf bytes.Buffer
	if err := WriteMIDI(&buf, got); err != nil || !bytes.HasPrefix(buf.Bytes(), []byte("MThd")) {
		t.Errorf("MIDI: %v", err)
	}
}
