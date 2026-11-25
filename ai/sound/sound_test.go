package sound

import (
	"bytes"
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/bonsaiai/ai/flow"
	"github.com/shibukawa/bonsaiai/internal/airole"
	"github.com/shibukawa/bonsaiai/internal/chat"
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

// TestFadeOut checks the raised cosine over the last frames of interleaved audio, all of it
// when it is shorter, as bonsaiai-audio faded the music it wrote.
func TestFadeOut(t *testing.T) {
	x := []float32{1, 1, 1, 1, 1, 1, 1, 1} // 4 stereo frames
	fadeOut(x, 2, 2)
	want := []float32{1, 1, 1, 1, 1, 1, 0.5, 0.5}
	if !slices.Equal(x, want) {
		t.Errorf("the last 2 frames: %v, want %v", x, want)
	}
	y := []float32{1, 1, 1}
	fadeOut(y, 1, 10)
	if y[0] != 1 || math.Abs(float64(y[1])-0.75) > 1e-6 || math.Abs(float64(y[2])-0.25) > 1e-6 {
		t.Errorf("a fade longer than the sound: %v", y)
	}
	z := []float32{1, 1}
	fadeOut(z, 1, 0)
	if !slices.Equal(z, []float32{1, 1}) {
		t.Errorf("no fade: %v", z)
	}
}

// TestPiece checks that a piece gives its samples on, the last Fade of them held back and
// faded in the final chunk, on the clock of its samples.
func TestPiece(t *testing.T) {
	var got []flow.Chunk[[]float32]
	p := &piece{fade: 4, yield: func(c flow.Chunk[[]float32], err error) bool {
		got = append(got, c)
		return true
	}}
	ones := func(n int) []float32 {
		x := make([]float32, n)
		for i := range x {
			x[i] = 1
		}
		return x
	}
	p.give(ones(2))  // all held back
	p.give(ones(10)) // 8 given
	p.end()
	if len(got) != 2 {
		t.Fatalf("%d chunks, want 2", len(got))
	}
	if len(got[0].Data) != 8 || got[0].Final || got[0].Start != 0 || got[0].Len != flow.Samples(4, MusicRate) {
		t.Errorf("the first chunk: %d samples, final %v, at %v for %v", len(got[0].Data), got[0].Final, got[0].Start, got[0].Len)
	}
	last := got[1]
	if !last.Final || last.Start != flow.Samples(4, MusicRate) || len(last.Data) != 4 {
		t.Errorf("the final chunk: %d samples, final %v, at %v", len(last.Data), last.Final, last.Start)
	}
	if last.Data[0] != 1 || last.Data[2] != 0.5 {
		t.Errorf("the final chunk is not faded out: %v", last.Data)
	}
	// a reader that stops stops the piece
	stopped := &piece{yield: func(flow.Chunk[[]float32], error) bool { return false }}
	if stopped.give(ones(2)) || stopped.give(ones(2)) {
		t.Error("a piece goes on after its reader stopped")
	}
}

func TestMusicOptions(t *testing.T) {
	for _, c := range []struct {
		o      MusicOptions
		frames int
	}{
		{MusicOptions{}, 251}, // 10 s and the decoder's lookahead
		{MusicOptions{Duration: 4 * time.Second}, 101},
		{MusicOptions{Duration: 4 * time.Second, Notes: []MIDINote{{End: 9}}}, 101},
		{MusicOptions{Notes: []MIDINote{{End: 2.98}, {End: 3}}}, 75 + 12 + 1}, // the notes and half a second
	} {
		codes, frames, err := c.o.check()
		if err != nil {
			t.Fatal(err)
		}
		if frames != c.frames {
			t.Errorf("%+v: %d frames, want %d", c.o, frames, c.frames)
		}
		if len(codes) != StyleCodes || codes[0] != -1 {
			t.Errorf("no style: codes %v", codes)
		}
		if c.o.Temperature != 1.3 || c.o.TopK != 40 || *c.o.Guidance != DefaultGuidance {
			t.Errorf("the defaults: %v, %d, %+v", c.o.Temperature, c.o.TopK, *c.o.Guidance)
		}
	}
	for _, o := range []MusicOptions{
		{Codes: []int{1, 2, 3}},
		{Codes: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1024}},
		{Duration: -time.Second},
		{Drums: 7},
	} {
		if _, _, err := o.check(); err == nil {
			t.Errorf("%+v: no error", o)
		}
	}
	// a bad option is the stream's error, and its seed is told at once
	s, st := (&MusicGenerator{}).Generate(context.Background(), MusicOptions{Codes: []int{1}})
	if st.Seed == 0 {
		t.Error("no seed drawn")
	}
	if _, err := flow.Collect(s); err == nil {
		t.Error("a style of one code made music")
	}
	if _, st := (&MusicGenerator{}).Generate(context.Background(), MusicOptions{Seed: 42}); st.Seed != 42 {
		t.Errorf("seed %d, want 42", st.Seed)
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
	if !slices.Equal(c.mono(), []float32{0.5, 0, 0.5}) {
		t.Errorf("mono: %v", c.mono())
	}
	if c.Seconds() != 1.5 || (Clip{}).Seconds() != 0 {
		t.Errorf("%v s", c.Seconds())
	}
}

func TestEffectOptions(t *testing.T) {
	g := &EffectGenerator{}
	ctx := context.Background()
	for _, c := range []struct {
		desc string
		o    EffectOptions
	}{
		{" ", EffectOptions{}},
		{"rain", EffectOptions{Duration: 31 * time.Second}},
		{"rain", EffectOptions{Duration: 40 * time.Millisecond}}, // 0.0 s to a tenth
		{"rain", EffectOptions{Steps: -1}},
	} {
		if _, err := g.Generate(ctx, c.desc, c.o); err == nil {
			t.Errorf("%q %+v: no error", c.desc, c.o)
		}
	}
}

func TestCheckNPU(t *testing.T) {
	var log bytes.Buffer
	if err := checkNPU(&log, false, []string{"decoder: placement not known (busy)"}); err != nil {
		t.Errorf("a placement not known stops: %v", err)
	}
	err := checkNPU(&log, false, []string{"temporal: the CPU", "depth: the GPU"})
	var off *OffNPUError
	if !errors.As(err, &off) || len(off.Programs) != 2 {
		t.Errorf("programs off the NPU: %v", err)
	}
	if err := checkNPU(&log, true, []string{"temporal: the CPU"}); err != nil {
		t.Errorf("allowed: %v", err)
	}
	if !strings.Contains(log.String(), "NOT all on the NPU: temporal: the CPU") || !strings.Contains(log.String(), "going on: decoder") {
		t.Errorf("the log: %q", log.String())
	}
}

func TestNotesOptions(t *testing.T) {
	o := NotesOptions{}.internal()
	if o.OnsetThreshold != 0.5 || o.FrameThreshold != 0.3 || o.MinNoteMS != 127.7 || !o.Melodia {
		t.Errorf("the defaults: %+v", o)
	}
	o = NotesOptions{Onset: 0.4, Frame: 0.2, MinNoteMS: 50, MinHz: 100, MaxHz: 1000, NoMelodia: true}.internal()
	if o.OnsetThreshold != 0.4 || o.FrameThreshold != 0.2 || o.MinNoteMS != 50 || o.MinHz != 100 || o.MaxHz != 1000 || o.Melodia {
		t.Errorf("given: %+v", o)
	}
	if DefaultNotesOptions.internal() != (NotesOptions{}).internal() {
		t.Error("DefaultNotesOptions are not the zero value's")
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

// TestConvertedRoute checks where a converted model runs by its definition: bonsaiai's executor
// for settings.cpu bonsaiai (all the cores unless threads), the CPU for device cpu, and threads.
func TestConvertedRoute(t *testing.T) {
	t.Setenv("BONSAIAI_DEVICE", "")
	t.Setenv("BONSAIAI_ONNXRUNTIME", "none")
	n := airole.NPU{Vendor: "amd"}
	r, err := convertedRoute(chat.Model{Settings: map[string]string{"cpu": "bonsaiai"}}, "swift-f0", n)
	if err != nil || r.runtime != "bonsaiai" || r.threads != 0 || r.Where() != "on the CPU interpreter" || r.line != "" {
		t.Errorf("cpu bonsaiai: %+v, %v", r, err)
	}
	r, _ = convertedRoute(chat.Model{Settings: map[string]string{"threads": "3"}}, "swift-f0", n)
	if r.device != "cpu" || r.runtime != "bonsaiai" || r.threads != 3 || !strings.Contains(r.line, "no converted form") {
		t.Errorf("on AMD's NPU: %+v", r)
	}
	r, _ = convertedRoute(chat.Model{Device: map[string]string{"amd": "cpu"}}, "swift-f0", n)
	if r.device != "cpu" || r.threads != 1 || !strings.Contains(r.line, "(--device;") {
		t.Errorf("device cpu: %+v", r)
	}
	r, _ = convertedRoute(chat.Model{Device: map[string]string{"apple": "npu"}}, "swift-f0", airole.NPU{Vendor: "apple"})
	if r.device != "npu" || r.runtime != "npu" || r.Where() != "" {
		t.Errorf("Apple's NPU: %+v", r)
	}
	if _, err := convertedRoute(chat.Model{Settings: map[string]string{"threads": "many"}}, "swift-f0", n); err == nil {
		t.Error("threads many: no error")
	}
}
