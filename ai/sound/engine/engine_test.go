package engine

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"math"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/sound"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/music"
	"github.com/shibukawa/hwnpu/internal/notes"
	"github.com/shibukawa/hwnpu/internal/pitch"
	"github.com/shibukawa/hwnpu/internal/sfx"
	"github.com/shibukawa/hwnpu/internal/soundwire"
)

// TestRoles checks that the engine gives every role of ai/sound its Prepare and Load.
func TestRoles(t *testing.T) {
	for _, name := range []string{"music", "sfx", "pitch", "notes"} {
		r := airole.Lookup(name)
		if r == nil || r.Prepare == nil || r.Load == nil {
			t.Errorf("the role %s: %+v", name, r)
		}
	}
}

// TestWire checks that what ai/sound knows without the engine (internal/soundwire) is the
// engine's: the models' constants and defaults, and the MIDI and WAV files read and written as
// internal/notes and internal/llm/lfm3audio do.
func TestWire(t *testing.T) {
	for _, c := range []struct {
		name      string
		got, want float64
	}{
		{"PitchRate", soundwire.PitchRate, pitch.SampleRate},
		{"PitchHop", soundwire.PitchHop, pitch.Hop},
		{"NotesRate", soundwire.NotesRate, notes.SampleRate},
		{"EffectRate", soundwire.EffectRate, sfx.SampleRate},
		{"EffectMaxSeconds", soundwire.EffectMaxSeconds, sfx.MaxSeconds},
		{"EffectPromptTokens", soundwire.EffectPromptTokens, sfx.PromptRows},
		{"MusicRate", soundwire.MusicRate, music.SampleRate},
		{"MusicFrameRate", soundwire.MusicFrameRate, music.FrameRate},
		{"StyleCodes", soundwire.StyleCodes, music.StyleCodes},
		{"NotesOnset", soundwire.NotesOnset, notes.DefaultOptions.OnsetThreshold},
		{"NotesFrame", soundwire.NotesFrame, notes.DefaultOptions.FrameThreshold},
		{"NotesMinNoteMS", soundwire.NotesMinNoteMS, notes.DefaultOptions.MinNoteMS},
	} {
		if c.got != c.want {
			t.Errorf("%s: %v, the engine's %v", c.name, c.got, c.want)
		}
	}
	if music.Guidance(sound.DefaultGuidance) != music.DefaultGuidance {
		t.Errorf("DefaultGuidance %+v, the engine's %+v", sound.DefaultGuidance, music.DefaultGuidance)
	}
	if notesOptions(sound.DefaultNotesOptions) != notes.DefaultOptions {
		t.Error("DefaultNotesOptions are not internal/notes's")
	}

	ns := []notes.Note{{Start: 0, End: 0.5, Pitch: 60, Amplitude: 0.5, Bends: []int{0, 1, -2}}, {Start: 0.25, End: 1, Pitch: 64, Amplitude: 1},
		{Start: 1.5, End: 2, Pitch: 67, Amplitude: 0.3, Bends: []int{3, -3}}}
	var want, got bytes.Buffer
	if err := notes.WriteMIDI(&want, ns); err != nil {
		t.Fatal(err)
	}
	wn := make([]soundwire.Note, len(ns))
	for i, n := range ns {
		wn[i] = soundwire.Note(n)
	}
	if err := soundwire.WriteMIDI(&got, wn); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got.Bytes(), want.Bytes()) {
		t.Error("WriteMIDI writes another file than internal/notes's")
	}
	for _, data := range [][]byte{want.Bytes(), []byte("MThd"), want.Bytes()[:30]} {
		a, aerr := notes.ReadMIDI(data)
		b, berr := soundwire.ReadMIDI(data)
		if (aerr == nil) != (berr == nil) || len(a) != len(b) {
			t.Errorf("ReadMIDI: %v, %v; internal/notes's %v, %v", b, berr, a, aerr)
			continue
		}
		for i := range a {
			if soundwire.MIDINote(a[i]) != b[i] {
				t.Errorf("ReadMIDI's note %d: %+v, internal/notes's %+v", i, b[i], a[i])
			}
		}
	}

	for _, w := range []struct{ format, bits int }{{1, 8}, {1, 16}, {1, 24}, {1, 32}, {3, 32}, {3, 64}} {
		data := wav(w.format, w.bits, 2, 8000, 37)
		a, ar, aerr := lfm3audio.DecodeWAV(data)
		b, br, berr := soundwire.DecodeWAV(data)
		if (aerr == nil) != (berr == nil) || ar != br || !reflect.DeepEqual(a, b) {
			t.Errorf("DecodeWAV of format %d, %d bits: %d Hz, %v; lfm3audio's %d Hz, %v", w.format, w.bits, br, berr, ar, aerr)
		}
	}
	if _, _, err := soundwire.DecodeWAV([]byte("RIFF")); err == nil {
		t.Error("DecodeWAV of no WAV: no error")
	}
}

// wav is a WAV file of frames frames of a ramp.
func wav(format, bits, channels, rate, frames int) []byte {
	var pcm bytes.Buffer
	for i := range frames * channels {
		v := float64(i%200)/100 - 1
		switch {
		case format == 3:
			binary.Write(&pcm, binary.LittleEndian, math.Float32bits(float32(v)))
		case bits == 8:
			pcm.WriteByte(byte(128 + v*127))
		case bits == 16:
			binary.Write(&pcm, binary.LittleEndian, int16(v*32767))
		case bits == 24:
			x := int32(v * 8388607)
			pcm.Write([]byte{byte(x), byte(x >> 8), byte(x >> 16)})
		default:
			binary.Write(&pcm, binary.LittleEndian, int32(v*2147483647))
		}
	}
	var b bytes.Buffer
	b.WriteString("RIFF")
	binary.Write(&b, binary.LittleEndian, uint32(36+pcm.Len()))
	b.WriteString("WAVEfmt ")
	binary.Write(&b, binary.LittleEndian, []uint32{16})
	binary.Write(&b, binary.LittleEndian, []uint16{uint16(format), uint16(channels)})
	binary.Write(&b, binary.LittleEndian, []uint32{uint32(rate), uint32(rate * channels * bits / 8)})
	binary.Write(&b, binary.LittleEndian, []uint16{uint16(channels * bits / 8), uint16(bits)})
	b.WriteString("data")
	binary.Write(&b, binary.LittleEndian, uint32(pcm.Len()))
	b.Write(pcm.Bytes())
	return b.Bytes()
}

// TestFadeOut checks the raised cosine over the last frames of interleaved audio, all of it
// when it is shorter, as hwnpu-audio faded the music it wrote.
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
	if len(got[0].Data) != 8 || got[0].Final || got[0].Start != 0 || got[0].Len != flow.Samples(4, sound.MusicRate) {
		t.Errorf("the first chunk: %d samples, final %v, at %v for %v", len(got[0].Data), got[0].Final, got[0].Start, got[0].Len)
	}
	last := got[1]
	if !last.Final || last.Start != flow.Samples(4, sound.MusicRate) || len(last.Data) != 4 {
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
		o      sound.MusicOptions
		frames int
	}{
		{sound.MusicOptions{}, 251}, // 10 s and the decoder's lookahead
		{sound.MusicOptions{Duration: 4 * time.Second}, 101},
		{sound.MusicOptions{Duration: 4 * time.Second, Notes: []sound.MIDINote{{End: 9}}}, 101},
		{sound.MusicOptions{Notes: []sound.MIDINote{{End: 2.98}, {End: 3}}}, 75 + 12 + 1}, // the notes and half a second
	} {
		codes, frames, err := checkMusic(&c.o)
		if err != nil {
			t.Fatal(err)
		}
		if frames != c.frames {
			t.Errorf("%+v: %d frames, want %d", c.o, frames, c.frames)
		}
		if len(codes) != sound.StyleCodes || codes[0] != -1 {
			t.Errorf("no style: codes %v", codes)
		}
		if c.o.Temperature != 1.3 || c.o.TopK != 40 || *c.o.Guidance != sound.DefaultGuidance {
			t.Errorf("the defaults: %v, %d, %+v", c.o.Temperature, c.o.TopK, *c.o.Guidance)
		}
	}
	for _, o := range []sound.MusicOptions{
		{Codes: []int{1, 2, 3}},
		{Codes: []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1024}},
		{Duration: -time.Second},
		{Drums: 7},
	} {
		if _, _, err := checkMusic(&o); err == nil {
			t.Errorf("%+v: no error", o)
		}
	}
	// a bad option is the stream's error, and its seed is told at once
	s, st := (&musicGenerator{}).Generate(context.Background(), sound.MusicOptions{Codes: []int{1}})
	if st.Seed == 0 {
		t.Error("no seed drawn")
	}
	if _, err := flow.Collect(s); err == nil {
		t.Error("a style of one code made music")
	}
	if _, st := (&musicGenerator{}).Generate(context.Background(), sound.MusicOptions{Seed: 42}); st.Seed != 42 {
		t.Errorf("seed %d, want 42", st.Seed)
	}
}

func TestMono(t *testing.T) {
	c := sound.Clip{Channels: [][]float32{{1, 0, 0.5}, {0, 0, 0.5}}, Rate: 2}
	if !slices.Equal(mono(c), []float32{0.5, 0, 0.5}) {
		t.Errorf("mono: %v", mono(c))
	}
}

func TestEffectOptions(t *testing.T) {
	g := &effectGenerator{}
	ctx := context.Background()
	for _, c := range []struct {
		desc string
		o    sound.EffectOptions
	}{
		{" ", sound.EffectOptions{}},
		{"rain", sound.EffectOptions{Duration: 31 * time.Second}},
		{"rain", sound.EffectOptions{Duration: 40 * time.Millisecond}}, // 0.0 s to a tenth
		{"rain", sound.EffectOptions{Steps: -1}},
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
	var off *sound.OffNPUError
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
	o := notesOptions(sound.NotesOptions{})
	if o.OnsetThreshold != 0.5 || o.FrameThreshold != 0.3 || o.MinNoteMS != 127.7 || !o.Melodia {
		t.Errorf("the defaults: %+v", o)
	}
	o = notesOptions(sound.NotesOptions{Onset: 0.4, Frame: 0.2, MinNoteMS: 50, MinHz: 100, MaxHz: 1000, NoMelodia: true})
	if o.OnsetThreshold != 0.4 || o.FrameThreshold != 0.2 || o.MinNoteMS != 50 || o.MinHz != 100 || o.MaxHz != 1000 || o.Melodia {
		t.Errorf("given: %+v", o)
	}
	if notesOptions(sound.DefaultNotesOptions) != notesOptions(sound.NotesOptions{}) {
		t.Error("DefaultNotesOptions are not the zero value's")
	}
}

// TestConvertedRoute checks where a converted model runs by its definition: hwnpu's executor
// for settings.cpu hwnpu (all the cores unless threads), the CPU for device cpu, and threads.
func TestConvertedRoute(t *testing.T) {
	t.Setenv("HWNPU_DEVICE", "")
	t.Setenv("HWNPU_ONNXRUNTIME", "none")
	n := airole.NPU{Vendor: "amd"}
	r, err := convertedRoute(chat.Model{Settings: map[string]string{"cpu": "hwnpu"}}, "swift-f0", n)
	if err != nil || r.runtime != "hwnpu" || r.threads != 0 || r.Where() != "on the CPU interpreter" || r.line != "" {
		t.Errorf("cpu hwnpu: %+v, %v", r, err)
	}
	r, _ = convertedRoute(chat.Model{Settings: map[string]string{"threads": "3"}}, "swift-f0", n)
	if r.device != "cpu" || r.runtime != "hwnpu" || r.threads != 3 || !strings.Contains(r.line, "no converted form") {
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
