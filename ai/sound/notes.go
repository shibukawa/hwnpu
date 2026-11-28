package sound

import (
	"context"
	"io"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/sound/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
	"github.com/shibukawa/hwnpu/internal/soundwire"
)

// NotesRate is the samples a second a NoteTranscriber reads (mono).
const NotesRate = soundwire.NotesRate

// Notes is the use of a model that writes down the notes of music (Basic Pitch) for an app: a
// model, a family or a use of the definitions ("notes", "basic-pitch").
func Notes(name string) ai.Use[*NoteTranscriber] {
	return ai.Define[*NoteTranscriber](notesRole, name)
}

// notesRole's Prepare and Load are the engine's (ai/sound/engine): the constant-Q transform and
// the making of notes on the host, the network on the NPU or the CPU.
var notesRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &NoteTranscriber{p: airpc.RemoteProxy[noteTranscriberImpl](h)} },
	Name:   "notes",
	Kinds:  []catalog.Kind{catalog.KindNotes},
}

// NoteTranscriber writes down the notes of music (Basic Pitch): any instrument, several notes
// at once, with their pitch bends. It runs one request at a time; others wait.
type NoteTranscriber struct {
	p airpc.Proxy[noteTranscriberImpl]
}

// noteTranscriberImpl is a notes model of the engine in this process.
type noteTranscriberImpl interface {
	Model() string
	Device() string
	Where() string
	Transcribe(ctx context.Context, pcm []float32, o NotesOptions) (NoteList, error)
	Close() error
}

// Model is the name of the model in the definitions.
func (t *NoteTranscriber) Model() string {
	return airpc.ProxyValue(t.p, "Model", noteTranscriberImpl.Model)
}

// Device is where the model runs: npu or cpu.
func (t *NoteTranscriber) Device() string {
	return airpc.ProxyValue(t.p, "Device", noteTranscriberImpl.Device)
}

// Where says where the model runs, for a log: "all on the NPU", "on the CPU (ONNX Runtime, 1
// threads)", "on the CPU interpreter", ...
func (t *NoteTranscriber) Where() string {
	return airpc.ProxyValue(t.p, "Where", noteTranscriberImpl.Where)
}

// NotesOptions are how the notes are made from the network's activations (basic-pitch's
// predict; the zero value is its defaults).
type NotesOptions struct {
	// Onset is the least onset activation of a note's start (0: 0.5); Frame the least note
	// activation that keeps a note on (0: 0.3).
	Onset, Frame float64
	// MinNoteMS is the shortest note, in milliseconds (0: 127.7).
	MinNoteMS float64
	// MinHz and MaxHz, when not 0, drop the notes outside them.
	MinHz, MaxHz float64
	// NoMelodia leaves out the notes taken from the strong activations left without an onset
	// (basic-pitch's melodia trick).
	NoMelodia bool
}

// DefaultNotesOptions are basic-pitch's.
var DefaultNotesOptions = NotesOptions{Onset: soundwire.NotesOnset, Frame: soundwire.NotesFrame,
	MinNoteMS: soundwire.NotesMinNoteMS}

// NoteList is the notes of some music.
type NoteList struct {
	Notes []Note
	// Seconds is the audio's length and Took how long writing it down took.
	Seconds float64
	Took    time.Duration
}

// Transcribe writes down the notes of music (mono, at NotesRate, in [-1, 1]).
func (t *NoteTranscriber) Transcribe(ctx context.Context, pcm []float32, o NotesOptions) (NoteList, error) {
	return airpc.ProxyCall1(t.p, ctx, "Transcribe", func(l noteTranscriberImpl) (NoteList, error) {
		return l.Transcribe(ctx, pcm, o)
	}, pcm, o)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (t *NoteTranscriber) Close() error { return t.p.Close() }

func init() {
	airole.Register(notesRole)
	bridge.Roles[notesRole.Name] = notesRole
	bridge.Wrap[notesRole.Name] = func(l any) io.Closer { return &NoteTranscriber{p: airpc.LocalProxy(l.(noteTranscriberImpl))} }
}
