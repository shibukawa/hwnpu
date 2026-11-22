package sound

import (
	"context"
	"io"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/notes"
)

// NotesRate is the samples a second a NoteTranscriber reads (mono).
const NotesRate = notes.SampleRate

// Notes is the use of a model that writes down the notes of music (Basic Pitch) for an app: a
// model, a family or a use of the definitions ("notes", "basic-pitch").
func Notes(name string) ai.Use[*NoteTranscriber] {
	return ai.Define[*NoteTranscriber](notesRole, name)
}

// notesRole's models run as the converted models do (converted.go): the constant-Q transform
// and the making of notes on the host, the network on the NPU or the CPU.
var notesRole = &airole.Role{
	Name:  "notes",
	Kinds: []chat.Kind{chat.KindNotes},
	Prepare: func(e airole.PrepareEnv) (airole.Files, error) {
		return convertedPrepare(e, func(backend string) (string, string, error) {
			return notes.Prepare(e.Ctx, notes.Setup{Dir: e.Dir, Repo: e.Model.Source(), SHA256: e.Model.SHA256[notes.ModelFile], Backend: backend, Log: e.Log})
		})
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		r, err := convertedRoute(e.Def, e.Model, e.NPU)
		if err != nil {
			return nil, err
		}
		c, dev, err := convertedLoad(ctx, f, e, r, convertedOpener{
			cpu: func(model string, threads int) (io.Closer, error) { return notes.OpenCPU(model, threads) },
			npu: func(ctx context.Context, dev *hwnpu.Device, model, program string) (io.Closer, error) {
				return notes.Open(ctx, dev, model, program)
			}})
		if err != nil {
			return nil, err
		}
		t := c.(*notes.Transcriber)
		if dev != nil {
			t.OnClose = func() { dev.Close() }
		}
		return &NoteTranscriber{t: t, r: r, model: e.Model}, nil
	},
}

// NoteTranscriber writes down the notes of music (Basic Pitch): any instrument, several notes
// at once, with their pitch bends. It runs one request at a time; others wait.
type NoteTranscriber struct {
	mu    sync.Mutex
	t     *notes.Transcriber
	r     route
	model string
}

// Model is the name of the model in the definitions.
func (t *NoteTranscriber) Model() string { return t.model }

// Device is where the model runs: npu or cpu.
func (t *NoteTranscriber) Device() string { return t.r.device }

// Where says where the model runs, for a log: "all on the NPU", "on the CPU (ONNX Runtime, 1
// threads)", "on the CPU interpreter", ...
func (t *NoteTranscriber) Where() string {
	if w := t.r.Where(); w != "" {
		return w
	}
	return onNPUWhere(t.t.OnNPU())
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
var DefaultNotesOptions = NotesOptions{Onset: notes.DefaultOptions.OnsetThreshold, Frame: notes.DefaultOptions.FrameThreshold,
	MinNoteMS: notes.DefaultOptions.MinNoteMS}

func (o NotesOptions) internal() notes.Options {
	n := notes.DefaultOptions
	if o.Onset != 0 {
		n.OnsetThreshold = o.Onset
	}
	if o.Frame != 0 {
		n.FrameThreshold = o.Frame
	}
	if o.MinNoteMS != 0 {
		n.MinNoteMS = o.MinNoteMS
	}
	n.MinHz, n.MaxHz, n.Melodia = o.MinHz, o.MaxHz, !o.NoMelodia
	return n
}

// NoteList is the notes of some music.
type NoteList struct {
	Notes []Note
	// Seconds is the audio's length and Took how long writing it down took.
	Seconds float64
	Took    time.Duration
}

// Transcribe writes down the notes of music (mono, at NotesRate, in [-1, 1]).
func (t *NoteTranscriber) Transcribe(ctx context.Context, pcm []float32, o NotesOptions) (NoteList, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t0 := time.Now()
	a, err := t.t.Run(ctx, pcm)
	if err != nil {
		return NoteList{}, err
	}
	ns := notes.Notes(a, o.internal())
	out := NoteList{Seconds: float64(len(pcm)) / notes.SampleRate, Took: time.Since(t0)}
	if ns != nil {
		out.Notes = make([]Note, 0, len(ns))
	}
	for _, n := range ns {
		out.Notes = append(out.Notes, Note(n))
	}
	return out, nil
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (t *NoteTranscriber) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.t.Close()
}
