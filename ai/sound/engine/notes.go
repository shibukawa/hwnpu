package engine

import (
	"context"
	"io"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai/sound"
	"github.com/shibukawa/hwnpu/ai/sound/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/notes"
)

// The notes' models run as the converted models do (converted.go): the constant-Q transform
// and the making of notes on the host, the network on the NPU or the CPU.
func prepareNotes(e airole.PrepareEnv) (airole.Files, error) {
	return convertedPrepare(e, func(backend string) (string, string, error) {
		return notes.Prepare(e.Ctx, notes.Setup{Dir: e.Dir, Repo: e.Model.Source(), SHA256: e.Model.SHA256[notes.ModelFile], Backend: backend, Log: e.Log})
	})
}

func loadNotes(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
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
	return bridge.Wrap["notes"](&noteTranscriber{t: t, r: r, model: e.Model}), nil
}

// noteTranscriber is a notes model of the engine (sound.NoteTranscriber's). It runs one request
// at a time; others wait.
type noteTranscriber struct {
	mu    sync.Mutex
	t     *notes.Transcriber
	r     route
	model string
}

func (t *noteTranscriber) Model() string  { return t.model }
func (t *noteTranscriber) Device() string { return t.r.device }

func (t *noteTranscriber) Where() string {
	if w := t.r.Where(); w != "" {
		return w
	}
	return onNPUWhere(t.t.OnNPU())
}

// notesOptions are o as internal/notes takes them.
func notesOptions(o sound.NotesOptions) notes.Options {
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

func (t *noteTranscriber) Transcribe(ctx context.Context, pcm []float32, o sound.NotesOptions) (sound.NoteList, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t0 := time.Now()
	a, err := t.t.Run(ctx, pcm)
	if err != nil {
		return sound.NoteList{}, err
	}
	ns := notes.Notes(a, notesOptions(o))
	out := sound.NoteList{Seconds: float64(len(pcm)) / notes.SampleRate, Took: time.Since(t0)}
	if ns != nil {
		out.Notes = make([]sound.Note, 0, len(ns))
	}
	for _, n := range ns {
		out.Notes = append(out.Notes, sound.Note(n))
	}
	return out, nil
}

func (t *noteTranscriber) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.t.Close()
}
