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
	"github.com/shibukawa/hwnpu/internal/pitch"
)

// The pitch's models run as the converted models do (converted.go): the front end (the three
// spectrograms) and the decoding of the logits on the host, the CNN on the NPU or the CPU.
func preparePitch(e airole.PrepareEnv) (airole.Files, error) {
	return convertedPrepare(e, func(backend string) (string, string, error) {
		return pitch.Prepare(e.Ctx, pitch.Setup{Dir: e.Dir, Repo: e.Model.Source(), Backend: backend, Log: e.Log})
	})
}

func loadPitch(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
	r, err := convertedRoute(e.Def, e.Model, e.NPU)
	if err != nil {
		return nil, err
	}
	d, dev, err := convertedLoad(ctx, f, e, r, convertedOpener[*pitch.Detector]{
		cpu: pitch.OpenCPU,
		npu: pitch.Open,
	})
	if err != nil {
		return nil, err
	}
	return bridge.Wrap["pitch"](&pitchDetector{d: d, dev: dev, r: r, model: e.Model}), nil
}

// pitchDetector is a pitch model of the engine (sound.PitchDetector's). It runs one request at
// a time; others wait.
type pitchDetector struct {
	mu    sync.Mutex
	d     *pitch.Detector
	dev   *hwnpu.Device
	r     route
	model string
}

func (d *pitchDetector) Model() string  { return d.model }
func (d *pitchDetector) Device() string { return d.r.device }

func (d *pitchDetector) Where() string {
	if w := d.r.Where(); w != "" {
		return w
	}
	return onNPUWhere(d.d.OnNPU())
}

func (d *pitchDetector) Detect(ctx context.Context, pcm []float32, o sound.PitchOptions) (sound.PitchTrack, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	t0 := time.Now()
	r, err := d.d.Pitch(ctx, pcm, o.FMin, o.FMax)
	if err != nil {
		return sound.PitchTrack{}, err
	}
	t := sound.PitchTrack{Seconds: float64(len(pcm)) / pitch.SampleRate, Took: time.Since(t0)}
	period := float64(pitch.Hop) / pitch.SampleRate
	for i := range r.PitchHz {
		t.Frames = append(t.Frames, sound.PitchFrame{Time: float64(i) * period, Hz: r.PitchHz[i], Confidence: r.Confidence[i], LoudnessDB: r.LoudnessDB[i]})
	}
	return t, nil
}

func (d *pitchDetector) Close() error {
	d.mu.Lock()
	defer d.mu.Unlock()
	err := d.d.Close()
	if d.dev != nil {
		d.dev.Close()
		d.dev = nil
	}
	return err
}
