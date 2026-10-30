// Package audio turns audio clips into LFM3-Audio / LFM3-Omni audio embeddings for the text
// model's input, with audio encoder programs converted by hwnpu-convert -arch lfm3-audio-encoder,
// one per clip length (the Neural Engine runs fixed shapes). A clip runs on the shortest program
// that holds it, its frames followed by zeros and masked, which gives the same embeddings as
// encoding the clip alone.
//
// In the prompt, LFM3-Audio places the embeddings bare and LFM3-Omni between <|vision_start|>
// and <|vision_end|>, as llama.cpp's mtmd does.
package audio

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// Encoder holds the loaded audio encoder programs.
type Encoder struct {
	cfg   *lfm3audio.Config
	progs []sized // by frames
}

type sized struct {
	runner.Program
	frames int
}

// Load reads the checkpoint's config.json in dir and loads the audio encoder packages on dev
// (nil: the CPU interpreter).
func Load(ctx context.Context, dev *hwnpu.Device, dir string, packages []string) (*Encoder, error) {
	_, m, err := llm.LoadConfig(filepath.Join(dir, "config.json"), lfm3audio.Arch.Name)
	if err != nil {
		return nil, err
	}
	e := &Encoder{cfg: m.(*lfm3audio.Config)}
	for _, path := range packages {
		p, err := runner.Open(ctx, dev, path)
		if err != nil {
			e.Close()
			return nil, err
		}
		meta := p.Metadata()
		frames, err := lfm3audio.Frames(meta[llm.MetaParam+"frames"])
		if err == nil && meta[llm.MetaArch] != lfm3audio.Arch.Name {
			err = fmt.Errorf("it is %q, not %s", meta[llm.MetaArch], lfm3audio.Arch.Name)
		}
		if err == nil {
			x, y := p.Inputs()["x"], p.Output()
			if len(x) != 3 || x[1] != frames || x[2] != e.cfg.MelBins || len(y) != 3 || y[1] != lfm3audio.Tokens(frames) {
				err = fmt.Errorf("input %v and output %v do not fit %d frames", x, y, frames)
			}
		}
		if err != nil {
			p.Close()
			e.Close()
			return nil, fmt.Errorf("audio: %s: %w", path, err)
		}
		e.progs = append(e.progs, sized{p, frames})
	}
	if len(e.progs) == 0 {
		return nil, errors.New("audio: no package")
	}
	slices.SortFunc(e.progs, func(a, b sized) int { return a.frames - b.frames })
	return e, nil
}

// Close releases the programs.
func (e *Encoder) Close() error {
	for _, p := range e.progs {
		p.Close()
	}
	e.progs = nil
	return nil
}

// Omni reports whether the encoder is LFM3-Omni's, whose audio the prompt wraps like an image.
func (e *Encoder) Omni() bool { return e.cfg.Omni }

// MaxSeconds is the longest clip the programs take.
func (e *Encoder) MaxSeconds() float64 {
	return float64(e.progs[len(e.progs)-1].frames) * 0.01
}

// OnNPU reports whether every program runs on the NPU.
func (e *Encoder) OnNPU() bool {
	for _, p := range e.progs {
		if !p.OnNPU() {
			return false
		}
	}
	return true
}

// ErrTooLong is returned for a clip longer than every program.
var ErrTooLong = errors.New("audio: the clip is longer than the audio encoder takes")

// Encode returns a clip's embeddings (samples at lfm3audio.SampleRate): rows of the text model's
// width, tokens of them.
func (e *Encoder) Encode(ctx context.Context, samples []float32) (rows []float32, tokens int, err error) {
	mel, frames := lfm3audio.Features(samples, e.cfg.MelBins)
	i := slices.IndexFunc(e.progs, func(p sized) bool { return p.frames >= frames })
	if i < 0 {
		return nil, 0, fmt.Errorf("%w: %.1f s, at most %.1f s", ErrTooLong, float64(frames)*0.01, e.MaxSeconds())
	}
	p := e.progs[i]
	x := make([]float32, p.frames*e.cfg.MelBins)
	copy(x, mel)
	mask := make([]float32, p.frames)
	for j := range frames {
		mask[j] = 1
	}
	y, err := p.Run(ctx, map[string][]float32{"x": x, "mask": mask})
	if err != nil {
		return nil, 0, err
	}
	n := lfm3audio.Tokens(frames)
	return y[:n*e.cfg.TextHidden], n, nil
}

// Decode reads a clip given as base64 WAV data (OpenAI's input_audio data) or a data URL
// (data:audio/...;base64,...).
func Decode(data string) ([]float32, error) {
	if rest, ok := strings.CutPrefix(data, "data:"); ok {
		meta, body, ok := strings.Cut(rest, ",")
		if !ok || !strings.HasSuffix(meta, ";base64") {
			return nil, errors.New("an audio data URL must be base64")
		}
		data = body
	}
	raw, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, fmt.Errorf("audio data: %w", err)
	}
	return lfm3audio.ReadWAV(raw)
}
