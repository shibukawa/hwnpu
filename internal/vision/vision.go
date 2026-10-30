// Package vision turns images into LFM3-VL image embeddings for the text model's input, with
// vision encoder programs converted by hwnpu-convert -arch lfm3-vision, one per image size (the
// Neural Engine runs fixed shapes). An image is resized to the program whose aspect ratio is
// closest to the size the checkpoint's image processor would give it (smart_resize), so its
// embeddings match the reference exactly when that size has a program, and approximately
// (stretched to a nearby grid) otherwise.
//
// In the prompt an image is <|vision_start|>, its embeddings, <|vision_end|>, as llama.cpp's mtmd
// writes LFM3-VL images.
package vision

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	"math"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/llm/runner"
)

// Start and End are the tokens around an image's embeddings.
const (
	Start = "<|vision_start|>"
	End   = "<|vision_end|>"
)

// Encoder holds the loaded vision programs.
type Encoder struct {
	cfg   *lfm3vl.Config
	proc  *lfm3vl.Processor
	progs []sized
}

type sized struct {
	runner.Program
	w, h int
}

// Load reads the checkpoint's config.json and image processor settings in dir and loads the
// vision encoder packages on dev (nil: the CPU interpreter).
func Load(ctx context.Context, dev *hwnpu.Device, dir string, packages []string) (*Encoder, error) {
	_, m, err := llm.LoadConfig(filepath.Join(dir, "config.json"), lfm3vl.Arch.Name)
	if err != nil {
		return nil, err
	}
	e := &Encoder{cfg: m.(*lfm3vl.Config)}
	if e.proc, err = lfm3vl.LoadProcessor(dir, e.cfg); err != nil {
		return nil, err
	}
	for _, path := range packages {
		p, err := runner.Open(ctx, dev, path)
		if err != nil {
			e.Close()
			return nil, err
		}
		meta := p.Metadata()
		w, h, err := e.cfg.ImageSize(meta[llm.MetaParam+"image"])
		if err == nil && meta[llm.MetaArch] != lfm3vl.Arch.Name {
			err = fmt.Errorf("it is %q, not %s", meta[llm.MetaArch], lfm3vl.Arch.Name)
		}
		if err == nil {
			x, y := p.Inputs()["x"], p.Output()
			n := (w / e.cfg.Vision.PatchSize) * (h / e.cfg.Vision.PatchSize)
			if len(x) != 3 || x[1] != n || x[2] != e.cfg.PatchDim() || len(y) != 3 || y[1] != e.cfg.Tokens(w, h) {
				err = fmt.Errorf("input %v and output %v do not fit an image of %dx%d", x, y, w, h)
			}
		}
		if err != nil {
			p.Close()
			e.Close()
			return nil, fmt.Errorf("vision: %s: %w", path, err)
		}
		e.progs = append(e.progs, sized{p, w, h})
	}
	if len(e.progs) == 0 {
		return nil, errors.New("vision: no package")
	}
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

// Sizes are the image sizes of the loaded programs.
func (e *Encoder) Sizes() [][2]int {
	var out [][2]int
	for _, p := range e.progs {
		out = append(out, [2]int{p.w, p.h})
	}
	return out
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

// pick returns the program for an image of w x h pixels: the closest aspect ratio to the size
// the processor would resize it to, then the closest area.
func (e *Encoder) pick(w, h int) sized {
	tw, th := e.proc.SmartResize(w, h)
	best, bestA, bestS := 0, math.Inf(1), math.Inf(1)
	for i, p := range e.progs {
		a := math.Abs(math.Log(float64(p.w)/float64(p.h)) - math.Log(float64(tw)/float64(th)))
		s := math.Abs(math.Log(float64(p.w*p.h)) - math.Log(float64(tw*th)))
		if a < bestA-1e-9 || a < bestA+1e-9 && s < bestS {
			best, bestA, bestS = i, a, s
		}
	}
	return e.progs[best]
}

// Encode returns an image's embeddings: rows of the text model's width, Tokens of them.
func (e *Encoder) Encode(ctx context.Context, img image.Image) (rows []float32, tokens int, err error) {
	b := img.Bounds()
	p := e.pick(b.Dx(), b.Dy())
	y, err := p.Run(ctx, map[string][]float32{"x": e.proc.Patches(img, p.w, p.h)})
	if err != nil {
		return nil, 0, err
	}
	return y, e.cfg.Tokens(p.w, p.h), nil
}

// DecodeDataURL decodes an image given as a data URL (data:image/...;base64,...). PNG, JPEG and
// GIF are read.
func DecodeDataURL(url string) (image.Image, error) {
	rest, ok := strings.CutPrefix(url, "data:")
	if !ok {
		return nil, errors.New("images must be data URLs (data:image/...;base64,...)")
	}
	meta, data, ok := strings.Cut(rest, ",")
	if !ok || !strings.HasSuffix(meta, ";base64") {
		return nil, errors.New("an image data URL must be base64")
	}
	raw, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, fmt.Errorf("image data: %w", err)
	}
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("image: %w (PNG, JPEG and GIF are supported)", err)
	}
	return img, nil
}
