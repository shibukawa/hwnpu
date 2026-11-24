package main

import (
	"context"
	"encoding/json"
	"fmt"
	"image"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/imagegen"
	"github.com/shibukawa/hwnpu/samples/internal/layering"
)

// source is the picture and its nearness as the scene was first made from them: what is parted
// into layers.
type source struct {
	img  *image.RGBA
	near []float32
}

// stage is what the window's main area shows while the picture is made deep.
type stage int

const (
	stageDepth   stage = iota // the picture, while its depth is estimated
	stageNear                 // its depth
	stageCut                  // a layer lifted out of the picture
	stageDrawing              // the model drawing what the layer hid (the layer still shown)
	stageDrawn                // the picture with what the layer hid drawn in
	stageReady                // the finished scene: the window puts the layers together
	stageFailed               // nothing more comes (the depth could not be estimated)
)

// step is a step of the making as the window shows it: its picture (the window keeps showing
// the one before when there is none), its line, how much of the whole
// was done when it came, and the finished scene (stageReady).
type step struct {
	stage stage
	pic   image.Image
	text  string
	total float64
	scene *Scene
}

// progress is how the making of the deep picture goes, which runs beside the window. Its steps
// are queued for the window, which shows each at least a while (minShow), so that a step that
// is quick (a depth estimated in a fraction of a second) is still seen; the drawing's line and
// how far it is are taken as they are.
type progress struct {
	mu    sync.Mutex
	steps []step
	stage stage   // the newest step's
	text  string  // the newest line (of the drawing as it goes)
	done  float64 // of the drawing, 0 to 1
	total float64 // of the whole making, 0 to 1 (the window adds the putting together)
	// the shares of the whole: the depth's, the layers', and the window's putting together
	wDepth, wLayers float64
	load            loading // the model being made ready, if one is
}

// loading is a model being made ready for its first use (ai.WithLoading): downloaded and
// converted, compiled for the NPU, or loaded, since when, and how long its load took the last
// time here (0: not known), with the newest line of its log.
type loading struct {
	use, model string
	stage      ai.Stage
	since      time.Time
	expected   time.Duration
	line       string
	phase      string // of stagePhase: imagegen's phase
}

// stagePhase is a loading of a phase of the picture model before its steps (imagegen.Phase*),
// the first picture of a size opening its programs: told by part, not by ai.WithLoading.
const stagePhase ai.Stage = "parallax-phase"

// phaseClock times the picture model's phases before its steps, and keeps how long each took
// in the user's cache (phases.json), for the next run to show how long it is expected to take.
type phaseClock struct {
	rt    *ai.Runtime
	file  string
	model string             // the picture model's name, once it is loaded
	last  map[string]float64 // seconds, by model and phase
	cur   string             // the phase going on, "" when none
	label string             // what the panel says of it
	since time.Time          // when it began
	// of the steps: the share done, when the last step was done, and how many there are
	done   float64
	stepAt time.Time
	steps  int
}

func newPhaseClock(rt *ai.Runtime) *phaseClock {
	c := &phaseClock{rt: rt, last: map[string]float64{}}
	if dir, err := os.UserCacheDir(); err == nil {
		c.file = filepath.Join(dir, "hwnpu", "parallax", "phases.json")
		if b, err := os.ReadFile(c.file); err == nil {
			_ = json.Unmarshal(b, &c.last)
		}
	}
	return c
}

// tell is told of a phase of the picture model (imagegen.Phase*) or of a step done (done, the
// share of the drawing): it ends the phase before, whose name and time it returns ("" when
// none ended). The steps and the decoding are timed as one, "drawing".
func (c *phaseClock) tell(phase string, done float64) (ended string, took time.Duration) {
	now := time.Now()
	if c.model == "" { // the phases come after the picture model's load
		for _, l := range c.rt.Loaded() {
			if l.Use == "image" {
				c.model = l.Model
			}
		}
	}
	switch {
	case phase == "": // a step done
		if c.cur == imagegen.PhaseSteps && done > c.done {
			n := int(math.Round(1/done)) - 1 // done is step / (steps + 1)
			if c.steps == 0 && n > 0 {
				c.steps = n
			}
			step := int(math.Round(done * float64(c.steps+1)))
			ended, took = fmt.Sprintf("step %d of %d", step, c.steps), now.Sub(c.stepAt)
			if step > c.steps {
				ended = imagegen.PhaseDecode // the last done is of the decoding after the steps
			}
			c.done, c.stepAt = done, now
		}
		return ended, took
	case phase == imagegen.PhaseDecode:
		c.label = phase
		return "", 0
	}
	if c.cur != "" && c.cur != imagegen.PhaseSteps {
		ended, took = c.cur, now.Sub(c.since)
		c.last[c.model+" "+ended] = took.Seconds()
	}
	c.cur, c.label, c.since, c.stepAt, c.done, c.steps = phase, phase, now, now, 0, 0
	return ended, took
}

// drawn ends the drawing: its time is kept for the next.
func (c *phaseClock) drawn() (ended string, took time.Duration) {
	if c.cur != imagegen.PhaseSteps {
		return "", 0
	}
	took = time.Since(c.since)
	c.last[c.model+" "+imagegen.PhaseSteps] = took.Seconds()
	c.cur = ""
	return "drawing and decoding", took
}

// loading is what the panel's second line shows of it.
func (c *phaseClock) loading() loading {
	if c.cur == "" {
		return loading{}
	}
	l := loading{use: "image", model: c.model, stage: stagePhase, phase: c.label, since: c.since,
		expected: time.Duration(c.last[c.model+" "+c.cur] * float64(time.Second))}
	if c.cur == imagegen.PhaseSteps {
		if c.done > 0 {
			// at this pace: the steps so far over their share
			l.expected = time.Duration(float64(c.stepAt.Sub(c.since)) / c.done)
		}
		if c.label == imagegen.PhaseSteps {
			l.phase = "drawing, step 1" // how many there are is known from the first done
			if c.steps > 0 {
				step := min(int(math.Round(c.done*float64(c.steps+1)))+1, c.steps)
				l.phase = fmt.Sprintf("drawing, step %d of %d", step, c.steps)
			}
		}
	}
	return l
}

// save writes the times down (a failure only leaves the next run without them).
func (c *phaseClock) save() {
	if c.file == "" {
		return
	}
	if b, err := json.MarshalIndent(c.last, "", "  "); err == nil && os.MkdirAll(filepath.Dir(c.file), 0o755) == nil {
		_ = os.WriteFile(c.file, b, 0o644)
	}
}

// loading tells p of a model being made ready (ai.WithLoading).
func (p *progress) loading(pr ai.Progress) {
	p.set(func(p *progress) {
		l := &p.load
		if pr.Stage == ai.StageDone {
			if l.use == pr.Use {
				*l = loading{}
			}
			return
		}
		if l.use != pr.Use || l.stage != pr.Stage {
			*l = loading{use: pr.Use, model: pr.Model, stage: pr.Stage, since: pr.Since, expected: pr.Expected}
		}
		if line := logLine(pr.Message); line != "" {
			l.line = line
		}
	})
}

// logLine is a line of a model's log as the window shows it: without the spinner and the bar
// of a line of progress redrawn on a terminal (internal/console), the rest (the label, the
// share, the bytes) as it is.
func logLine(s string) string {
	var b strings.Builder
	bar := false
	for _, r := range s {
		switch {
		case r == '▕':
			bar = true
		case r == '▏':
			bar = false
		case bar, r >= 0x2800 && r <= 0x28ff: // in the bar, a braille spinner
		default:
			b.WriteRune(r)
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

// newProgress is the progress of a picture to be made deep in n layers (1: its depth alone; 0:
// as many as its depth has steps).
func newProgress(img image.Image, n int) *progress {
	p := &progress{wDepth: 0.06, wLayers: 0.88}
	if n == 1 {
		p.wDepth, p.wLayers = 0.85, 0
	}
	p.show(stageDepth, img, "Estimating how far each point of the picture is")
	return p
}

func (p *progress) set(fn func(p *progress)) {
	p.mu.Lock()
	defer p.mu.Unlock()
	fn(p)
}

// show queues a step for the window.
func (p *progress) show(st stage, pic image.Image, text string) {
	p.set(func(p *progress) {
		p.stage, p.text = st, text
		p.steps = append(p.steps, step{stage: st, pic: pic, text: text, total: p.total})
	})
}

// ready queues the finished scene: the window puts its layers together.
func (p *progress) ready(s *Scene) {
	p.set(func(p *progress) {
		p.total = p.wDepth + p.wLayers
		p.stage = stageReady
		p.steps = append(p.steps, step{stage: stageReady, scene: s, total: p.total})
	})
}

// news is what the window takes of the progress: the steps queued since it last took, the
// drawing's line and how far it is, how much of the whole is done, and the share of the putting
// together.
type news struct {
	steps       []step
	text        string
	done, total float64
	wCompose    float64
	load        loading
}

func (p *progress) take() news {
	p.mu.Lock()
	defer p.mu.Unlock()
	n := news{steps: p.steps, text: p.text, done: p.done, total: p.total, wCompose: 1 - p.wDepth - p.wLayers, load: p.load}
	p.steps = nil
	return n
}

// minShow is how long the window shows a step at least before the next.
func minShow(st stage) time.Duration {
	switch st {
	case stageDepth:
		return 1800 * time.Millisecond
	case stageNear:
		return 4 * time.Second // the depth, a while: the first thing the models found
	case stageDrawn:
		return 3 * time.Second // what the model drew where the layer was: the step's result
	case stageCut:
		return 2500 * time.Millisecond
	}
	return 0
}

// stack is the scene of a picture in layers, from the farthest: each layer a scene over the one
// behind it. The farthest layer's nearness is pulled apart as a picture of one layer's is (Far);
// each nearer layer, a thing cut out of the picture, is given its relief (Relief): the layer
// sits where Far puts it, and inside it the nearness is as the depth model says, with a
// roundness added from its outline in.
func stack(layers []layering.Layer, k look) *Scene {
	var top *Scene
	for i, l := range layers {
		near := Layers(l.Near, k.far)
		if i > 0 {
			near = Relief(l.Near, alphaMask(l.Img), l.Img.Rect.Dx(), l.Img.Rect.Dy(), k.far, k.relief, k.round)
		}
		s := NewScene(l.Img, near, k.edge, 0)
		s.Behind, top = top, s
	}
	return top
}

// alphaMask is where a layer has its colour (alpha over a half).
func alphaMask(img *image.NRGBA) []bool {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	m := make([]bool, w*h)
	for y := 0; y < h; y++ {
		row := img.Pix[y*img.Stride:]
		for x := 0; x < w; x++ {
			m[y*w+x] = row[4*x+3] >= 128
		}
	}
	return m
}

// part parts the picture into layers (samples/internal/layering: the depth and the picture
// model in turns) and returns its scene. Each step is told on a line of log, and to prog when
// not nil: the layer lifted out, the drawing of what it hid as it goes and what was drawn, and
// how far the whole is.
func (c *picture) part(ctx context.Context, rt *ai.Runtime, p *parts, src *source, prog *progress, log io.Writer) (*Scene, error) {
	if prog == nil {
		prog = newProgress(src.img, c.Layers)
	}
	w, h := src.img.Rect.Dx(), src.img.Rect.Dy()
	o := layering.Options{Layers: c.Layers, Far: c.Far, Edge: -1, Hint: c.Hint, Seed: c.Seed}
	if c.Layers == 0 {
		// how many layers too, from the depth: none where it has no step, and the picture moves
		// by its depth alone (the depth's scene is the one already shown)
		if o.Cuts = layering.Auto(src.near, w, h, c.Far, layering.AutoMost); len(o.Cuts) == 0 {
			fmt.Fprintf(log, "no layers: the depth has no step along %.0f %% of a cut's outline, so the picture moves by its depth alone\n", 100*layering.StepOK)
			return NewScene(src.img, src.near, c.edge(w), c.Far), nil
		}
		fmt.Fprintf(log, "%d layers: the depth steps where it is cut\n", len(o.Cuts)+1)
	}
	phases := newPhaseClock(rt)
	defer phases.save()
	o.Progress = func(s layering.Step) {
		of := fmt.Sprintf("Layer %d of %d", s.Layer, s.Of) // 1 the farthest; the nearest is cut first
		if s.Layer == s.Of {
			of += ", the nearest"
		}
		switch s.Kind {
		case layering.Cut:
			text := fmt.Sprintf("%s: lifted out of the picture (%.0f %% of it)", of, 100*s.Share)
			fmt.Fprintln(log, text)
			prog.show(stageCut, s.Cutout, text)
		case layering.Drawing:
			if s.Phase != "" || s.Done > 0 {
				// what the picture model does, on the panel's second line with its time: each
				// phase before the steps, then the steps and the decoding as one, with how
				// long they took the last time here and, once a step is done, at this pace
				if line, took := phases.tell(s.Phase, s.Done); line != "" {
					fmt.Fprintf(log, "%s: the picture model, %s: %.1f s\n", of, line, took.Seconds())
				}
				prog.set(func(p *progress) { p.load = phases.loading() })
				if s.Phase != "" && s.Phase != imagegen.PhaseSteps {
					return
				}
			}
			text := fmt.Sprintf("%s: drawing what was behind it, %.0f %%", of, 100*s.Done)
			if s.Done == 0 && s.Phase != imagegen.PhaseSteps {
				text = of + ": getting the picture model ready"
			}
			if s.Try > 1 {
				text += fmt.Sprintf(" (try %d)", s.Try)
			}
			prog.set(func(p *progress) {
				if p.stage != stageDrawing { // its first news: a step of its own, on the painted picture
					p.steps = append(p.steps, step{stage: stageDrawing, text: text, total: p.total})
				}
				p.stage, p.text, p.done = stageDrawing, text, s.Done
			})
		case layering.Drawn:
			if line, took := phases.drawn(); line != "" {
				fmt.Fprintf(log, "%s: the picture model, %s: %.1f s\n", of, line, took.Seconds())
			}
			prog.set(func(p *progress) { p.load = loading{} })
			text := of + ": what was behind it is drawn"
			if s.Seam >= layering.SeamOK {
				text = fmt.Sprintf("%s: its shape is still in the drawing, drawn again (try %d)", of, s.Try)
			}
			fmt.Fprintf(log, "%s (seam %.0f)\n", text, s.Seam)
			prog.show(stageDrawn, s.Picture, text)
		case layering.Kept:
			text := fmt.Sprintf("%s stays in the picture: what was behind it was not drawn well", of)
			fmt.Fprintln(log, text)
			prog.show(stageDrawn, nil, text)
		case layering.Parted:
			text := fmt.Sprintf("%d layers", len(s.Layers))
			if s.Left > 0 {
				text += fmt.Sprintf(", %d more to cut", s.Left)
			}
			fmt.Fprintln(log, text)
		}
		prog.set(func(p *progress) { p.total = p.wDepth + p.wLayers*s.Total })
	}
	layers, err := layering.Split(ctx, src.img, src.near, p.split.Models(rt), o)
	if err == nil {
		err = p.split.Done(rt)
	}
	if err == nil && c.SaveLayers != "" {
		err = layering.Write(c.SaveLayers, layers)
	}
	if err != nil {
		return nil, hint(err)
	}
	return stack(layers, c.look(w)), nil
}

// depthColors is a nearness as a picture in colours, the near warm and the far cool (Google's
// Turbo, in its polynomial approximation): how the depth is shown while the picture is made.
func depthColors(near []float32, w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i, v := range near {
		x := float64(min(max(v, 0), 1))
		r := 0.13572138 + x*(4.61539260+x*(-42.66032258+x*(132.13108234+x*(-152.94239396+x*59.28637943))))
		g := 0.09140261 + x*(2.19418839+x*(4.84296658+x*(-14.18503333+x*(4.27729857+x*2.82956604))))
		b := 0.10667330 + x*(12.64194608+x*(-60.58204836+x*(110.36276771+x*(-89.90310912+x*27.34824973))))
		c := func(f float64) uint8 { return uint8(min(max(f, 0), 1)*255 + 0.5) }
		img.Pix[4*i], img.Pix[4*i+1], img.Pix[4*i+2], img.Pix[4*i+3] = c(r), c(g), c(b), 255
	}
	return img
}

// toRGBA is a picture as RGBA.
func toRGBA(img image.Image) *image.RGBA {
	if r, ok := img.(*image.RGBA); ok && r.Rect.Min == (image.Point{}) {
		return r
	}
	return layering.Stretch(img, img.Bounds().Dx(), img.Bounds().Dy())
}
