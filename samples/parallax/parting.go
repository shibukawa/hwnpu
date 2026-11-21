package main

import (
	"context"
	"fmt"
	"image"
	"io"
	"sync"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/samples/layering"
)

// source is the picture and its nearness as the scene was first made from them: what is parted
// into layers.
type source struct {
	img  *image.RGBA
	near []float32
}

// progress is what the parting, which runs beside the window, shows in it: the window takes it
// each frame.
type progress struct {
	mu sync.Mutex
	// scene is the picture in the layers there are so far, until the window takes it
	scene *Scene
	// inset is a small picture of the step (what the model was asked to draw from, what it
	// drew), fresh until the window takes it
	inset *image.RGBA
	fresh bool
	text  string
	// total is how much of the parting is done, 0 to 1
	total  float64
	active bool
}

func (p *progress) set(fn func(p *progress)) {
	p.mu.Lock()
	defer p.mu.Unlock()
	fn(p)
}

// take is what the window shows now: a new scene and a new inset when there is one (the inset
// nil with fresh for "none any more"), the line of the step, how much is done, and whether the
// parting still runs.
func (p *progress) take() (scene *Scene, inset *image.RGBA, fresh bool, text string, total float64, active bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	scene, inset, fresh = p.scene, p.inset, p.fresh
	p.scene, p.fresh = nil, false
	return scene, inset, fresh, p.text, p.total, p.active
}

// stack is the scene of a picture in layers, from the farthest: each layer a scene over the one
// behind it. The things of one layer still have outlines inside it (a flower before the grass),
// so the near things are made larger in each layer's nearness as in a picture of one layer.
func stack(layers []layering.Layer, edge int, far float64) *Scene {
	var top *Scene
	for _, l := range layers {
		s := NewScene(l.Img, l.Near, edge, far)
		s.Behind, top = top, s
	}
	return top
}

// part parts the picture into layers (samples/layering: the depth and the picture model in
// turns) and returns its scene. Each step is told on a line of log, and to prog when not nil:
// the picture the model is asked to draw from and what it drew, how far the work is, and the
// scene of the layers so far each time a layer is done, so that a window shows the picture
// coming apart.
func (c *picture) part(ctx context.Context, rt *ai.Runtime, p *parts, src *source, prog *progress, log io.Writer) (*Scene, error) {
	if prog == nil {
		prog = &progress{}
	}
	w := src.img.Rect.Dx()
	small := func(img *image.RGBA) *image.RGBA { // for the corner of the window
		return layering.Stretch(img, 320, max(1, 320*img.Rect.Dy()/img.Rect.Dx()))
	}
	prog.set(func(p *progress) {
		p.active, p.text = true, fmt.Sprintf("parting the picture into %d layers", c.Layers)
	})
	o := layering.Options{Layers: c.Layers, Far: c.Far, Edge: -1, Hint: c.Hint, Seed: c.Seed}
	o.Progress = func(s layering.Step) {
		var text string
		var inset *image.RGBA
		var scene *Scene
		switch s.Kind {
		case layering.Cut:
			text = fmt.Sprintf("layer %d of %d cut out: %.0f %% of the picture", s.Layer, s.Of, 100*s.Share)
			fmt.Fprintln(log, text)
		case layering.Asked:
			text, inset = fmt.Sprintf("layer %d of %d: its shape painted over; the picture model is asked what is behind it", s.Layer, s.Of), small(s.Picture)
		case layering.Drawing:
			text = fmt.Sprintf("layer %d of %d: drawing what is behind it, %.0f %%", s.Layer, s.Of, 100*s.Done)
			if s.Try > 1 {
				text += fmt.Sprintf(" (try %d)", s.Try)
			}
		case layering.Drawn:
			text, inset = fmt.Sprintf("layer %d of %d: what is behind it is drawn", s.Layer, s.Of), small(s.Picture)
			if s.Seam >= layering.SeamOK {
				text = fmt.Sprintf("layer %d of %d: the shape is still in the drawing (try %d)", s.Layer, s.Of, s.Try)
			}
			fmt.Fprintf(log, "%s (seam %.0f)\n", text, s.Seam)
		case layering.Kept:
			scene = stack(s.Layers, c.edge(w), c.Far)
			text = fmt.Sprintf("layer %d of %d stays in the picture: what is behind it was not drawn well; %d layers", s.Layer, s.Of, len(s.Layers))
			fmt.Fprintln(log, text)
		case layering.Parted:
			scene = stack(s.Layers, c.edge(w), c.Far)
			text = fmt.Sprintf("%d layers", len(s.Layers))
			if s.Left > 0 {
				text += fmt.Sprintf(", %d more to cut", s.Left)
			}
			fmt.Fprintln(log, text)
		}
		prog.set(func(p *progress) {
			p.text, p.total = text, s.Total
			if inset != nil {
				p.inset, p.fresh = inset, true
			}
			if scene != nil {
				p.scene = scene
			}
		})
	}
	layers, err := layering.Split(ctx, src.img, src.near, p.split.Models(rt), o)
	if err == nil {
		err = p.split.Done(rt)
	}
	if err == nil && c.SaveLayers != "" {
		err = layering.Write(c.SaveLayers, layers)
	}
	if err != nil {
		err = hint(err)
		prog.set(func(p *progress) {
			p.active, p.inset, p.fresh = false, nil, true
			p.text = fmt.Sprintf("not parted into layers: %v", err)
		})
		return nil, err
	}
	scene := stack(layers, c.edge(w), c.Far)
	prog.set(func(p *progress) {
		p.active, p.inset, p.fresh, p.total = false, nil, true, 1
		if len(layers) == c.Layers {
			p.text = fmt.Sprintf("%d layers", len(layers))
		}
	})
	return scene, nil
}
