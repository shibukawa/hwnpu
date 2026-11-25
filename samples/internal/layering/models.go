package layering

import (
	"context"
	"image"
	"math"

	"github.com/shibukawa/bonsaiai/ai"
	"github.com/shibukawa/bonsaiai/ai/imagegen"
	"github.com/shibukawa/bonsaiai/ai/vision"
)

// Calls are the two calls of an app that the parting runs: the depth of a picture, and the
// picture an editing model draws from a painted one. They run in turns, each model unloaded
// before the other is asked (the use's Release), so that the two need not fit in memory
// together.
type Calls struct {
	depth   ai.Use[*vision.DepthEstimator]
	artist  ai.Use[*imagegen.Generator]
	depthOf *ai.Call[image.Image, *vision.DepthMap]
	behind  *ai.Call[imagegen.Request, image.Image]
	// drawing is the picture model's node, whose settings change with each picture (the seed,
	// who is told of its steps)
	drawing *ai.ModelNode[imagegen.Request, image.Image, imagegen.Options]
}

// Register adds the two calls to an app, and says that their models run in turns (App.InTurns):
// the app's plan counts the larger alone.
func Register(app *ai.App, depth ai.Use[*vision.DepthEstimator], artist ai.Use[*imagegen.Generator]) *Calls {
	app.InTurns(depth, artist)
	c := &Calls{depth: depth, artist: artist, drawing: imagegen.GenerateNode(artist, imagegen.Options{})}
	c.depthOf = ai.NewCall[image.Image, *vision.DepthMap](app, "depth of the picture").Then(vision.DepthNode(depth))
	c.behind = ai.NewCall[imagegen.Request, image.Image](app, "what a layer hid").Then(c.drawing)
	return c
}

// Near is how near each pixel of a picture is, and the depth model's name. The picture model is
// unloaded before the depth is asked, and the depth model after it has answered.
func (c *Calls) Near(ctx context.Context, rt *ai.Runtime, img *image.RGBA) ([]float32, string, error) {
	if err := c.artist.Release(rt); err != nil {
		return nil, "", err
	}
	d, err := c.depthOf.Do(ctx, rt, img)
	if err != nil {
		return nil, "", err
	}
	name := ""
	if est, err := c.depth.Get(ctx, rt); err == nil {
		name = est.Model()
	}
	if err := c.depth.Release(rt); err != nil {
		return nil, "", err
	}
	return d.Near(img.Rect.Dx(), img.Rect.Dy()), name, nil
}

// Draw is the picture the picture model draws from a prompt and a reference. The reference is
// stretched to the smallest size the model draws of its shape (smallest), and the answer back.
func (c *Calls) Draw(ctx context.Context, rt *ai.Runtime, prompt string, ref *image.RGBA, seed uint64, progress func(done float64), phase func(string)) (*image.RGBA, error) {
	g, err := c.artist.Get(ctx, rt)
	if err != nil {
		return nil, err
	}
	size := smallest(g, ref.Rect.Dx(), ref.Rect.Dy())
	c.drawing.Update(func(o *imagegen.Options) {
		o.Seed, o.Phase = seed, phase
		o.Progress = func(step, steps int) {
			if progress != nil {
				progress(float64(step) / float64(steps+1)) // the steps, and the decoding after them
			}
		}
	})
	out, err := c.behind.Do(ctx, rt, imagegen.Request{Prompt: prompt, References: []image.Image{Stretch(ref, size.X, size.Y)}})
	if err != nil {
		return nil, err
	}
	return Stretch(out, ref.Rect.Dx(), ref.Rect.Dy()), nil
}

// smallest is the smallest size the model draws of the shape nearest w x h: what a layer hid is
// drawn behind it and seen only where the layer moves aside, so the small size does (and on
// Apple's Neural Engine a 512 x 512 picture's steps take a quarter of a 1024 x 1024 one's, whose
// programs are also compiled at their first use). Of the sizes the model edits at when it says
// (Generator.EditSizes: FLUX.2 klein on Apple's Neural Engine edits at 1024 x 1024 and 512 x 512
// alone, the 1024 ones compiled on the device), else of those it draws.
func smallest(g *imagegen.Generator, w, h int) image.Point {
	sizes := g.EditSizes()
	if len(sizes) == 0 {
		sizes = g.Sizes()
	}
	best := sizes[0]
	shape := func(p image.Point) float64 { return math.Log(float64(p.X) / float64(p.Y)) }
	for _, s := range sizes[1:] {
		if math.Abs(shape(s)-shape(image.Pt(w, h))) < math.Abs(shape(best)-shape(image.Pt(w, h))) {
			best = s
		}
	}
	for _, s := range sizes {
		if math.Abs(shape(s)-shape(best)) < 0.01 && s.X*s.Y < best.X*best.Y {
			best = s
		}
	}
	return best
}

// Models are the calls as Split asks them, run on rt.
func (c *Calls) Models(rt *ai.Runtime) Models {
	return Models{
		Near: func(ctx context.Context, img *image.RGBA) ([]float32, error) {
			near, _, err := c.Near(ctx, rt, img)
			return near, err
		},
		Draw: func(ctx context.Context, prompt string, ref *image.RGBA, seed uint64, progress func(float64), phase func(string)) (*image.RGBA, error) {
			return c.Draw(ctx, rt, prompt, ref, seed, progress, phase)
		},
	}
}

// Done unloads the two models, after the last picture.
func (c *Calls) Done(rt *ai.Runtime) error {
	if err := c.artist.Release(rt); err != nil {
		return err
	}
	return c.depth.Release(rt)
}
