package layering

import (
	"context"
	"image"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/imagegen"
	"github.com/shibukawa/hwnpu/ai/vision"
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
// stretched to the size the model draws that is nearest its shape, and the answer back.
func (c *Calls) Draw(ctx context.Context, rt *ai.Runtime, prompt string, ref *image.RGBA, seed uint64, progress func(done float64), phase func(string)) (*image.RGBA, error) {
	g, err := c.artist.Get(ctx, rt)
	if err != nil {
		return nil, err
	}
	size := g.SizeFor(ref.Rect.Dx(), ref.Rect.Dy())
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
