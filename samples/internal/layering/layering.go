// Package layering parts a picture into layers by depth, with what each layer hid drawn in. It
// is the samples' own (layers writes the layers as a Photoshop document, parallax draws them as
// if the picture were deep); the models are given to it as functions, so that it holds none.
//
// The picture is cut from the nearest layer on. A layer is cut out where the picture is nearer
// than a cut; its shape is painted over in a plain colour and the picture model is asked to
// take the shape away and draw the scenery there; the nearness of that picture is estimated and
// brought to the first estimate's scale, and the next layer is cut from it. The farthest layer
// is the picture that is left. Each layer has its own nearness, which goes on past the layer's
// edge, so that a layer drawn by its nearness ends at its edge.
package layering

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// Models are the two models the parting asks, as the app runs them.
type Models struct {
	// Near is how near each pixel of a picture is, row major: 1 the nearest point, 0 the
	// farthest, in proportion to the inverse of the depth.
	Near func(ctx context.Context, img *image.RGBA) ([]float32, error)
	// Draw is the picture an editing model draws from a prompt and a reference (image 1 in the
	// prompt), at the reference's size. It calls progress, when it can, with how much of the
	// drawing is done, 0 to 1, and phase with what it does before its steps (imagegen's
	// Options.Phase: opening the programs of the size, reading the reference and the prompt).
	Draw func(ctx context.Context, prompt string, ref *image.RGBA, seed uint64, progress func(done float64), phase func(string)) (*image.RGBA, error)
}

// Options are how a picture is parted.
type Options struct {
	// Layers is how many layers (1 to 8) when Cuts is empty: the cuts are then found from the
	// nearness (Cuts of this package). 0 finds how many too (Auto, up to AutoMost): none where
	// the depth has no step, and the picture is then one layer.
	Layers int
	// Cuts are where to cut, as nearness on the curve of Far, ascending, each between 0 and 1.
	Cuts []float32
	// Far is how far the far end of the nearness is pulled apart before it is cut (0: not at
	// all; 30 is a good value).
	Far float64
	// Edge is how many pixels a layer reaches past where the depth cuts it, so that it holds
	// its own outline; below 0: a 200th of the picture's width.
	Edge int
	// Clean is how many pixels past a layer's edge the layer behind takes the picture drawn
	// instead of the picture before. The picture before may hold there what the depth missed
	// of the layer (the tips of grass, a strand of hair), which stays behind when the layer
	// moves; the picture drawn does not, but where it does not go on from the surroundings
	// exactly, it shows as a rim about the layer. 0 takes it only where the layer was.
	Clean int
	// Hint is added to the request to draw what a layer hid.
	Hint string
	// Seed is the seed of the first try of each picture drawn (the tries after take the next
	// ones); 0 leaves the seed to the model.
	Seed uint64
	// Tries is how many times what a layer hid is drawn at most (3 when 0): it is drawn again
	// when the model left the layer's shape there. When no try is good, the layer is not cut out
	// and the parting ends with the layers there are (Kept).
	Tries int
	// Progress, when not nil, is told each step.
	Progress func(Step)
}

// Layer is a layer of a picture: its picture, transparent where the layer has nothing, and how
// near each of its points is, which goes on past the layer's own part.
type Layer struct {
	Img  *image.NRGBA
	Near []float32
}

// The kinds of steps.
const (
	// Cut: a layer was cut out. Layer is its number, Cutout the layer (transparent where it has
	// nothing).
	Cut = "cut"
	// Asked: the picture model is asked to draw what the layer hid. Picture is what it draws
	// from: the picture with the layer's shape painted over; Mask is where it is painted (1),
	// row major at the picture's size.
	Asked = "asked"
	// Drawing: the model is drawing. Done is how much of this drawing is done, 0 to 1, Try the
	// try's number from 1, Phase what it does before its steps (imagegen's phases), "" when
	// not told.
	Drawing = "drawing"
	// Drawn: the model has drawn. Picture is its drawing, Seam how much of the shape is left
	// (SeamOK and over: it is drawn again while tries are left).
	Drawn = "drawn"
	// Kept: the model drew nothing in any try that goes on from the surroundings, so the layer
	// was not cut out: it stays in the picture behind the layers cut before, which is the
	// farthest layer, and the parting ends. Layers are the layers there are.
	Kept = "kept"
	// Parted: the picture is in layers so far. Layers are the layers cut out and, as the
	// farthest, the picture behind them with its own nearness; Left how many layers are still
	// to be cut from that picture.
	Parted = "parted"
)

// Step is a step of the parting, for a log or a screen.
type Step struct {
	Kind string
	// Layer is the layer being cut, from 1 (the farthest); Of how many layers there will be.
	Layer, Of int
	// Share is the share of the picture the layer covers (Cut).
	Share   float64
	Picture *image.RGBA
	Cutout  *image.NRGBA
	Mask    []float32
	Seam    float64
	Try     int
	Done    float64
	Phase   string
	Layers  []Layer
	Left    int
	// Total is how much of the whole parting is done, 0 to 1: each layer but the farthest is an
	// equal share of it, nearly all of which is its drawing (a try that is drawn again starts
	// its share again).
	Total float64
}

// SeamOK is the seam under which a part drawn again is taken as going on from its surroundings.
const SeamOK = 50

// Curve is a nearness with its far end pulled apart by far (Options.Far), as it is cut.
func Curve(near []float32, far float64) []float32 { return curve(near, far) }

// Cuts are where a nearness is cut into n layers: the cuts that leave each layer's nearness, on
// the curve of far, closest together.
func Cuts(near []float32, n int, far float64) []float32 { return cuts(curve(near, far), n) }

// AutoMost is the most layers Split finds from the depth (Options.Layers 0).
const AutoMost = 3

// StepOK is the least share of a cut's outline that steps in the depth (Steps) for the cut to
// part a layer when the layers are found from the depth (Auto).
const StepOK = 0.2

// Steps is, for each cut of at, the share of its outline where the nearness steps: where what
// is near the cut stands before what is behind it, instead of a surface going on past it. near
// is row major, w x h; the cuts are on the curve of far.
func Steps(near []float32, w, h int, far float64, at []float32) []float64 {
	c := curve(near, far)
	out := make([]float64, len(at))
	for i, x := range at {
		out[i] = steep(c, w, h, x)
	}
	return out
}

// Auto are the cuts of the most layers, up to most, whose every cut steps along StepOK of its
// outline or more (Steps): a thing parted from what is behind it, so that it moves before it
// with what it hid drawn in behind. None when no cut does: the depth goes on (animals lying on
// a blanket, a field under the sky), and the picture moves best by its depth alone.
func Auto(near []float32, w, h int, far float64, most int) []float32 {
	for n := min(most, 8); n >= 2; n-- {
		at := Cuts(near, n, far)
		ok := true
		for _, s := range Steps(near, w, h, far, at) {
			ok = ok && s >= StepOK
		}
		if ok {
			return at
		}
	}
	return nil
}

// Soften is a mask (row major, w x h) with its edge made a ramp 2r pixels wide.
func Soften(m []float32, w, h, r int) []float32 { return soften(m, w, h, r) }

// Stretch is a picture at w x h (its shape is not kept), for a model that draws other sizes.
func Stretch(img image.Image, w, h int) *image.RGBA { return stretch(img, w, h) }

// Split parts a picture into layers, from the farthest to the nearest. near is the picture's
// nearness (Models.Near's).
func Split(ctx context.Context, img *image.RGBA, near []float32, m Models, o Options) ([]Layer, error) {
	at := o.Cuts
	if len(at) == 0 {
		switch {
		case o.Layers == 0:
			at = Auto(near, img.Rect.Dx(), img.Rect.Dy(), o.Far, AutoMost)
		case o.Layers < 1 || o.Layers > 8:
			return nil, fmt.Errorf("layering: %d layers; 1 to 8, or 0 to find them from the depth", o.Layers)
		case o.Layers > 1:
			at = Cuts(near, o.Layers, o.Far)
		}
		if len(at) == 0 {
			// one layer: the picture and its nearness, nothing drawn
			whole := make([]float32, len(near))
			for i := range whole {
				whole[i] = 1
			}
			return []Layer{{Img: cutOut(img, whole), Near: near}}, nil
		}
	}
	for i, x := range at {
		if x <= 0 || x >= 1 || i > 0 && x <= at[i-1] {
			return nil, fmt.Errorf("layering: the cuts %v are not ascending between 0 and 1", at)
		}
	}
	// the drawing is 95 % of a layer's share of the work, the depth after it the rest
	const drawing = 0.95
	finished := 0 // the layers whose share is done
	tell := func(s Step, share float64) {
		if o.Progress != nil {
			s.Of = len(at) + 1
			s.Total = min((float64(finished)+share)/float64(len(at)), 1)
			o.Progress(s)
		}
	}
	w, h := img.Rect.Dx(), img.Rect.Dy()
	edge := o.Edge
	if edge < 0 {
		edge = max(1, w/200)
	}
	margin := max(4, w/64)                // how far past the layer the picture is drawn again
	clean := min(max(o.Clean, 0), margin) // and how far past it that picture is taken
	tries := o.Tries
	if tries <= 0 {
		tries = 3
	}

	// from the nearest layer to the one before the farthest: cut it out of the picture, draw
	// what it hid, and go on with that picture
	plate := img
	layers := make([]Layer, len(at)+1)
	// parted are the layers so far, the picture left as the farthest
	parted := func(from int) []Layer {
		out := []Layer{{Img: cutOut(plate, above(near, -1)), Near: near}}
		for _, l := range layers[from:] {
			if l.Img != nil {
				out = append(out, l)
			}
		}
		return out
	}
	for l := len(at); l >= 1; l-- {
		in := above(curve(near, o.Far), at[l-1])
		if covered(in) == 0 {
			finished++
			continue // nothing is that near
		}
		layers[l] = Layer{Img: cutOut(plate, soften(grow(in, w, h, edge), w, h, 1)), Near: extend(near, in, w, h)}
		tell(Step{Kind: Cut, Layer: l + 1, Share: covered(in), Cutout: layers[l].Img}, 0)

		// the layer's shape, and a little more, in a plain colour the picture has little of
		hole := grow(in, w, h, edge+margin)
		colour, word := key(plate)
		prompt := fmt.Sprintf("Remove the %s shape from image 1. Where it was, draw the scenery that is behind it, continuing the surroundings, so that no trace of the shape is left. Do not add people or animals. Keep everything else unchanged.", word)
		if o.Hint != "" {
			prompt += " " + strings.TrimSuffix(o.Hint, ".") + "."
		}
		asked := paint(plate, hole, colour)
		tell(Step{Kind: Asked, Layer: l + 1, Picture: asked, Mask: hole}, 0)
		// The model sometimes leaves the shape, in its colour or as a shadow. That shows as a
		// seam all along the hole's edge, and the picture is then drawn again from another seed;
		// the one with the least seam is kept.
		var drawn *image.RGBA
		least := math.Inf(1)
		for try := 0; try < tries; try++ {
			seed := o.Seed
			if seed != 0 {
				seed += uint64(try)
			}
			tell(Step{Kind: Drawing, Layer: l + 1, Try: try + 1}, 0)
			out, err := m.Draw(ctx, prompt, asked, seed, func(done float64) {
				done = min(max(done, 0), 1)
				tell(Step{Kind: Drawing, Layer: l + 1, Try: try + 1, Done: done}, drawing*done)
			}, func(phase string) {
				tell(Step{Kind: Drawing, Layer: l + 1, Try: try + 1, Phase: phase}, 0)
			})
			if err != nil {
				return nil, err
			}
			if out.Rect.Dx() != w || out.Rect.Dy() != h {
				out = stretch(out, w, h)
			}
			s := seam(out, hole, w, h)
			tell(Step{Kind: Drawn, Layer: l + 1, Picture: out, Seam: s, Try: try + 1, Done: 1}, drawing)
			if s < least {
				drawn, least = out, s
			}
			if s < SeamOK {
				break
			}
		}
		if least >= SeamOK {
			// No try went on from the surroundings (the shape left as it was, or something else
			// drawn in its place: a model shown a large plain shape may draw people there). A
			// layer with such a picture behind it is worse than no layer: this layer stays in
			// the picture, which is then the farthest layer, and nothing more is cut.
			layers[l] = Layer{}
			finished = len(at)
			tell(Step{Kind: Kept, Layer: l + 1, Seam: least, Layers: parted(l + 1)}, 0)
			return parted(l + 1), nil
		}
		// the picture drawn is taken where the layer was (and clean pixels past it), the picture
		// before elsewhere: the layers put back together are then the picture
		plate = mix(plate, drawn, soften(grow(in, w, h, edge+clean), w, h, max(clean/2, 1)))
		// how near the points drawn are: a second estimate, brought to the first one's scale
		// from the part both saw, and behind the layer cut out
		again, err := m.Near(ctx, plate)
		if err != nil {
			return nil, err
		}
		a, b := fit(again, near, hole)
		limit := uncurve(at[l-1], o.Far)
		merged := make([]float32, len(near))
		for i := range near {
			merged[i] = near[i]
			if hole[i] > 0 {
				merged[i] = min(max(float32(a*float64(again[i])+b), 0), limit)
			}
		}
		near = merged
		finished++
		tell(Step{Kind: Parted, Layer: l + 1, Layers: parted(l), Left: l - 1}, 0)
	}
	return parted(1), nil
}

// Write writes layers into dir, from the farthest: layer-N.png (the picture with its
// transparency) and near-N.png (how near its points are: 16 bit gray, white the nearest).
func Write(dir string, layers []Layer) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	for i, l := range layers {
		depth := image.NewGray16(l.Img.Rect)
		for k, n := range l.Near {
			v := uint16(min(max(n, 0), 1)*65535 + 0.5)
			depth.Pix[2*k], depth.Pix[2*k+1] = uint8(v>>8), uint8(v)
		}
		for name, img := range map[string]image.Image{"layer": l.Img, "near": depth} {
			f, err := os.Create(filepath.Join(dir, fmt.Sprintf("%s-%d.png", name, i+1)))
			if err != nil {
				return err
			}
			if err := png.Encode(f, img); err != nil {
				f.Close()
				return err
			}
			if err := f.Close(); err != nil {
				return err
			}
		}
	}
	return nil
}

// Read reads the layers Write wrote into dir.
func Read(dir string) ([]Layer, error) {
	read := func(file string) (image.Image, error) {
		f, err := os.Open(file)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		img, err := png.Decode(f)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", file, err)
		}
		return img, nil
	}
	var layers []Layer
	for n := 1; ; n++ {
		file := filepath.Join(dir, fmt.Sprintf("layer-%d.png", n))
		if _, err := os.Stat(file); err != nil {
			break
		}
		img, err := read(file)
		if err != nil {
			return nil, err
		}
		depth, err := read(filepath.Join(dir, fmt.Sprintf("near-%d.png", n)))
		if err != nil {
			return nil, fmt.Errorf("layer %d has no nearness: %w", n, err)
		}
		b := img.Bounds()
		if depth.Bounds().Dx() != b.Dx() || depth.Bounds().Dy() != b.Dy() ||
			len(layers) > 0 && (b.Dx() != layers[0].Img.Rect.Dx() || b.Dy() != layers[0].Img.Rect.Dy()) {
			return nil, fmt.Errorf("%s: the layers and their nearness are not of one size", dir)
		}
		l := Layer{Img: image.NewNRGBA(image.Rect(0, 0, b.Dx(), b.Dy())), Near: make([]float32, 0, b.Dx()*b.Dy())}
		for y := 0; y < b.Dy(); y++ {
			for x := 0; x < b.Dx(); x++ {
				l.Img.Set(x, y, img.At(b.Min.X+x, b.Min.Y+y))
				l.Near = append(l.Near, float32(color.Gray16Model.Convert(depth.At(depth.Bounds().Min.X+x, depth.Bounds().Min.Y+y)).(color.Gray16).Y)/0xffff)
			}
		}
		layers = append(layers, l)
	}
	if len(layers) == 0 {
		return nil, fmt.Errorf("%s has no layer-1.png", dir)
	}
	return layers, nil
}
