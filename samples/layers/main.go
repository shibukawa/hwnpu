// layers is a sample app of the ai package: it parts a picture into layers by depth and writes
// them as a Photoshop document, with what each layer hid drawn in.
//
// The picture's depth is estimated and cut into layers, the nearest first. A layer is cut out;
// where it was is painted over in one plain colour, and the picture model is asked to draw the
// background there (FLUX.2 klein, an edit); the depth of that picture gives the next layer, and
// so on to the farthest, which is whole. In the document a layer can then be moved, scaled or
// hidden, and there is a picture behind it.
//
// It runs its two models in turns, each unloaded before the other is asked (the use's Release),
// so that they need not fit in memory together:
//
//	the picture → depth → cut the nearest layer out → paint it over → draw what is behind
//	    ↑                                                                     |
//	    └───────────────────── the picture without that layer ←───────────────┘
//
//	layers models prepare                # download, convert and compile the models ahead
//	layers plan picture.png              # the layers the picture would be parted into, as a picture (no drawing)
//	layers picture.png -o picture.psd    # part it: a minute for each layer but the last
//	layers picture.png --layers 4 --png parts/ # the layers and how near their points are too: "parallax parts/" draws them
//
// Drawing from a reference runs on AMD's NPU for now.
package main

import (
	"context"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"time"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/cli/konghelper"
	"github.com/shibukawa/hwnpu/ai/imagegen"
	"github.com/shibukawa/hwnpu/ai/vision"
)

// The models: the depth of a picture, and the one that draws what a layer hid.
var (
	depth  = vision.Depth("depth")
	artist = imagegen.Generate("image")
)

// parts are the app's calls, which its commands run.
type parts struct {
	depthOf *ai.Call[image.Image, *vision.DepthMap]
	behind  *ai.Call[imagegen.Request, image.Image]
	// drawing is the picture model's node, whose settings a command's flags change (the seed)
	drawing *ai.ModelNode[imagegen.Request, image.Image, imagegen.Options]
}

func build() (*ai.App, *parts) {
	app := ai.NewApp("layers")
	p := &parts{drawing: imagegen.GenerateNode(artist, imagegen.Options{})}
	p.depthOf = ai.NewCall[image.Image, *vision.DepthMap](app, "depth of the picture").Then(vision.DepthNode(depth))
	p.behind = ai.NewCall[imagegen.Request, image.Image](app, "what a layer hid").Then(p.drawing)
	return app, p
}

var cmds struct {
	Split  SplitCmd          `cmd:"" default:"withargs" help:"Part a picture into layers by depth, draw what each hid, and write a Photoshop document."`
	Plan   PlanCmd           `cmd:"" help:"Show the layers a picture would be parted into, each in a colour of its own: the depth alone, nothing is drawn."`
	Models konghelper.Models `cmd:"" help:"The models layers uses: what they take, download them, compile them for this machine's NPU."`
	Status konghelper.Status `cmd:"" help:"This machine's NPU: the device, how busy it is and what uses it."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	app, p := build()
	k := kong.Parse(&cmds,
		kong.Name("layers"),
		kong.Description("Part a picture into layers by depth, with what each layer hid drawn in, as a Photoshop document: a sample of hwnpu's ai package."),
		kong.UsageOnError(),
		konghelper.Bind(ctx, app),
		kong.Bind(p))
	k.FatalIfErrorf(k.Run())
}

// picture is what the commands share: the picture and how it is parted.
type picture struct {
	Picture string    `arg:"" type:"existingfile" name:"PICTURE" help:"The picture (PNG or JPEG)."`
	Layers  int       `default:"3" help:"How many layers, 2 to 8: where to cut is found from the picture's depth (the cuts that leave each layer's depths closest together)."`
	Cuts    []float32 `placeholder:"NEARNESS,..." help:"Where to cut instead, as nearness from 0 (the farthest) to 1 (the nearest), ascending: 0.3,0.7 makes three layers. \"layers plan\" prints the cuts it finds."`
	Far     float64   `default:"30" help:"How far the far end of the nearness is pulled apart before it is cut (0: not at all). A nearness is the inverse of the depth, so behind something right before the eye all the rest is close to 0."`
	Prepare bool      `help:"Download, convert and compile the models that are not ready instead of failing."`
}

func (c *picture) options() []ai.Option {
	if !c.Prepare {
		return nil
	}
	return []ai.Option{ai.WithAutoPrepare(func(p ai.Progress) {
		if p.Message != "" {
			fmt.Fprintf(os.Stderr, "  %s: %s\n", p.Model, p.Message)
		}
	})}
}

// read is the picture, the nearness of its pixels, and where it is cut (on the curve of --far).
// The depth model is unloaded once it has answered.
func (c *picture) read(ctx context.Context, rt *ai.Runtime, p *parts) (*image.RGBA, []float32, []float32, error) {
	if len(c.Cuts) == 0 && (c.Layers < 2 || c.Layers > 8) {
		return nil, nil, nil, fmt.Errorf("--layers %d: 2 to 8", c.Layers)
	}
	for i, x := range c.Cuts {
		if x <= 0 || x >= 1 || i > 0 && x <= c.Cuts[i-1] {
			return nil, nil, nil, fmt.Errorf("--cuts: %v is not ascending between 0 and 1", c.Cuts)
		}
	}
	f, err := os.Open(c.Picture)
	if err != nil {
		return nil, nil, nil, err
	}
	src, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("%s: %w", c.Picture, err)
	}
	img := stretch(src, src.Bounds().Dx(), src.Bounds().Dy())
	near, err := c.near(ctx, rt, p, img)
	if err != nil {
		return nil, nil, nil, err
	}
	at := c.Cuts
	if len(at) == 0 {
		at = cuts(curve(near, c.Far), c.Layers)
	}
	return img, near, at, nil
}

// near is the nearness of a picture's pixels (1 the nearest, 0 the farthest, in proportion to the
// inverse of the depth), by the depth model, which is unloaded after.
func (c *picture) near(ctx context.Context, rt *ai.Runtime, p *parts, img *image.RGBA) ([]float32, error) {
	t0 := time.Now()
	d, err := p.depthOf.Do(ctx, rt, img)
	if err != nil {
		return nil, hint(err)
	}
	est, _ := depth.Get(ctx, rt)
	fmt.Fprintf(os.Stderr, "depth by %s in %.1f s\n", est.Model(), time.Since(t0).Seconds())
	if err := depth.Release(rt); err != nil {
		return nil, err
	}
	return d.Near(img.Rect.Dx(), img.Rect.Dy()), nil
}

// PlanCmd is layers plan PICTURE.
type PlanCmd struct {
	picture
	Out string `short:"o" type:"path" placeholder:"FILE" help:"The PNG to write (default: PICTURE's name with .layers.png)."`
}

func (c *PlanCmd) Run(ctx context.Context, app *ai.App, p *parts) error {
	rt, err := app.Open(ctx, c.options()...)
	if err != nil {
		return err
	}
	defer rt.Close()
	img, near, at, err := c.read(ctx, rt, p)
	if err != nil {
		return err
	}
	// each layer in a tint of its own, from the farthest: blue, green, yellow, red, ...
	tints := []color.RGBA{{40, 90, 255, 255}, {40, 200, 120, 255}, {250, 210, 40, 255}, {255, 90, 40, 255},
		{220, 40, 200, 255}, {40, 220, 230, 255}, {150, 255, 60, 255}, {255, 255, 255, 255}}
	n := len(at) + 1
	share := make([]int, n)
	out := image.NewRGBA(img.Rect)
	for i, x := range curve(near, c.Far) {
		l := 0
		for l < len(at) && x > at[l] {
			l++
		}
		share[l]++
		t := tints[l%len(tints)]
		for k, tv := range []uint8{t.R, t.G, t.B} {
			out.Pix[4*i+k] = uint8((int(img.Pix[4*i+k]) + int(tv)) / 2)
		}
		out.Pix[4*i+3] = 255
	}
	file := c.Out
	if file == "" {
		file = strings.TrimSuffix(c.Picture, filepath.Ext(c.Picture)) + ".layers.png"
	}
	if err := writePNG(file, out); err != nil {
		return err
	}
	for l := 0; l < n; l++ {
		from, to := float32(0), float32(1)
		if l > 0 {
			from = at[l-1]
		}
		if l < len(at) {
			to = at[l]
		}
		fmt.Printf("layer %d: nearness %.2f to %.2f, %.0f %% of the picture\n", l+1, from, to, 100*float64(share[l])/float64(len(near)))
	}
	cs := make([]string, len(at))
	for i, x := range at {
		cs[i] = fmt.Sprintf("%.2f", x)
	}
	fmt.Printf("--cuts %s\n%s\n", strings.Join(cs, ","), file)
	return nil
}

// SplitCmd is layers PICTURE.
type SplitCmd struct {
	picture
	Out   string `short:"o" type:"path" placeholder:"FILE" help:"The Photoshop document to write (default: PICTURE's name with .psd)."`
	PNG   string `name:"png" type:"path" placeholder:"DIR" help:"Also write into this directory each layer as a PNG with its transparency (layer-N.png, 1 the farthest) and how near its points are (near-N.png: 16 bit gray, white the nearest, on one scale for all the layers, and going on past the layer's own part), which \"parallax DIR\" draws; and the picture the model was asked to draw from (asked-N.png) and the picture behind each layer (behind-N.png)."`
	Hint  string `placeholder:"TEXT" help:"Words added to the request to draw what a layer hid: \"no people\", \"a calm sea\"."`
	Seed  uint64 `default:"1" help:"The seed of the pictures drawn (0: from the time)."`
	Tries int    `default:"3" help:"How many times what a layer hid is drawn at most: it is drawn again, from another seed, when the model left the layer's shape there (as a flat colour or a shadow)."`
	Edge  int    `default:"-1" placeholder:"PIXELS" help:"How far a layer reaches past where the depth cuts it, so that it holds its own outline (default: a 200th of the picture's width)."`
	Clean int    `default:"-1" placeholder:"PIXELS" help:"How far past a layer's edge the layer behind takes the picture drawn instead of the picture before (default: a 100th of the picture's width). The picture before may hold there what the depth missed of the layer (the tips of grass, a strand of hair), which would stay behind when the layer moves. 0: only where the layer was, and the layers put back together are the picture but for 2 pixels about each layer."`
}

func (c *SplitCmd) Run(ctx context.Context, app *ai.App, p *parts) error {
	rt, err := app.Open(ctx, c.options()...)
	if err != nil {
		return err
	}
	defer rt.Close()
	img, near, at, err := c.read(ctx, rt, p)
	if err != nil {
		return err
	}
	w, h := img.Rect.Dx(), img.Rect.Dy()
	edge := c.Edge
	if edge < 0 {
		edge = max(1, w/200)
	}
	margin := max(4, w/64) // how far past the layer the picture is drawn again
	clean := c.Clean       // and how far past it that picture is taken
	if clean < 0 {
		clean = max(2, w/100)
	}
	clean = min(clean, margin)

	// from the nearest layer to the one before the farthest: cut it out of the picture, draw
	// what it hid, and go on with that picture
	plate := img
	layers := make([]Layer, len(at)+1)
	nears := make([][]float32, len(at)+1) // how near each layer's points are, past its own part too
	for l := len(at); l >= 1; l-- {
		in := above(curve(near, c.Far), at[l-1])
		if covered(in) == 0 {
			fmt.Fprintf(os.Stderr, "layer %d: nothing is nearer than %.2f\n", l+1, at[l-1])
			continue
		}
		layer := soften(grow(in, w, h, edge), w, h, 1)
		layers[l] = Layer{Name: name(l, len(at)+1), Img: cutOut(plate, layer)}
		nears[l] = extend(near, in, w, h)

		hole := grow(in, w, h, edge+margin)
		fmt.Fprintf(os.Stderr, "layer %d: %.0f %% of the picture\n", l+1, 100*covered(in))
		drawn, err := c.fill(ctx, rt, p, plate, hole, l+1)
		if err != nil {
			return err
		}
		// the picture drawn is taken where the layer was and --clean pixels past it, the picture
		// before elsewhere: thin things of the layer that the depth missed (the tips of grass, a
		// strand of hair) would otherwise stay in the layer behind and show when the layer moves
		plate = mix(plate, drawn, soften(grow(in, w, h, edge+clean), w, h, max(clean/2, 1)))
		if c.PNG != "" {
			if err := writePNG(filepath.Join(c.PNG, fmt.Sprintf("behind-%d.png", l+1)), plate); err != nil {
				return err
			}
		}
		// how near the points drawn are: a second estimate, brought to the first one's scale
		// from the part both saw, and behind the layer cut out
		again, err := c.near(ctx, rt, p, plate)
		if err != nil {
			return err
		}
		a, b := fit(again, near, hole)
		limit := uncurve(at[l-1], c.Far)
		merged := make([]float32, len(near))
		for i := range near {
			merged[i] = near[i]
			if hole[i] > 0 {
				merged[i] = min(max(float32(a*float64(again[i])+b), 0), limit)
			}
		}
		near = merged
	}
	layers[0] = Layer{Name: name(0, len(at)+1), Img: cutOut(plate, above(near, -1))}
	nears[0] = near
	kept, keptNears := layers[:0], nears[:0]
	for i, l := range layers {
		if l.Img != nil {
			kept, keptNears = append(kept, l), append(keptNears, nears[i])
		}
	}

	file := c.Out
	if file == "" {
		file = strings.TrimSuffix(c.Picture, filepath.Ext(c.Picture)) + ".psd"
	}
	f, err := os.Create(file)
	if err != nil {
		return err
	}
	if err := WritePSD(f, img, kept); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if c.PNG != "" {
		// the layers and how near their points are: what "parallax DIR" draws
		for i, l := range kept {
			if err := writePNG(filepath.Join(c.PNG, fmt.Sprintf("layer-%d.png", i+1)), l.Img); err != nil {
				return err
			}
			depth := image.NewGray16(l.Img.Rect)
			for k, n := range keptNears[i] {
				v := uint16(min(max(n, 0), 1)*65535 + 0.5)
				depth.Pix[2*k], depth.Pix[2*k+1] = uint8(v>>8), uint8(v)
			}
			if err := writePNG(filepath.Join(c.PNG, fmt.Sprintf("near-%d.png", i+1)), depth); err != nil {
				return err
			}
		}
	}
	fmt.Printf("%s: %d layers of %d x %d\n", file, len(kept), w, h)
	return nil
}

// name is the name of layer l of n, from the farthest (0).
func name(l, n int) string {
	switch l {
	case 0:
		return fmt.Sprintf("%d far", l+1)
	case n - 1:
		return fmt.Sprintf("%d near", l+1)
	}
	return fmt.Sprint(l + 1)
}

// seamOK is the seam under which a part drawn again is taken as going on from its surroundings.
const seamOK = 50

// fill is the picture with the part of a mask drawn again as what is behind it: the part is
// painted in a plain colour the picture has little of, and the picture model is asked to take
// that shape away and draw the scenery there. The picture is stretched to a size the model
// draws and the answer back.
//
// The model sometimes leaves the shape, in its colour or as a shadow. That shows as a seam all
// along the mask's edge (seam), and the picture is then drawn again from another seed, up to
// --tries times; the one with the least seam is kept.
func (c *SplitCmd) fill(ctx context.Context, rt *ai.Runtime, p *parts, plate *image.RGBA, hole []float32, layer int) (*image.RGBA, error) {
	g, err := artist.Get(ctx, rt)
	if err != nil {
		return nil, hint(err)
	}
	w, h := plate.Rect.Dx(), plate.Rect.Dy()
	size := g.SizeFor(w, h)
	colour, word := key(plate)
	prompt := fmt.Sprintf("Remove the %s shape from image 1. Where it was, draw the scenery that is behind it, continuing the surroundings, so that no trace of the shape is left. Keep everything else unchanged.", word)
	if c.Hint != "" {
		prompt += " " + strings.TrimSuffix(c.Hint, ".") + "."
	}
	masked := paint(plate, hole, colour)
	if c.PNG != "" {
		if err := writePNG(filepath.Join(c.PNG, fmt.Sprintf("asked-%d.png", layer)), masked); err != nil {
			return nil, err
		}
	}
	req := imagegen.Request{Prompt: prompt, References: []image.Image{stretch(masked, size.X, size.Y)}}
	var best *image.RGBA
	least := math.Inf(1)
	for try := 0; try < max(c.Tries, 1); try++ {
		if c.Seed != 0 {
			p.drawing.Update(func(o *imagegen.Options) { o.Seed = c.Seed + uint64(try) })
		}
		t0 := time.Now()
		out, err := p.behind.Do(ctx, rt, req)
		if err != nil {
			return nil, hint(err)
		}
		drawn := stretch(out, w, h)
		s := seam(drawn, hole, w, h)
		verdict := ""
		if s >= seamOK {
			verdict = ": the shape is still there"
			if try+1 < c.Tries {
				verdict += ", drawing again"
			}
		}
		fmt.Fprintf(os.Stderr, "layer %d: what it hid drawn in %.0f s, seam %.0f%s\n", layer, time.Since(t0).Seconds(), s, verdict)
		if s < least {
			best, least = drawn, s
		}
		if s < seamOK {
			break
		}
	}
	if err := artist.Release(rt); err != nil { // its memory is given back before the depth is asked again
		return nil, err
	}
	return best, nil
}

func writePNG(file string, img image.Image) error {
	if err := os.MkdirAll(filepath.Dir(file), 0o755); err != nil {
		return err
	}
	f, err := os.Create(file)
	if err != nil {
		return err
	}
	if err := png.Encode(f, img); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// hint adds how to prepare the models to the error of one that is not ready.
func hint(err error) error {
	if strings.Contains(err.Error(), "not prepared") {
		return fmt.Errorf("%w (\"layers models prepare\" makes the models ready, or --prepare)", err)
	}
	return err
}
