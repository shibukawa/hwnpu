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
// Drawing from a reference runs on AMD's NPU and Apple's Neural Engine.
package main

import (
	"context"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
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
	"github.com/shibukawa/hwnpu/samples/internal/layering"
)

// The models: the depth of a picture, and the one that draws what a layer hid.
var (
	depth = vision.Depth("depth")
	// Edit, not Generate: it draws from a reference (on Apple's Neural Engine with programs of
	// their own, which Generate leaves out)
	artist = imagegen.Edit("image")
)

// build is the app with its two calls (samples/internal/layering runs them in turns).
func build() (*ai.App, *layering.Calls) {
	app := ai.NewApp("layers")
	return app, layering.Register(app, depth, artist)
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
}

// read is the picture and the nearness of its pixels. The depth model is unloaded once it has
// answered.
func (c *picture) read(ctx context.Context, rt *ai.Runtime, p *layering.Calls) (*image.RGBA, []float32, error) {
	if len(c.Cuts) == 0 && (c.Layers < 2 || c.Layers > 8) {
		return nil, nil, fmt.Errorf("--layers %d: 2 to 8", c.Layers)
	}
	f, err := os.Open(c.Picture)
	if err != nil {
		return nil, nil, err
	}
	src, _, err := image.Decode(f)
	f.Close()
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", c.Picture, err)
	}
	img := layering.Stretch(src, src.Bounds().Dx(), src.Bounds().Dy())
	t0 := time.Now()
	near, model, err := p.Near(ctx, rt, img)
	if err != nil {
		return nil, nil, hint(err)
	}
	fmt.Fprintf(os.Stderr, "depth by %s in %.1f s\n", model, time.Since(t0).Seconds())
	return img, near, nil
}

// cuts are where the picture is cut: --cuts, else found from the nearness.
func (c *picture) cuts(near []float32) []float32 {
	if len(c.Cuts) > 0 {
		return c.Cuts
	}
	return layering.Cuts(near, c.Layers, c.Far)
}

// PlanCmd is layers plan PICTURE.
type PlanCmd struct {
	picture
	Out string `short:"o" type:"path" placeholder:"FILE" help:"The PNG to write (default: PICTURE's name with .layers.png)."`
}

func (c *PlanCmd) Run(ctx context.Context, app *ai.App, p *layering.Calls) error {
	rt, err := app.Open(ctx, konghelper.AutoPrepare())
	if err != nil {
		return err
	}
	defer rt.Close()
	img, near, err := c.read(ctx, rt, p)
	if err != nil {
		return err
	}
	at := c.cuts(near)
	// each layer in a tint of its own, from the farthest: blue, green, yellow, red, ...
	tints := []color.RGBA{{40, 90, 255, 255}, {40, 200, 120, 255}, {250, 210, 40, 255}, {255, 90, 40, 255},
		{220, 40, 200, 255}, {40, 220, 230, 255}, {150, 255, 60, 255}, {255, 255, 255, 255}}
	n := len(at) + 1
	share := make([]int, n)
	out := image.NewRGBA(img.Rect)
	for i, x := range layering.Curve(near, c.Far) {
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
	PNG   string `name:"png" type:"path" placeholder:"DIR" help:"Also write into this directory each layer as a PNG with its transparency (layer-N.png, 1 the farthest) and how near its points are (near-N.png: 16 bit gray, white the nearest, on one scale for all the layers, and going on past the layer's own part), which \"parallax DIR\" draws; and the picture the model was asked to draw from (asked-N.png) and what it drew at each try (drawnT-N.png)."`
	Hint  string `placeholder:"TEXT" help:"Words added to the request to draw what a layer hid: \"no people\", \"a calm sea\"."`
	Seed  uint64 `default:"1" help:"The seed of the pictures drawn (0: from the time)."`
	Tries int    `default:"3" help:"How many times what a layer hid is drawn at most: it is drawn again, from another seed, when the model left the layer's shape there (as a flat colour or a shadow)."`
	Edge  int    `default:"-1" placeholder:"PIXELS" help:"How far a layer reaches past where the depth cuts it, so that it holds its own outline (default: a 200th of the picture's width)."`
	Clean int    `default:"0" placeholder:"PIXELS" help:"How far past a layer's edge the layer behind takes the picture drawn instead of the picture before. 0: only where the layer was, so that the layers put back together are the picture. The picture before may hold past the edge what the depth missed of the layer (the tips of grass, a strand of hair), which stays behind when the layer moves; some pixels here (10 of a picture 1024 wide) take that away, but show as a rim about the layer where the picture drawn does not go on from its surroundings exactly."`
}

func (c *SplitCmd) Run(ctx context.Context, app *ai.App, p *layering.Calls) error {
	rt, err := app.Open(ctx, konghelper.AutoPrepare())
	if err != nil {
		return err
	}
	defer rt.Close()
	img, near, err := c.read(ctx, rt, p)
	if err != nil {
		return err
	}
	// each step is told on a line; the pictures of the steps go into --png's directory
	var failed error
	t0 := time.Now()
	o := layering.Options{Layers: c.Layers, Cuts: c.Cuts, Far: c.Far, Edge: c.Edge, Clean: c.Clean, Hint: c.Hint, Seed: c.Seed, Tries: c.Tries}
	o.Progress = func(s layering.Step) {
		save := func(name string) {
			if c.PNG != "" && failed == nil {
				failed = writePNG(filepath.Join(c.PNG, fmt.Sprintf("%s-%d.png", name, s.Layer)), s.Picture)
			}
		}
		switch s.Kind {
		case layering.Cut:
			fmt.Fprintf(os.Stderr, "layer %d of %d: %.0f %% of the picture\n", s.Layer, s.Of, 100*s.Share)
		case layering.Kept:
			fmt.Fprintf(os.Stderr, "layer %d stays in the picture: what is behind it was not drawn well in any try; %d layers\n", s.Layer, len(s.Layers))
		case layering.Asked:
			save("asked")
			t0 = time.Now()
		case layering.Drawn:
			verdict := ""
			if s.Seam >= layering.SeamOK {
				verdict = ": the shape is still there"
				if s.Try < max(c.Tries, 1) {
					verdict += ", drawing again"
				}
			}
			fmt.Fprintf(os.Stderr, "layer %d: what it hid drawn in %.0f s, seam %.0f%s (%.0f %% of the whole)\n", s.Layer, time.Since(t0).Seconds(), s.Seam, verdict, 100*s.Total)
			save(fmt.Sprintf("drawn%d", s.Try))
			t0 = time.Now()
		}
	}
	layers, err := layering.Split(ctx, img, near, p.Models(rt), o)
	if err != nil {
		return hint(err)
	}
	if failed != nil {
		return failed
	}
	if err := p.Done(rt); err != nil {
		return err
	}

	file := c.Out
	if file == "" {
		file = strings.TrimSuffix(c.Picture, filepath.Ext(c.Picture)) + ".psd"
	}
	named := make([]Layer, len(layers))
	for i, l := range layers {
		named[i] = Layer{Name: name(i, len(layers)), Img: l.Img}
	}
	f, err := os.Create(file)
	if err != nil {
		return err
	}
	if err := WritePSD(f, img, named); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if c.PNG != "" {
		// the layers and how near their points are: what "parallax DIR" draws
		if err := layering.Write(c.PNG, layers); err != nil {
			return err
		}
	}
	fmt.Printf("%s: %d layers of %d x %d\n", file, len(layers), img.Rect.Dx(), img.Rect.Dy())
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
		return fmt.Errorf("%w (\"layers models prepare\" makes the models ready)", err)
	}
	return err
}
