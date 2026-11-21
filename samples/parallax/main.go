// parallax is a sample app of the ai package: it shows a picture as if it were deep. The
// picture's depth is estimated once, the viewer's head is followed with the camera, and the
// picture is drawn from where the head is, a far point moved a little and a near one more
// (Ebitengine, a Kage shader), so that it looks like a view through a window.
//
// It joins its models in the two ways an app does (ai/flow):
//
//	calls, in turns:            the picture → depth; a painted picture → what is behind  (Depth Anything 3, FLUX.2 klein)
//	a network, kept running:    camera → faces → head → steady → view  (YuNet, and nodes of the program)
//
// and the window draws on its own beat, 60 times a second, with whatever view the network last
// gave. The window shows the picture as soon as its depth is there, and beside it the picture is
// parted into layers by its depth, what each layer hid drawn in (samples/internal/layering): the window
// shows each step and how far the work is, and takes the layers as they come, so that where a
// near thing moves aside there is a picture behind it. The camera is read through hwcamera (github.com/shibukawa/hwcamera).
//
//	parallax models prepare                    # download, convert and compile the models ahead
//	parallax picture.png                       # the window, following the head with the camera; the picture comes apart into 3 layers as it is shown
//	parallax picture.png --layers 1            # with its depth alone, no layers
//	parallax picture.png --head pointer        # the pointer stands in for the head (no camera)
//	parallax picture.png --head sweep          # the view goes around by itself
//	parallax render picture.png -o deep.gif    # no window: the picture from a head that goes around
//	parallax parts/                            # a picture in layers (samples/layers --png parts/): what a near layer hid shows
//	parallax track --frames a.png,b.png        # no camera: follow the head in pictures
//	parallax cameras                           # the cameras, numbered as --device takes them
//
// Pictures with clear layers work best: a figure before a landscape, things at several depths.
// The window is built with Ebitengine; a machine without a display or its libraries builds the
// rest with -tags headless.
package main

import (
	"context"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"os"
	"os/signal"
	"strings"
	"time"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/cli/konghelper"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/imagegen"
	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/samples/internal/capture"
	"github.com/shibukawa/hwnpu/samples/internal/layering"
)

// The models: the depth of the picture, the one that draws what a layer of the picture hid, and
// the faces before the camera. The first two run in turns and are unloaded when they are done,
// so the app picks the definitions' models for the machine.
var (
	depth  = vision.Depth("depth")
	artist = imagegen.Edit("image") // it draws from a reference
	faces  = vision.Faces("face-detect")
)

// parts are the app's network and calls, which its commands run.
type parts struct {
	// split are the calls that part the picture into layers: its depth, and what a layer hid
	split   *layering.Calls
	tracker *tracker
}

func build() (*ai.App, *parts) {
	app := ai.NewApp("parallax")
	p := &parts{}
	p.split = layering.Register(app, depth, artist)
	p.tracker = newTracker(app, faces, vision.HeadOptions{})
	return app, p
}

var cmds struct {
	Show    ShowCmd           `cmd:"" default:"withargs" help:"Show a picture in a window as if it were deep: it turns with the viewer's head."`
	Render  RenderCmd         `cmd:"" help:"Write the picture as seen from a head that goes around, as an animated GIF: no window and no camera."`
	Track   TrackCmd          `cmd:"" help:"Follow a head and print where it is: the camera's part alone, from the camera or from pictures."`
	Cameras CamerasCmd        `cmd:"" help:"The cameras, numbered as --device takes them."`
	Models  konghelper.Models `cmd:"" help:"The models parallax uses and its network: what they take, download them, compile them for this machine's NPU."`
	Status  konghelper.Status `cmd:"" help:"This machine's NPU: the device, how busy it is and what uses it."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	app, p := build()
	k := kong.Parse(&cmds,
		kong.Name("parallax"),
		kong.Description("Show a picture as if it were deep, turning with the viewer's head: a sample of hwnpu's ai package."),
		kong.UsageOnError(),
		konghelper.Bind(ctx, app),
		kong.Bind(p))
	k.FatalIfErrorf(k.Run())
}

// picture is what the commands that draw share: the picture, its depth and how deep it looks.
type picture struct {
	Picture    string  `arg:"" type:"path" name:"PICTURE" help:"The picture (PNG or JPEG), or a directory of a picture in layers, each with its own nearness (layer-N.png and near-N.png, 1 the farthest: what \"layers PICTURE --png DIR\" of samples/layers writes). In layers, what a near layer hid is seen where it moves aside, instead of its edge stretched; no depth is estimated then."`
	Near       string  `type:"existingfile" placeholder:"FILE" help:"The picture's nearness as a gray picture of its size (white the nearest), instead of estimating it: --save-near's file."`
	SaveNear   string  `name:"save-near" type:"path" placeholder:"FILE" help:"Write the nearness estimated as a gray PNG."`
	Strength   float64 `default:"0.03" help:"How far the nearest point moves when the head is at the edge of the view, as a share of the picture's width."`
	Focus      float64 `default:"0" help:"The nearness that stays where it is: 0 moves the far points least and the near ones most, 0.5 turns the picture about its middle."`
	Far        float64 `default:"30" help:"How far the far end of the depth is pulled apart: 0 moves each point as its depth says (a thing twice as far moves half as much, so that behind something right before the eye all the rest moves as one); larger gives the far things more of the move, so that a bridge, the houses behind it and the mountains move apart."`
	Layers     int     `default:"3" help:"The layers the picture is parted into by its depth, with what each layer hid drawn in by the picture model, so that where a near thing moves aside there is a picture behind it instead of its edge stretched. The window shows the picture at once with its depth alone, and the parting goes on beside it (about a minute for each layer but the last), each step shown. 1: no layers. Where the picture model is not ready, the picture stays as it is."`
	Hint       string  `placeholder:"TEXT" help:"Words added to the request to draw what a layer hid: \"no people\", \"a calm sea\"."`
	Seed       uint64  `default:"1" help:"The seed of the pictures drawn (0: from the time)."`
	SaveLayers string  `name:"save-layers" type:"path" placeholder:"DIR" help:"Write the layers and their nearness into this directory once the picture is parted; \"parallax DIR\" shows them again without the models."`
	Edge       int     `default:"-1" placeholder:"PIXELS" help:"How far the near things are made larger in the nearness, so that their outlines move with them (default: a 200th of the picture's width; 0: not at all)."`
	Prepare    bool    `help:"Download, convert and compile the models that are not ready instead of failing."`
}

func (c *picture) params() Params { return Params{Strength: c.Strength, Focus: c.Focus} }

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

func readPicture(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return img, nil
}

// scene is the picture with its nearness: read from --near, else estimated by the app's call,
// once, after which the depth model is unloaded (the tracker's is the only one that stays). It
// also returns what the scene was made from, to part into layers; nil for a picture that is in
// layers already (a directory).
func (c *picture) scene(ctx context.Context, rt *ai.Runtime, p *parts) (*Scene, *source, error) {
	if st, err := os.Stat(c.Picture); err == nil && st.IsDir() {
		layers, err := layering.Read(c.Picture)
		if err != nil {
			return nil, nil, fmt.Errorf("%w (\"layers PICTURE --png DIR\" of samples/layers, or --save-layers, writes the layers)", err)
		}
		fmt.Fprintf(os.Stderr, "%d layers of %d x %d\n", len(layers), layers[0].Img.Rect.Dx(), layers[0].Img.Rect.Dy())
		return stack(layers, c.edge(layers[0].Img.Rect.Dx()), c.Far), nil, nil
	}
	img, err := c.image()
	if err != nil {
		return nil, nil, err
	}
	near, err := c.near(ctx, rt, p, img)
	if err != nil {
		return nil, nil, err
	}
	return NewScene(img, near, c.edge(img.Rect.Dx()), c.Far), &source{img: img, near: near}, nil
}

// image is the picture, as RGBA.
func (c *picture) image() (*image.RGBA, error) {
	pic, err := readPicture(c.Picture)
	if err != nil {
		return nil, err
	}
	return layering.Stretch(pic, pic.Bounds().Dx(), pic.Bounds().Dy()), nil
}

// near is how near each point of the picture is: read from --near, else estimated by the app's
// call, once, after which the depth model is unloaded.
func (c *picture) near(ctx context.Context, rt *ai.Runtime, p *parts, img *image.RGBA) ([]float32, error) {
	b := img.Bounds()
	var (
		near []float32
		err  error
	)
	if c.Near != "" {
		n, err := readPicture(c.Near)
		if err != nil {
			return nil, err
		}
		if n.Bounds().Dx() != b.Dx() || n.Bounds().Dy() != b.Dy() {
			return nil, fmt.Errorf("%s is %d x %d, the picture %d x %d", c.Near, n.Bounds().Dx(), n.Bounds().Dy(), b.Dx(), b.Dy())
		}
		near = make([]float32, 0, b.Dx()*b.Dy())
		for y := n.Bounds().Min.Y; y < n.Bounds().Max.Y; y++ {
			for x := n.Bounds().Min.X; x < n.Bounds().Max.X; x++ {
				r, g, bl, _ := n.At(x, y).RGBA()
				near = append(near, float32(r+g+bl)/(3*0xffff))
			}
		}
	} else {
		t0 := time.Now()
		var model string
		if near, model, err = p.split.Near(ctx, rt, img); err != nil {
			return nil, hint(err)
		}
		fmt.Fprintf(os.Stderr, "the picture's depth by %s in %.1f s (the model's loading included)\n", model, time.Since(t0).Seconds())
		if c.SaveNear != "" {
			gray := image.NewGray(b)
			for i, v := range near {
				gray.Pix[i] = uint8(v*255 + 0.5)
			}
			f, err := os.Create(c.SaveNear)
			if err != nil {
				return nil, err
			}
			if err := png.Encode(f, gray); err != nil {
				f.Close()
				return nil, err
			}
			if err := f.Close(); err != nil {
				return nil, err
			}
		}
	}
	return near, nil
}

// edge is --edge for a picture of width w.
func (c *picture) edge(w int) int {
	if c.Edge >= 0 {
		return c.Edge
	}
	return max(1, w/200)
}

// camera is what the commands that follow a head share.
type camera struct {
	Device string   `placeholder:"N" help:"The camera: its number in \"parallax cameras\" (or its ID); none: the system's default camera."`
	Frames []string `placeholder:"FILE,..." help:"Pictures given as the camera's frames, 15 a second, again and again: the head is followed in them (for a machine without a camera)."`
	FOV    float64  `name:"fov" default:"60" help:"The camera's horizontal field of view in degrees, for how far the head is."`
	Gain   float64  `default:"2.5" help:"How far the picture turns for the head's move: 2.5 turns it fully for a head 40% of the way to the edge of the camera's view."`
}

// frames is the stream of the camera, or of the pictures that stand in for it (once when once,
// else for as long as ctx lasts).
func (c *camera) frames(ctx context.Context, once bool) (flow.Stream[image.Image], error) {
	if len(c.Frames) == 0 {
		return capture.Camera(ctx, capture.CameraOptions{Device: c.Device}), nil
	}
	var imgs []image.Image
	for _, path := range c.Frames {
		img, err := readPicture(path)
		if err != nil {
			return nil, err
		}
		imgs = append(imgs, img)
	}
	const period = time.Second / 15
	recorded := flow.Stream[image.Image](func(yield func(flow.Chunk[image.Image], error) bool) {
		for i := 0; once && i < len(imgs) || !once && ctx.Err() == nil; i++ {
			if !yield(flow.Chunk[image.Image]{Data: imgs[i%len(imgs)], Start: time.Duration(i) * period, Len: period, Final: true}, nil) {
				return
			}
		}
	})
	if once {
		return recorded, nil
	}
	return flow.Live(ctx, recorded), nil // at a camera's pace
}

// ShowCmd is parallax PICTURE: the window.
type ShowCmd struct {
	picture
	camera
	Head   string        `enum:"camera,pointer,sweep" default:"camera" help:"What moves the view: the viewer's head before the camera, the pointer, or by itself (sweep)."`
	Dump   string        `type:"path" placeholder:"DIR" help:"Write the first --dump-frames frames the shader draws into this directory and end (to check the shader)."`
	DumpN  int           `name:"dump-frames" default:"8" help:"How many frames --dump writes."`
	Record string        `type:"path" placeholder:"DIR" help:"Write the window as a PNG into this directory every 2 seconds: how the picture came apart into layers, to look at after."`
	For    time.Duration `name:"for" placeholder:"TIME" help:"Close the window after this long (5m)."`
}

func (c *ShowCmd) Run(ctx context.Context, app *ai.App, p *parts) error {
	rt, err := app.Open(ctx, c.options()...)
	if err != nil {
		return err
	}
	defer rt.Close()
	o := windowOptions{title: "parallax: " + c.Picture, sweep: c.Head == "sweep", dump: c.Dump, quiet: c.Dump != ""}
	if c.Record != "" {
		o.record = c.Record
		if err := os.MkdirAll(c.Record, 0o755); err != nil {
			return err
		}
	}
	o.closeAfter = c.For
	var scene *Scene
	if st, err := os.Stat(c.Picture); (err == nil && st.IsDir()) || c.Dump != "" {
		// a picture in layers already, or the shader's frames to check: no making to show
		if scene, _, err = c.scene(ctx, rt, p); err != nil {
			return err
		}
	} else {
		// the window opens at once and shows the picture being made deep beside it: its depth,
		// each layer, and the layers put together
		img, err := c.image()
		if err != nil {
			return err
		}
		o.progress, o.picture = newProgress(img, c.Layers), img
		mctx, stop := context.WithCancel(ctx)
		made := make(chan struct{})
		go func() {
			defer close(made)
			c.make(mctx, rt, p, img, o.progress)
		}()
		defer func() { // before the models are unloaded
			stop()
			<-made
		}()
	}
	if c.Dump != "" {
		o.frames = c.DumpN
		if err := os.MkdirAll(c.Dump, 0o755); err != nil {
			return err
		}
	}
	if c.Head == "camera" {
		// the tracker's network runs beside the window, until the window is closed
		p.tracker.Gain = c.Gain
		o.camera, o.gain = p.tracker.Seen, p.tracker.AddGain
		frames, err := c.frames(ctx, false)
		if err != nil {
			return err
		}
		p.tracker.camera.From(frames)
		if _, err := faces.Get(ctx, rt); err != nil { // loaded before the window opens
			return hint(err)
		}
		tctx, cancel := context.WithCancel(ctx)
		done := make(chan error, 1)
		go func() { done <- rt.Run(tctx) }()
		defer func() {
			cancel()
			if err := <-done; err != nil && tctx.Err() == nil {
				fmt.Fprintln(os.Stderr, "the tracker:", err)
			}
		}()
		failed := ""
		o.target = func() (View, string) {
			select {
			case err := <-done: // the camera could not be read: the pointer takes over
				done <- err
				if err != nil {
					failed = err.Error()
				}
			default:
			}
			if failed != "" {
				return View{}, "no camera (" + failed + "): restart with --head pointer"
			}
			return p.tracker.Target()
		}
	}
	return runWindow(scene, c.params(), o)
}

// make makes the picture deep beside the window and tells prog of each step: its depth, then
// its layers (--layers), and last the scene, which the window puts together. A picture that is
// not parted into layers is shown with its depth alone.
func (c *ShowCmd) make(ctx context.Context, rt *ai.Runtime, p *parts, img *image.RGBA, prog *progress) {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	near, err := c.near(ctx, rt, p, img)
	if err != nil {
		if ctx.Err() == nil {
			fmt.Fprintln(os.Stderr, err)
			prog.show(stageFailed, nil, nil, "The depth was not estimated: "+err.Error())
		}
		return
	}
	prog.show(stageNear, depthColors(near, w, h), nil, "The depth: the near warm, the far cool")
	prog.set(func(p *progress) { p.total = p.wDepth })
	scene := NewScene(img, near, c.edge(w), c.Far)
	if c.Layers > 1 {
		layered, err := c.part(ctx, rt, p, &source{img: img, near: near}, prog, os.Stderr)
		switch {
		case ctx.Err() != nil:
			return
		case err != nil:
			fmt.Fprintln(os.Stderr, "not parted into layers:", err)
		default:
			scene = layered
		}
	}
	prog.ready(scene)
}

// RenderCmd is parallax render PICTURE: no window, an animated GIF drawn on the CPU.
type RenderCmd struct {
	picture
	Out    string `short:"o" default:"parallax.gif" type:"path" placeholder:"FILE" help:"The animated GIF to write."`
	Width  int    `default:"480" help:"The GIF's width (the picture is made smaller first)."`
	N      int    `name:"frames" default:"24" help:"The frames of one turn of the head."`
	PNGDir string `name:"png" type:"path" placeholder:"DIR" help:"Write the frames as PNG files too, at the picture's own size (to compare with the window's --dump)."`
}

func (c *RenderCmd) Run(ctx context.Context, app *ai.App, p *parts) error {
	rt, err := app.Open(ctx, c.options()...)
	if err != nil {
		return err
	}
	defer rt.Close()
	scene, src, err := c.scene(ctx, rt, p)
	if err != nil {
		return err
	}
	if src != nil && c.Layers > 1 {
		if scene, err = c.part(ctx, rt, p, src, nil, os.Stderr); err != nil {
			return err
		}
	}
	return render(scene, c.params(), c.Out, c.Width, c.N, c.PNGDir)
}

// TrackCmd is parallax track: the tracker's network alone.
type TrackCmd struct {
	camera
	Seconds time.Duration `placeholder:"TIME" help:"Stop after this long (10s); none: at Ctrl-C, or once the pictures of --frames were all read."`
	Prepare bool          `help:"Download, convert and compile the models that are not ready instead of failing."`
}

func (c *TrackCmd) Run(ctx context.Context, app *ai.App, p *parts) error {
	var opts []ai.Option
	if c.Prepare {
		opts = append(opts, ai.WithAutoPrepare(nil))
	}
	if c.Seconds > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, c.Seconds)
		defer cancel()
	}
	frames, err := c.frames(ctx, c.Seconds == 0 && len(c.Frames) > 0)
	if err != nil {
		return err
	}
	// each frame is counted as it is read, and the view printed when the tracker gives a new one
	read := 0
	p.tracker.Gain = c.Gain
	p.tracker.camera.From(flow.Map(frames, func(f flow.Chunk[image.Image]) (image.Image, error) {
		read++
		return f.Data, nil
	}))
	start := time.Now()
	last := -1
	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			case <-time.After(20 * time.Millisecond):
			}
			p.tracker.mu.Lock()
			n, v, h := p.tracker.found, p.tracker.target, p.tracker.head
			p.tracker.mu.Unlock()
			if n != last && n > 0 {
				last = n
				fmt.Printf("[%5.2fs] head %+.2f %+.2f, %.2f m away (%+.2f m right, %+.2f m up) -> view %+.2f %+.2f\n",
					time.Since(start).Seconds(), h.X, h.Y, h.Z, h.Right, h.Up, v.X, v.Y)
			}
		}
	}()
	err = app.Run(ctx, opts...)
	close(stop)
	p.tracker.mu.Lock()
	found := p.tracker.found
	p.tracker.mu.Unlock()
	fmt.Fprintf(os.Stderr, "%d frames read in %.1f s, a face found in %d\n", read, time.Since(start).Seconds(), found)
	if err != nil && ctx.Err() == nil {
		return hint(err)
	}
	return nil
}

// CamerasCmd is parallax cameras.
type CamerasCmd struct{}

func (CamerasCmd) Run(ctx context.Context) error {
	cams, err := capture.Cameras(ctx)
	if err != nil {
		return err
	}
	if len(cams) == 0 {
		fmt.Println("no cameras")
	}
	for i, c := range cams {
		fmt.Printf("%d  %s\n", i+1, c[1])
	}
	return nil
}

// hint adds how to prepare the models to the error of one that is not ready.
func hint(err error) error {
	if strings.Contains(err.Error(), "not prepared") {
		return fmt.Errorf("%w (\"parallax models prepare\" makes the models ready, or --prepare)", err)
	}
	return err
}
