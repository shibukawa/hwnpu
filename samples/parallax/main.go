// parallax is a sample app of the ai package: it shows a picture as if it were deep. The
// picture's depth is estimated once, the viewer's head is followed with the camera, and the
// picture is drawn from where the head is, a far point moved a little and a near one more
// (Ebitengine, a Kage shader), so that it looks like a view through a window.
//
// It joins its models in the two ways an app does (ai/flow):
//
//	a call, once:               the picture → depth                    (Depth Anything 3)
//	a network, kept running:    camera → faces → head → steady → view  (YuNet, and nodes of the program)
//
// and the window draws on its own beat, 60 times a second, with whatever view the network last
// gave. The camera is read through hwcamera (github.com/shibukawa/hwcamera).
//
//	parallax models prepare                    # download, convert and compile the models ahead
//	parallax picture.png                       # the window, following the head with the camera
//	parallax picture.png --head pointer        # the pointer stands in for the head (no camera)
//	parallax picture.png --head sweep          # the view goes around by itself
//	parallax render picture.png -o deep.gif    # no window: the picture from a head that goes around
//	parallax parts/                            # a picture in layers (samples/layers --png parts/): what a near layer hid shows
//	parallax track --frames a.png,b.png        # no camera: follow the head in pictures
//	parallax cameras                           # the cameras --device takes
//
// Pictures with clear layers work best: a figure before a landscape, things at several depths.
// The window is built with Ebitengine; a machine without a display or its libraries builds the
// rest with -tags headless.
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
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/samples/capture"
)

// The models: the depth of the picture, and the faces before the camera. The depth is asked
// for once and its model unloaded, so the app picks the definitions' model for the machine.
var (
	depth = vision.Depth("depth")
	faces = vision.Faces("face-detect")
)

// parts are the app's network and call, which its commands run.
type parts struct {
	depthOf *ai.Call[image.Image, *vision.DepthMap]
	tracker *tracker
}

func build() (*ai.App, *parts) {
	app := ai.NewApp("parallax")
	p := &parts{}
	p.depthOf = ai.NewCall[image.Image, *vision.DepthMap](app, "depth of the picture").Then(vision.DepthNode(depth))
	p.tracker = newTracker(app, faces, vision.HeadOptions{})
	return app, p
}

var cmds struct {
	Show    ShowCmd           `cmd:"" default:"withargs" help:"Show a picture in a window as if it were deep: it turns with the viewer's head."`
	Render  RenderCmd         `cmd:"" help:"Write the picture as seen from a head that goes around, as an animated GIF: no window and no camera."`
	Track   TrackCmd          `cmd:"" help:"Follow a head and print where it is: the camera's part alone, from the camera or from pictures."`
	Cameras CamerasCmd        `cmd:"" help:"The cameras --device takes."`
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
	Picture  string  `arg:"" type:"path" name:"PICTURE" help:"The picture (PNG or JPEG), or a directory of a picture in layers, each with its own nearness (layer-N.png and near-N.png, 1 the farthest: what \"layers PICTURE --png DIR\" of samples/layers writes). In layers, what a near layer hid is seen where it moves aside, instead of its edge stretched; no depth is estimated then."`
	Near     string  `type:"existingfile" placeholder:"FILE" help:"The picture's nearness as a gray picture of its size (white the nearest), instead of estimating it: --save-near's file."`
	SaveNear string  `name:"save-near" type:"path" placeholder:"FILE" help:"Write the nearness estimated as a gray PNG."`
	Strength float64 `default:"0.03" help:"How far the nearest point moves when the head is at the edge of the view, as a share of the picture's width."`
	Focus    float64 `default:"0" help:"The nearness that stays where it is: 0 moves the far points least and the near ones most, 0.5 turns the picture about its middle."`
	Layers   float64 `default:"30" help:"How far the far layers are pulled apart: 0 moves each point as its depth says (a thing twice as far moves half as much, so that behind something right before the eye all the rest moves as one); larger gives the far layers more of the move, so that a bridge, the houses behind it and the mountains move apart."`
	Edge     int     `default:"-1" placeholder:"PIXELS" help:"How far the near things are made larger in the nearness, so that their outlines move with them (default: a 200th of the picture's width; 0: not at all)."`
	Prepare  bool    `help:"Download, convert and compile the models that are not ready instead of failing."`
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
// once, after which the depth model is unloaded (the tracker's is the only one that stays).
func (c *picture) scene(ctx context.Context, rt *ai.Runtime, p *parts) (*Scene, error) {
	if st, err := os.Stat(c.Picture); err == nil && st.IsDir() {
		return c.layered(c.Picture)
	}
	img, err := readPicture(c.Picture)
	if err != nil {
		return nil, err
	}
	b := img.Bounds()
	if c.Near != "" {
		n, err := readPicture(c.Near)
		if err != nil {
			return nil, err
		}
		if n.Bounds().Dx() != b.Dx() || n.Bounds().Dy() != b.Dy() {
			return nil, fmt.Errorf("%s is %d x %d, the picture %d x %d", c.Near, n.Bounds().Dx(), n.Bounds().Dy(), b.Dx(), b.Dy())
		}
		near := make([]float32, 0, b.Dx()*b.Dy())
		for y := n.Bounds().Min.Y; y < n.Bounds().Max.Y; y++ {
			for x := n.Bounds().Min.X; x < n.Bounds().Max.X; x++ {
				r, g, bl, _ := n.At(x, y).RGBA()
				near = append(near, float32(r+g+bl)/(3*0xffff))
			}
		}
		return NewScene(img, near, c.edge(b.Dx()), c.Layers), nil
	}
	t0 := time.Now()
	d, err := p.depthOf.Do(ctx, rt, img)
	if err != nil {
		return nil, hint(err)
	}
	est, _ := depth.Get(ctx, rt)
	fmt.Fprintf(os.Stderr, "the picture's depth by %s in %.1f s (the model's loading included)\n", est.Model(), time.Since(t0).Seconds())
	if err := depth.Release(rt); err != nil { // asked once: its memory is given back
		return nil, err
	}
	if c.SaveNear != "" {
		f, err := os.Create(c.SaveNear)
		if err != nil {
			return nil, err
		}
		if err := png.Encode(f, d.NearImage(b.Dx(), b.Dy())); err != nil {
			f.Close()
			return nil, err
		}
		if err := f.Close(); err != nil {
			return nil, err
		}
	}
	return NewScene(img, d.Near(b.Dx(), b.Dy()), c.edge(b.Dx()), c.Layers), nil
}

// layered is the scene of a picture in layers in dir: layer-1.png (the farthest) and so on,
// each with near-N.png, how near its points are (gray; white the nearest), on one scale for all
// the layers.
func (c *picture) layered(dir string) (*Scene, error) {
	var top *Scene
	for n := 1; ; n++ {
		file := filepath.Join(dir, fmt.Sprintf("layer-%d.png", n))
		if _, err := os.Stat(file); err != nil {
			break
		}
		img, err := readPicture(file)
		if err != nil {
			return nil, err
		}
		depth, err := readPicture(filepath.Join(dir, fmt.Sprintf("near-%d.png", n)))
		if err != nil {
			return nil, fmt.Errorf("layer %d has no nearness: %w", n, err)
		}
		b := img.Bounds()
		if depth.Bounds().Dx() != b.Dx() || depth.Bounds().Dy() != b.Dy() || top != nil && (b.Dx() != top.W || b.Dy() != top.H) {
			return nil, fmt.Errorf("%s: the layers and their nearness are not of one size", dir)
		}
		near := make([]float32, 0, b.Dx()*b.Dy())
		for y := depth.Bounds().Min.Y; y < depth.Bounds().Max.Y; y++ {
			for x := depth.Bounds().Min.X; x < depth.Bounds().Max.X; x++ {
				near = append(near, float32(color.Gray16Model.Convert(depth.At(x, y)).(color.Gray16).Y)/0xffff)
			}
		}
		// the things of one layer still have outlines inside it (a flower before the grass)
		s := NewScene(img, near, c.edge(b.Dx()), c.Layers)
		s.Behind, top = top, s
	}
	if top == nil {
		return nil, fmt.Errorf("%s has no layer-1.png (\"layers PICTURE --png DIR\" of samples/layers writes the layers)", dir)
	}
	fmt.Fprintf(os.Stderr, "%d layers of %d x %d\n", len(top.layers()), top.W, top.H)
	return top, nil
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
	Device string   `placeholder:"ID" help:"The camera (parallax cameras lists them); none: the system's default camera."`
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
	Head  string `enum:"camera,pointer,sweep" default:"camera" help:"What moves the view: the viewer's head before the camera, the pointer, or by itself (sweep)."`
	Dump  string `type:"path" placeholder:"DIR" help:"Write the first --dump-frames frames the shader draws into this directory and end (to check the shader)."`
	DumpN int    `name:"dump-frames" default:"8" help:"How many frames --dump writes."`
}

func (c *ShowCmd) Run(ctx context.Context, app *ai.App, p *parts) error {
	rt, err := app.Open(ctx, c.options()...)
	if err != nil {
		return err
	}
	defer rt.Close()
	scene, err := c.scene(ctx, rt, p)
	if err != nil {
		return err
	}
	o := windowOptions{title: "parallax: " + c.Picture, sweep: c.Head == "sweep", dump: c.Dump, quiet: c.Dump != ""}
	if c.Dump != "" {
		o.frames = c.DumpN
		if err := os.MkdirAll(c.Dump, 0o755); err != nil {
			return err
		}
	}
	if c.Head == "camera" {
		// the tracker's network runs beside the window, until the window is closed
		p.tracker.Gain = c.Gain
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
	scene, err := c.scene(ctx, rt, p)
	if err != nil {
		return err
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
	for _, c := range cams {
		fmt.Printf("%s\t%s\n", c[0], c[1])
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
