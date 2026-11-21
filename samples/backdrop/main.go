// backdrop is a sample app of the ai package: a camera's picture with what is behind the person
// blurred, or a picture or a colour put there, as a video call does. The person is told from
// what is behind them in every frame (MediaPipe's Selfie Segmentation on the CPU, a few
// milliseconds a frame on one thread), and the window draws the frame with its matte
// (Ebitengine, Kage shaders).
//
// Its network is kept running beside the window (ai/flow):
//
//	camera → cutout (the newest frame only) → steady → view
//
// cutout is a model. steady is a node of the program: it smooths the matte between frames, which
// halves the model's flicker. view keeps the newest frame and matte for the window, which draws
// on its own beat (60 times a second). The camera is read through hwcamera
// (github.com/shibukawa/hwcamera).
//
//	backdrop models prepare                  # download the model ahead (0.5 MB)
//	backdrop                                 # the window: the camera, what is behind the person blurred
//	backdrop --behind beach.jpg              # a picture behind the person
//	backdrop --behind "#00aa44"              # a colour: a green screen
//	backdrop --frames a.png,b.png            # pictures in place of the camera (a machine without one)
//	backdrop cameras                         # the cameras, numbered: --device 2 takes the second
//
// In the window B goes through what is behind (blurred, the picture or the colour, the frame as
// it is), M shows the matte, up and down change the blur.
//
// The model looks for a person before a camera: a frame without one gives a matte of whatever
// stands out. The window is built with Ebitengine; a machine without a display or its libraries
// builds the rest with -tags headless.
package main

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/alecthomas/kong"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/cli/konghelper"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/samples/internal/capture"
)

// The model: the people of a frame apart from what is behind them.
var cut = vision.Cutout("cutout")

// studio is the app's network: it keeps running on the camera's frames and holds the newest
// frame with its matte for the window.
type studio struct {
	camera *flow.Src[image.Image]

	// Keep is the share of the matte before that stays in each frame's (0: the model's matte as
	// it is; 0.5 halves its flicker at the cost of a frame's delay at a moving edge).
	Keep float32

	mu     sync.Mutex
	frame  *image.RGBA
	matte  *vision.Matte
	seq    int       // the frames so far
	since  time.Time // of the second being counted
	count  int       // its frames
	persec int       // the frames of the second before
}

func newStudio(app *ai.App) *studio {
	s := &studio{camera: flow.Source[image.Image]("camera"), Keep: 0.5}
	var smooth *vision.Matte
	steady := flow.Each("steady", func(c flow.Chunk[vision.Matted], emit func(flow.Chunk[vision.Matted])) error {
		smooth = smooth.Smooth(c.Data.Matte, s.Keep)
		// the window reads it while the next frame is smoothed: a copy
		m := &vision.Matte{W: smooth.W, H: smooth.H, A: append([]float32(nil), smooth.A...)}
		emit(flow.Chunk[vision.Matted]{Data: vision.Matted{Picture: c.Data.Picture, Matte: m}, Start: c.Start, Len: c.Len, Final: true})
		return nil
	})
	view := flow.Sink[vision.Matted]("view").Do(func(c flow.Chunk[vision.Matted]) error {
		frame := rgba(c.Data.Picture)
		s.mu.Lock()
		defer s.mu.Unlock()
		s.frame, s.matte, s.seq = frame, c.Data.Matte, s.seq+1
		if now := time.Now(); now.Sub(s.since) >= time.Second {
			s.persec, s.count, s.since = s.count, 0, now
		}
		s.count++
		return nil
	})
	// the model reads the newest frame only: a frame that waited is of no use
	app.From(s.camera).To(vision.CutoutNode(cut), flow.Newest()).To(steady).To(view)
	return s
}

// Newest is the newest frame with its matte, which frame it is (0: none yet) and how many
// frames came in the last second.
func (s *studio) Newest() (frame *image.RGBA, matte *vision.Matte, seq, persec int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.frame, s.matte, s.seq, s.persec
}

// rgba is img as the window takes it: a camera's frame is one already.
func rgba(img image.Image) *image.RGBA {
	b := img.Bounds()
	if p, ok := img.(*image.RGBA); ok && b.Min == (image.Point{}) && p.Stride == 4*b.Dx() {
		return p
	}
	out := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(out, out.Bounds(), img, b.Min, draw.Src)
	return out
}

func build() (*ai.App, *studio) {
	app := ai.NewApp("backdrop")
	return app, newStudio(app)
}

var cmds struct {
	Show    ShowCmd           `cmd:"" default:"withargs" help:"Show the camera in a window with what is behind the person blurred, or a picture or a colour there."`
	Cameras CamerasCmd        `cmd:"" help:"The cameras --device takes."`
	Models  konghelper.Models `cmd:"" help:"The model backdrop uses and its network: what it takes, download it."`
	Status  konghelper.Status `cmd:"" help:"This machine's NPU: the device, how busy it is and what uses it."`
}

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	app, s := build()
	k := kong.Parse(&cmds,
		kong.Name("backdrop"),
		kong.Description("A camera's picture with what is behind the person blurred or replaced: a sample of hwnpu's ai package."),
		kong.UsageOnError(),
		konghelper.Bind(ctx, app),
		kong.Bind(s))
	k.FatalIfErrorf(k.Run())
}

// ShowCmd is backdrop: the window.
type ShowCmd struct {
	Behind  string        `default:"blur" placeholder:"blur|FILE|#RRGGBB" help:"What is put behind the person: blur (the frame itself, blurred), a picture (PNG or JPEG, cut to the frame's shape) or a colour."`
	Blur    float64       `default:"0.02" help:"How far the blur reaches, as a share of the frame's width."`
	Steady  float64       `default:"0.5" help:"The share of the matte before that stays in each frame's: 0 takes the model's matte as it is, more flickers less and follows a move later."`
	Matte   bool          `help:"Start with the matte shown instead of the frame (M in the window): white a person."`
	Mirror  bool          `default:"true" negatable:"" help:"Show the frame as a mirror does, the way a video call shows oneself."`
	Device  string        `placeholder:"N" help:"The camera: its number in \"backdrop cameras\" (or its ID); none: the system's default camera."`
	Size    string        `default:"1280x720" placeholder:"WxH" help:"The size asked of the camera (it gives its nearest mode)."`
	Frames  []string      `placeholder:"FILE,..." help:"Pictures given as the camera's frames, 15 a second, again and again (for a machine without a camera)."`
	Record  string        `type:"path" placeholder:"DIR" help:"Write the window as a PNG into this directory every second, to look at after."`
	For     time.Duration `name:"for" placeholder:"TIME" help:"Close the window after this long (30s)."`
	Prepare bool          `help:"Download the model when it is not here instead of failing."`
}

func (c *ShowCmd) Run(ctx context.Context, app *ai.App, s *studio) error {
	o := windowOptions{blur: c.Blur, mirror: c.Mirror, matte: c.Matte, record: c.Record, closeAfter: c.For, newest: s.Newest}
	switch {
	case c.Behind == "blur":
	case strings.HasPrefix(c.Behind, "#"):
		v, err := strconv.ParseUint(c.Behind[1:], 16, 32)
		if err != nil || len(c.Behind) != 7 {
			return fmt.Errorf("--behind %q: a colour is #RRGGBB", c.Behind)
		}
		o.colour = &color.RGBA{uint8(v >> 16), uint8(v >> 8), uint8(v), 255}
	default:
		img, err := readPicture(c.Behind)
		if err != nil {
			return fmt.Errorf("--behind: %w (blur, a picture's file or #RRGGBB)", err)
		}
		o.picture = img
	}
	if c.Record != "" {
		if err := os.MkdirAll(c.Record, 0o755); err != nil {
			return err
		}
	}
	var opts []ai.Option
	if c.Prepare {
		opts = append(opts, ai.WithAutoPrepare(func(p ai.Progress) {
			if p.Message != "" {
				fmt.Fprintf(os.Stderr, "  %s: %s\n", p.Model, p.Message)
			}
		}))
	}
	rt, err := app.Open(ctx, opts...)
	if err != nil {
		return hint(err)
	}
	defer rt.Close()
	frames, err := c.frames(ctx)
	if err != nil {
		return err
	}
	s.Keep = float32(min(max(c.Steady, 0), 0.95))
	s.camera.From(frames)
	model, err := cut.Get(ctx, rt) // loaded before the window opens
	if err != nil {
		return hint(err)
	}
	o.title = "backdrop: " + model.Model()
	// the network runs beside the window, until the window is closed
	nctx, cancel := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { done <- rt.Run(nctx) }()
	defer func() {
		cancel()
		if err := <-done; err != nil && nctx.Err() == nil {
			fmt.Fprintln(os.Stderr, "the camera's network:", err)
		}
	}()
	o.failed = func() string {
		select {
		case err := <-done: // the camera could not be read
			done <- err
			if err != nil {
				return err.Error()
			}
		default:
		}
		return ""
	}
	return runWindow(o)
}

// frames is the stream of the camera, or of the pictures that stand in for it, for as long as
// ctx lasts.
func (c *ShowCmd) frames(ctx context.Context) (flow.Stream[image.Image], error) {
	if len(c.Frames) == 0 {
		var w, h int
		if _, err := fmt.Sscanf(c.Size, "%dx%d", &w, &h); err != nil || w <= 0 || h <= 0 {
			return nil, fmt.Errorf("--size %q: not WxH", c.Size)
		}
		return capture.Camera(ctx, capture.CameraOptions{Device: c.Device, Width: w, Height: h}), nil
	}
	var imgs []image.Image
	for _, path := range c.Frames {
		img, err := readPicture(path)
		if err != nil {
			return nil, err
		}
		imgs = append(imgs, rgba(img)) // once, not for each frame
	}
	const period = time.Second / 15
	recorded := flow.Stream[image.Image](func(yield func(flow.Chunk[image.Image], error) bool) {
		for i := 0; ctx.Err() == nil; i++ {
			if !yield(flow.Chunk[image.Image]{Data: imgs[i%len(imgs)], Start: time.Duration(i) * period, Len: period, Final: true}, nil) {
				return
			}
		}
	})
	return flow.Live(ctx, recorded), nil // at a camera's pace
}

// windowOptions are what the window is told.
type windowOptions struct {
	title string
	// blur is how far the blur reaches, as a share of the frame's width
	blur float64
	// picture or colour is what is put behind the person in place of the blurred frame
	picture image.Image
	colour  *color.RGBA
	mirror  bool
	// matte starts the window with the matte shown
	matte  bool
	record string
	// closeAfter closes the window after this long (0: never)
	closeAfter time.Duration
	// newest is the newest frame and its matte (studio.Newest)
	newest func() (*image.RGBA, *vision.Matte, int, int)
	// failed is why the network ended, "" while it runs
	failed func() string
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

// CamerasCmd is backdrop cameras.
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

// hint adds how to prepare the model to the error of one that is not ready.
func hint(err error) error {
	if strings.Contains(err.Error(), "not prepared") {
		return fmt.Errorf("%w (\"backdrop models prepare\" downloads the model, or --prepare)", err)
	}
	return err
}
