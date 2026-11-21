package vision

import (
	"context"
	"fmt"
	"image"
	"io"
	"sync"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/flow"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/vision/cutout"
)

// Cutout is the use of a model that tells the people of a picture from what is behind them
// (MediaPipe's Selfie Segmentation) for an app: a model, a family or a use of the definitions
// ("cutout").
func Cutout(name string) ai.Use[*Cutter] {
	return ai.Define[*Cutter](cutoutRole, name)
}

// cutoutONNX is the file of the model's repository.
const cutoutONNX = "onnx/model.onnx"

var cutoutRole = &airole.Role{
	Name:  "cutout",
	Kinds: []chat.Kind{chat.KindCutout},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		// It runs on the CPU of every machine, whatever HWNPU_DEVICE asks of the other models:
		// it is 0.46 MB of weights, a few milliseconds a picture on one thread, and no NPU has a
		// program of it (models.yaml's device).
		path, err := chat.Fetch(e.Ctx, e.Model.Source(), cutoutONNX, e.Log)
		if err != nil {
			return f, err
		}
		f.Set("onnx", path)
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		m, err := cutout.OpenCPU(f.Path("onnx"), 1)
		if err != nil {
			return nil, fmt.Errorf("%s runs on the CPU with ONNX Runtime: %w", e.Model, err)
		}
		return &Cutter{m: m, name: e.Model}, nil
	},
}

// Matte is how much of each pixel of a picture is a person.
type Matte struct {
	// W and H are the matte's size: that of the model's input the picture was stretched to
	// (256 x 256, or 256 x 144), not the picture's (Alpha and AlphaImage give it at any size).
	W, H int
	// A is how much of each point is a person, row major: 1 a person, 0 what is behind, and
	// between them at a person's edge.
	A []float32
}

// Alpha is the matte at w x h (a picture's size), row major.
func (m *Matte) Alpha(w, h int) []float32 { return cutout.Stretch(m.A, m.W, m.H, w, h) }

// AlphaImage is Alpha as a picture of w x h: what a window's shader takes as a texture, at the
// matte's own size (W, H) for the least work (the shader stretches it).
func (m *Matte) AlphaImage(w, h int) *image.Alpha {
	img := image.NewAlpha(image.Rect(0, 0, w, h))
	for i, v := range m.Alpha(w, h) {
		img.Pix[i] = uint8(min(max(v, 0), 1)*255 + 0.5)
	}
	return img
}

// Cut is img with what is behind the people made clear: the picture's own colors, with the
// matte as its alpha (PNG keeps it).
func (m *Matte) Cut(img image.Image) *image.NRGBA { return cutout.Cut(img, m.A, m.W, m.H) }

// Smooth moves the matte toward next by keep's share of itself (0: next as it is; 0.5 halves a
// model's flicker between a camera's frames at the cost of a frame's delay at a moving edge)
// and returns it. A nil or differently sized matte becomes a copy of next.
func (m *Matte) Smooth(next *Matte, keep float32) *Matte {
	if m == nil || m.W != next.W || m.H != next.H {
		return &Matte{W: next.W, H: next.H, A: append([]float32(nil), next.A...)}
	}
	for i, v := range next.A {
		m.A[i] = m.A[i]*keep + v*(1-keep)
	}
	return m
}

// Cutter tells the people of pictures from what is behind them. It runs one picture at a time;
// others wait.
type Cutter struct {
	mu   sync.Mutex
	m    *cutout.Model
	name string
}

// Model is the name of the model in the definitions.
func (c *Cutter) Model() string { return c.name }

// Size is the size of the model's mattes (Matte.W, Matte.H).
func (c *Cutter) Size() image.Point { return image.Pt(c.m.W, c.m.H) }

// Matte is the matte of a picture: the picture is stretched to the model's size, so one of
// another shape than the model's gives a matte of its own shape all the same.
func (c *Cutter) Matte(ctx context.Context, img image.Image) (*Matte, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	a, err := c.m.Alpha(ctx, img)
	if err != nil {
		return nil, err
	}
	return &Matte{W: c.m.W, H: c.m.H, A: a}, nil
}

// Matted is a picture with its matte: a frame of a camera and the people in that very frame.
type Matted struct {
	Picture image.Image
	Matte   *Matte
}

// MatteStream is the matte of each picture of a stream (a camera's frames) with the picture, in
// a chunk of the picture's times. A stream whose pictures come faster than they are read is to
// be read through flow.Latest, or over an edge of flow.Newest.
func (c *Cutter) MatteStream(ctx context.Context, s flow.Stream[image.Image]) flow.Stream[Matted] {
	return flow.Map(s, func(ch flow.Chunk[image.Image]) (Matted, error) {
		m, err := c.Matte(ctx, ch.Data)
		return Matted{Picture: ch.Data, Matte: m}, err
	})
}

// CutoutNode is the node of a cutter in a network or a call (flow, ai.App.From, ai.NewCall):
// pictures in, each with its matte out.
func CutoutNode(u ai.Use[*Cutter]) *flow.Node[image.Image, Matted] {
	return ai.Node(u, func(ctx context.Context, c *Cutter, in flow.Stream[image.Image]) flow.Stream[Matted] {
		return c.MatteStream(ctx, in)
	})
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (c *Cutter) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.m.Close()
}
