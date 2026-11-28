package vision

import (
	"context"
	"errors"
	"image"
	"image/color"
	"io"
	"time"

	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/ai/vision/internal/bridge"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/airpc"
	"github.com/shibukawa/hwnpu/internal/catalog"
)

// Segment is the use of a model that cuts out the thing a prompt picks in a picture
// (EfficientSAM, as Segment Anything) for an app: a model, a family or a use of the definitions
// ("segment", "efficientsam:s").
func Segment(name string) ai.Use[*Segmenter] {
	return ai.Define[*Segmenter](segmentRole, name)
}

// The definition of a segmentation model may give the settings threads and the device
// (common.go). The image encoder runs on AMD's NPU from an export of
// tools/amd/iron_efficientsam.py (made with IRON on the first run), on Apple's and Intel's NPU
// converted from its ONNX file (at its first load), elsewhere on the CPU (ONNX Runtime where its
// library is here, else hwnpu's evaluator: slow); the mask decoder on the host. segmentRole's
// Prepare and Load are the engine's (ai/vision/engine).
var segmentRole = &airole.Role{
	Remote: func(h *airpc.Handle) io.Closer { return &Segmenter{p: airpc.RemoteProxy[segmenterImpl](h)} },
	Name:   "segment",
	Kinds:  []catalog.Kind{catalog.KindSegment},
}

// Prompt picks the thing to cut out of a picture, in its pixels: points on it, points off it, a
// box around it, or any of them together.
type Prompt struct {
	Points, Negative []Point
	Box              *Box
}

// empty reports whether the prompt has no points and no box.
func (p Prompt) empty() bool { return len(p.Points) == 0 && len(p.Negative) == 0 && p.Box == nil }

// PictureEncoding is a picture as the model's image encoder read it: what any prompt's masks
// are decoded from (Segmenter.Masks), so that a picture is encoded once for several prompts.
type PictureEncoding struct {
	// local is the engine's encoding (ai/vision/engine), of a Segmenter in this process
	local any
	// remote is the encoding hwnpud holds, of a Segmenter in hwnpud (airpc.RegisterRef): it
	// stays there, and is let go when this one is garbage collected
	remote *airpc.Handle
	// Took is how long encoding it took: Resize the picture's resizing, Encode the encoder's.
	Took, Resize, Encode time.Duration
}

// Mask is where the thing a prompt picks is in a picture: In true on it, W x H (the picture's
// size), row major. Score is the IoU the decoder predicts for it.
type Mask struct {
	W, H  int
	In    []bool
	Score float64
}

// Area is the mask's pixels.
func (k *Mask) Area() int {
	n := 0
	for _, v := range k.In {
		if v {
			n++
		}
	}
	return n
}

// Bounds is the mask's bounding box (empty for an empty mask).
func (k *Mask) Bounds() image.Rectangle {
	x1, y1, x2, y2 := k.W, k.H, -1, -1
	for y := range k.H {
		for x := range k.W {
			if k.In[y*k.W+x] {
				x1, y1, x2, y2 = min(x1, x), min(y1, y), max(x2, x), max(y2, y)
			}
		}
	}
	if x2 < 0 {
		return image.Rectangle{}
	}
	return image.Rect(x1, y1, x2+1, y2+1)
}

// Image is the mask as a picture: white on the thing, black off it.
func (k *Mask) Image() *image.Gray {
	img := image.NewGray(image.Rect(0, 0, k.W, k.H))
	for i, v := range k.In {
		if v {
			img.Pix[i] = 255
		}
	}
	return img
}

// Cutout is the picture the mask is of, transparent off the thing.
func (k *Mask) Cutout(img image.Image) *image.NRGBA {
	b := img.Bounds()
	cut := image.NewNRGBA(image.Rect(0, 0, k.W, k.H))
	for y := range k.H {
		for x := range k.W {
			if !k.In[y*k.W+x] {
				continue
			}
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			cut.SetNRGBA(x, y, color.NRGBA{uint8(r >> 8), uint8(g >> 8), uint8(bl >> 8), 255})
		}
	}
	return cut
}

// Masks are a prompt's masks, the decoder's candidates from the whole thing to a part, and how
// long decoding them took.
type Masks struct {
	Masks []Mask
	Took  time.Duration
}

// Best is the index of the mask the decoder scores highest.
func (m *Masks) Best() int {
	j := 0
	for i, k := range m.Masks {
		if k.Score > m.Masks[j].Score {
			j = i
		}
	}
	return j
}

// Segmenter cuts out the things prompts pick in pictures. It runs one picture at a time; others
// wait.
type Segmenter struct {
	p airpc.Proxy[segmenterImpl]
}

// segmenterImpl is a segmentation model of the engine in this process.
type segmenterImpl interface {
	Model() string
	Device() string
	Where() string
	// Encode is the picture's encoding: its times, and the engine's own (PictureEncoding.local).
	Encode(ctx context.Context, img image.Image) (*PictureEncoding, any, error)
	// Masks decodes the prompt's masks (not empty) on the engine's encoding of a picture.
	Masks(encoding any, p Prompt) (*Masks, error)
	Close() error
}

// Model is the name of the model in the definitions.
func (s *Segmenter) Model() string { return airpc.ProxyValue(s.p, "Model", segmenterImpl.Model) }

// Device is where the image encoder runs: npu or cpu.
func (s *Segmenter) Device() string { return airpc.ProxyValue(s.p, "Device", segmenterImpl.Device) }

// Where says where the image encoder runs, for a log: "AMD's NPU", "the CPU (ONNX Runtime)", ...
func (s *Segmenter) Where() string { return airpc.ProxyValue(s.p, "Where", segmenterImpl.Where) }

// Encode runs the image encoder on a picture.
func (s *Segmenter) Encode(ctx context.Context, img image.Image) (*PictureEncoding, error) {
	return airpc.ProxyCall1(s.p, ctx, "Encode", func(l segmenterImpl) (*PictureEncoding, error) {
		e, local, err := l.Encode(ctx, img)
		if err != nil {
			return nil, err
		}
		e.local = local
		return e, nil
	}, img)
}

// Masks decodes a prompt's masks on a picture's encoding.
func (s *Segmenter) Masks(e *PictureEncoding, p Prompt) (*Masks, error) {
	if p.empty() {
		return nil, errors.New("vision: a prompt of no points and no box")
	}
	return airpc.ProxyCall1(s.p, context.Background(), "Masks", func(l segmenterImpl) (*Masks, error) {
		return l.Masks(e.local, p)
	}, e, p)
}

// Segment is Encode and Masks: the masks of a prompt on a picture.
func (s *Segmenter) Segment(ctx context.Context, img image.Image, p Prompt) (*Masks, error) {
	e, err := s.Encode(ctx, img)
	if err != nil {
		return nil, err
	}
	return s.Masks(e, p)
}

// Close unloads the model. An app leaves it to the Runtime (its Close, or the use's Release).
func (s *Segmenter) Close() error { return s.p.Close() }

func init() {
	airole.Register(segmentRole)
	bridge.Roles[segmentRole.Name] = segmentRole
	bridge.Wrap[segmentRole.Name] = func(l any) io.Closer { return &Segmenter{p: airpc.LocalProxy(l.(segmenterImpl))} }
	// a picture's encoding stays in hwnpud, where its masks are decoded
	airpc.RegisterRef(func(e *PictureEncoding, h *airpc.Handle) { e.remote = h }, func(e *PictureEncoding) *airpc.Handle { return e.remote })
}
