package vision

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/shibukawa/hwnpu"
	"github.com/shibukawa/hwnpu/ai"
	"github.com/shibukawa/hwnpu/internal/airole"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/vision/segment"
)

// Segment is the use of a model that cuts out the thing a prompt picks in a picture
// (EfficientSAM, as Segment Anything) for an app: a model, a family or a use of the definitions
// ("segment", "efficientsam:s").
func Segment(name string) ai.Use[*Segmenter] {
	return ai.Define[*Segmenter](segmentRole, name)
}

// segmentSetup is what segment downloads for model m: its encoder's and decoder's ONNX files,
// which models.yaml's sha256 names.
func segmentSetup(m chat.Model, log io.Writer) (segment.Setup, error) {
	s := segment.Setup{Dir: chat.ModelDir(m), Repo: m.Source(), SHA256: m.SHA256, Log: log}
	for f := range m.SHA256 {
		switch {
		case strings.HasSuffix(f, "_encoder.onnx"):
			s.Encoder = f
		case strings.HasSuffix(f, "_decoder.onnx"):
			s.Decoder = f
		}
	}
	if s.Encoder == "" || s.Decoder == "" {
		return s, fmt.Errorf("%s: models.yaml names no *_encoder.onnx and *_decoder.onnx in its sha256", shown(m.Name))
	}
	return s, nil
}

// The definition of a segmentation model may give the settings threads and the device
// (common.go). The image encoder runs on AMD's NPU from an export of
// tools/amd/iron_efficientsam.py (made with IRON on the first run), on Apple's and Intel's NPU
// converted from its ONNX file (at its first load), elsewhere on the CPU (ONNX Runtime where its
// library is here, else hwnpu's evaluator: slow); the mask decoder on the host.
var segmentRole = &airole.Role{
	Name:  "segment",
	Kinds: []chat.Kind{chat.KindSegment},
	Prepare: func(e airole.PrepareEnv) (f airole.Files, err error) {
		m := e.Model
		s, err := segmentSetup(m, e.Log)
		if err != nil {
			return f, err
		}
		flag := imagedev.DeviceFlag(m)
		fetchEnc := func() (string, error) { enc, _, err := segment.Fetch(e.Ctx, s); return enc, err }
		have := func(dir string) error { _, err := os.Stat(filepath.Join(dir, "model.json")); return err }
		export := imagedev.AMDFirstRunExport(e.Ctx, e.Base, flag, false, logf(e.Log), e.Log, have, "iron_efficientsam.py", segment.ExportVersion, fetchEnc,
			func(python []string, model, dir string) error {
				return segment.ExportNPU(e.Ctx, python, e.Base.Name, model, dir, e.Log)
			})
		enc, dec, err := segment.Fetch(e.Ctx, s)
		if err != nil {
			return f, err
		}
		f.Set("encoder", enc)
		f.Set("decoder", dec)
		switch {
		case export != "":
			f.Set("export", export)
		case imagedev.ProgramDevice(m, e.NPU.Vendor, flag):
			// Intel's NPU runs the encoder converted from its ONNX file to OpenVINO IR, Apple's
			// Neural Engine the same as Core AI's program (M3: 0.4 s where hwnpu's evaluator
			// takes 33 s, the mask's IoU with the CPU's 0.9994; docs/design/vision-models.md),
			// converted into the model's directory at its first load
			dir := chat.ModelDir(e.Base)
			if err := os.MkdirAll(dir, 0o755); err != nil {
				return f, err
			}
			f.Set("programs", dir)
		}
		return f, nil
	},
	Load: func(ctx context.Context, f airole.Files, e airole.LoadEnv) (io.Closer, error) {
		var (
			sm  *segment.Model
			dev *hwnpu.Device
			err error
		)
		device := "npu"
		switch {
		case f.Path("export") != "":
			if dev, err = hwnpu.Open(ctx, hwnpu.WithVendor(hwnpu.VendorAMD)); err != nil {
				return nil, err
			}
			sm, err = segment.OpenNPU(ctx, dev, f.Path("export"), f.Path("decoder"))
		case f.Path("programs") != "":
			var backend string
			var opts []hwnpu.Option
			if e.NPU.Vendor == "apple" {
				opts = append(opts, hwnpu.WithPlacement(hwnpu.PlacementStrict))
			}
			if dev, backend, err = imagedev.Open(ctx, e.NPU.Device, e.Cache, opts...); err != nil {
				return nil, err
			}
			sm, err = segment.OpenProgram(ctx, dev, backend, f.Path("encoder"), f.Path("decoder"), f.Path("programs"), e.Log)
		default:
			device = "cpu"
			sm, err = segment.OpenCPU(f.Path("encoder"), f.Path("decoder"), threads(e.Def))
		}
		if err != nil {
			if dev != nil {
				dev.Close()
			}
			return nil, err
		}
		return &Segmenter{m: sm, dev: dev, name: e.Model, device: device}, nil
	},
}

// Prompt picks the thing to cut out of a picture, in its pixels: points on it, points off it, a
// box around it, or any of them together.
type Prompt struct {
	Points, Negative []Point
	Box              *Box
}

// points is the prompt as the decoder reads it: the points on the thing, those off it, then the
// box's corners.
func (p Prompt) points() []segment.Point {
	var out []segment.Point
	for _, q := range p.Points {
		out = append(out, segment.Point{X: q.X, Y: q.Y, Label: 1})
	}
	for _, q := range p.Negative {
		out = append(out, segment.Point{X: q.X, Y: q.Y, Label: 0})
	}
	if p.Box != nil {
		out = append(out, segment.Point{X: p.Box.Min.X, Y: p.Box.Min.Y, Label: 2}, segment.Point{X: p.Box.Max.X, Y: p.Box.Max.Y, Label: 3})
	}
	return out
}

// PictureEncoding is a picture as the model's image encoder read it: what any prompt's masks
// are decoded from (Segmenter.Masks), so that a picture is encoded once for several prompts.
type PictureEncoding struct {
	e *segment.Embedding
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
	mu     sync.Mutex
	m      *segment.Model
	dev    *hwnpu.Device
	name   string
	device string
}

// Model is the name of the model in the definitions.
func (s *Segmenter) Model() string { return s.name }

// Device is where the image encoder runs: npu or cpu.
func (s *Segmenter) Device() string { return s.device }

// Where says where the image encoder runs, for a log: "AMD's NPU", "the CPU (ONNX Runtime)", ...
func (s *Segmenter) Where() string { return s.m.Where() }

// Encode runs the image encoder on a picture.
func (s *Segmenter) Encode(ctx context.Context, img image.Image) (*PictureEncoding, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	t0 := time.Now()
	e, err := s.m.Embed(ctx, img)
	if err != nil {
		return nil, err
	}
	return &PictureEncoding{e: e, Took: time.Since(t0), Resize: s.m.Times[0], Encode: s.m.Times[1]}, nil
}

// Masks decodes a prompt's masks on a picture's encoding.
func (s *Segmenter) Masks(e *PictureEncoding, p Prompt) (*Masks, error) {
	pts := p.points()
	if len(pts) == 0 {
		return nil, errors.New("vision: a prompt of no points and no box")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	t0 := time.Now()
	ks, err := s.m.Segment(e.e, pts)
	if err != nil {
		return nil, err
	}
	out := &Masks{Took: time.Since(t0)}
	for _, k := range ks {
		out.Masks = append(out.Masks, Mask{W: k.W, H: k.H, In: k.In, Score: k.Score})
	}
	return out, nil
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
func (s *Segmenter) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	err := s.m.Close()
	if s.dev != nil {
		s.dev.Close()
	}
	return err
}
