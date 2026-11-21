// Package cutout tells the people of a picture from what is behind them with MediaPipe's Selfie
// Segmentation (Google, Apache-2.0; the ONNX files of onnx-community): how much of each pixel is
// a person, 0 to 1, at the model's own size (256 x 256, or 256 x 144 for the landscape model).
//
// It runs on the CPU with ONNX Runtime: 3 to 6 ms a picture on one thread (Ryzen AI 9 365),
// which is a camera's every frame. The graph is 0.46 MB of weights, far below what pays on an
// NPU that switches its array for each operator (YuNet: 16 times the energy on AMD's).
//
// The host stretches the picture to the model's size (the model's own preprocessing does, as
// MediaPipe's), gives it as RGB in 0..1 and takes the output as it is: the model ends in a
// sigmoid.
package cutout

import (
	"context"
	"fmt"
	"image"
	"image/color"
	"time"

	"github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	"github.com/shibukawa/hwnpu/internal/onnx"
	"github.com/shibukawa/hwnpu/internal/onnxrt"
	"github.com/shibukawa/hwnpu/internal/vision/moge"
)

// Model is a loaded model on the CPU. It runs one picture at a time.
type Model struct {
	// W and H are the size of the model's input and of its output.
	W, H int
	// Logf, when set, is told what is done.
	Logf func(format string, args ...any)

	s *onnxrt.Session
	x []float32 // the input (1, 3, H, W), kept between pictures
}

// OpenCPU loads the ONNX file (onnx/model.onnx of onnx-community/mediapipe_selfie_segmentation
// or of its landscape model) into ONNX Runtime on threads threads of the CPU (0: the calling
// thread). The size is the graph's input's.
func OpenCPU(path string, threads int) (*Model, error) {
	g, err := onnx.Load(path)
	if err != nil {
		return nil, err
	}
	if len(g.Inputs) != 1 || len(g.Outputs) != 1 || len(g.Inputs[0].Shape) != 4 {
		return nil, fmt.Errorf("cutout: %s is not a selfie segmentation model (one picture in, one mask out)", path)
	}
	sh := g.Inputs[0].Shape
	if sh[1] != 3 || sh[2] <= 0 || sh[3] <= 0 {
		return nil, fmt.Errorf("cutout: %s takes %v, not a picture of a fixed size", path, sh)
	}
	rt, err := onnxrt.Load()
	if err != nil {
		return nil, err
	}
	b, err := onnxrt.ReadModel(path)
	if err != nil {
		return nil, err
	}
	s, err := rt.Open(b, onnxrt.Options{Threads: threads})
	if err != nil {
		return nil, err
	}
	return &Model{W: sh[3], H: sh[2], s: s, x: make([]float32, 3*sh[2]*sh[3])}, nil
}

// Close releases the session.
func (m *Model) Close() error { return m.s.Close() }

// Alpha is how much of each pixel of img, stretched to W x H, is a person: 0 (what is behind)
// to 1, row major.
func (m *Model) Alpha(ctx context.Context, img image.Image) ([]float32, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	t0 := time.Now()
	Planes(m.x, img, m.W, m.H)
	t1 := time.Now()
	out, err := m.s.Run(map[string]onnxrt.Tensor{m.s.Inputs[0]: {Shape: []int64{1, 3, int64(m.H), int64(m.W)}, Data: m.x}})
	if err != nil {
		return nil, fmt.Errorf("cutout: %w", err)
	}
	a := out[m.s.Outputs[0]].Data
	if len(a) != m.W*m.H {
		return nil, fmt.Errorf("cutout: the model gave %d values for %dx%d", len(a), m.W, m.H)
	}
	if m.Logf != nil {
		b := img.Bounds()
		m.Logf("cutout: %dx%d at %dx%d on the cpu, resize %.1f ms, network %.1f ms", b.Dx(), b.Dy(), m.W, m.H,
			t1.Sub(t0).Seconds()*1000, time.Since(t1).Seconds()*1000)
	}
	return a, nil
}

// Planes writes img, stretched to w x h, into x as (3, h, w) planes of R, G and B in 0..1.
//
// A picture at least as large as w x h (a camera's frame) is averaged over the pixels each
// output pixel covers, with whole pixels: the mean of a box, which costs one pass over the
// picture and no memory. A smaller one is scaled with the antialiased bilinear filter of the
// model's own preprocessing (PIL's).
func Planes(x []float32, img image.Image, w, h int) {
	b := img.Bounds()
	sw, sh := b.Dx(), b.Dy()
	if sw < w || sh < h {
		px := lfm3vl.Resize(img, w, h, "bilinear")
		for i := 0; i < w*h; i++ {
			x[i], x[w*h+i], x[2*w*h+i] = float32(px[3*i])/255, float32(px[3*i+1])/255, float32(px[3*i+2])/255
		}
		return
	}
	var (
		pix    []uint8 // 4 bytes a pixel, R G B first
		stride int
	)
	switch p := img.(type) {
	case *image.RGBA: // alpha-premultiplied: a camera's frames are opaque
		pix, stride = p.Pix[p.PixOffset(b.Min.X, b.Min.Y):], p.Stride
	case *image.NRGBA:
		pix, stride = p.Pix[p.PixOffset(b.Min.X, b.Min.Y):], p.Stride
	default:
		rgb := lfm3vl.RGB(img)
		box(x, w, h, sw, sh, func(o int) (uint8, uint8, uint8) { return rgb[3*o], rgb[3*o+1], rgb[3*o+2] })
		return
	}
	// the columns' bounds once
	x0 := make([]int, w+1)
	for i := range x0 {
		x0[i] = i * sw / w
	}
	for oy := 0; oy < h; oy++ {
		ya, yb := oy*sh/h, (oy+1)*sh/h
		for ox := 0; ox < w; ox++ {
			var r, g, bl uint32
			for y := ya; y < yb; y++ {
				row := pix[y*stride+4*x0[ox] : y*stride+4*x0[ox+1]]
				for i := 0; i+3 < len(row); i += 4 {
					r += uint32(row[i])
					g += uint32(row[i+1])
					bl += uint32(row[i+2])
				}
			}
			k := 1 / (255 * float32((yb-ya)*(x0[ox+1]-x0[ox])))
			o := oy*w + ox
			x[o], x[w*h+o], x[2*w*h+o] = float32(r)*k, float32(g)*k, float32(bl)*k
		}
	}
}

// box is Planes' mean of boxes for pixels read through at (their place in raster order).
func box(x []float32, w, h, sw, sh int, at func(o int) (r, g, b uint8)) {
	for oy := 0; oy < h; oy++ {
		ya, yb := oy*sh/h, (oy+1)*sh/h
		for ox := 0; ox < w; ox++ {
			xa, xb := ox*sw/w, (ox+1)*sw/w
			var r, g, bl uint32
			for y := ya; y < yb; y++ {
				for i := xa; i < xb; i++ {
					pr, pg, pb := at(y*sw + i)
					r, g, bl = r+uint32(pr), g+uint32(pg), bl+uint32(pb)
				}
			}
			k := 1 / (255 * float32((yb-ya)*(xb-xa)))
			o := oy*w + ox
			x[o], x[w*h+o], x[2*w*h+o] = float32(r)*k, float32(g)*k, float32(bl)*k
		}
	}
}

// Stretch is the matte a (w x h, row major) at W x H, between its points in straight lines.
func Stretch(a []float32, w, h, W, H int) []float32 {
	if w == W && h == H {
		return a
	}
	return moge.Bilinear(a, w, h, W, H)
}

// Cut is img with what is behind the people made clear: the picture's own colors, with the
// matte a (w x h) stretched over it as its alpha.
func Cut(img image.Image, a []float32, w, h int) *image.NRGBA {
	b := img.Bounds()
	W, H := b.Dx(), b.Dy()
	out := image.NewNRGBA(image.Rect(0, 0, W, H))
	a = Stretch(a, w, h, W, H)
	for y := 0; y < H; y++ {
		for x := 0; x < W; x++ {
			c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
			c.A = uint8(float32(c.A)*min(max(a[y*W+x], 0), 1) + 0.5)
			out.SetNRGBA(x, y, c)
		}
	}
	return out
}

// Share is the part of the matte that is a person, 0 to 1: the mean of a.
func Share(a []float32) float64 {
	var s float64
	for _, v := range a {
		s += float64(v)
	}
	return s / float64(max(1, len(a)))
}
