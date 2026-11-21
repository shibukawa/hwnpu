package cutout

import (
	"context"
	"image"
	"image/color"
	"math"
	"os"
	"testing"

	"github.com/shibukawa/hwnpu/internal/onnxrt"
)

// TestPlanes: a picture larger than the model's size is the mean of each output pixel's box, the
// same whichever kind of picture holds it.
func TestPlanes(t *testing.T) {
	// 4 x 2: the left half red 100 and 200, the right half blue 50 everywhere
	rgba := image.NewRGBA(image.Rect(0, 0, 4, 2))
	for y := 0; y < 2; y++ {
		rgba.SetRGBA(0, y, color.RGBA{100, 0, 0, 255})
		rgba.SetRGBA(1, y, color.RGBA{200, 0, 0, 255})
		rgba.SetRGBA(2, y, color.RGBA{0, 0, 50, 255})
		rgba.SetRGBA(3, y, color.RGBA{0, 0, 50, 255})
	}
	want := []float32{150.0 / 255, 0, 0, 0, 0, 50.0 / 255} // R, G, B planes of 2 x 1
	check := func(name string, img image.Image) {
		t.Helper()
		x := make([]float32, 6)
		Planes(x, img, 2, 1)
		for i := range want {
			if math.Abs(float64(x[i]-want[i])) > 1e-6 {
				t.Errorf("%s: %v, want %v", name, x, want)
				return
			}
		}
	}
	check("RGBA", rgba)
	nrgba := image.NewNRGBA(rgba.Bounds())
	gray := image.NewRGBA64(rgba.Bounds()) // a kind without a fast path
	for y := 0; y < 2; y++ {
		for x := 0; x < 4; x++ {
			nrgba.Set(x, y, rgba.At(x, y))
			gray.Set(x, y, rgba.At(x, y))
		}
	}
	check("NRGBA", nrgba)
	check("RGBA64", gray)
	// a part of a larger picture: its own pixels only
	big := image.NewRGBA(image.Rect(0, 0, 8, 6))
	for i := range big.Pix {
		big.Pix[i] = 255
	}
	for y := 0; y < 2; y++ {
		for x := 0; x < 4; x++ {
			big.Set(2+x, 3+y, rgba.At(x, y))
		}
	}
	check("a part", big.SubImage(image.Rect(2, 3, 6, 5)))

	// a smaller picture is scaled up: a flat color stays the color
	small := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for i := 0; i < 4; i++ {
		small.SetRGBA(i%2, i/2, color.RGBA{51, 102, 204, 255})
	}
	x := make([]float32, 3*8*8)
	Planes(x, small, 8, 8)
	for i := 0; i < 64; i++ {
		if math.Abs(float64(x[i]-0.2)) > 1e-3 || math.Abs(float64(x[64+i]-0.4)) > 1e-3 || math.Abs(float64(x[128+i]-0.8)) > 1e-3 {
			t.Fatalf("a small picture's pixel %d: %v %v %v", i, x[i], x[64+i], x[128+i])
		}
	}
}

func TestCut(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 4, 2))
	for i := 0; i < 8; i++ {
		img.SetRGBA(i%4, i/4, color.RGBA{10, 20, 30, 255})
	}
	// a matte of 2 x 1: the left a person, the right not
	out := Cut(img, []float32{1, 0}, 2, 1)
	if c := out.NRGBAAt(0, 0); c != (color.NRGBA{10, 20, 30, 255}) {
		t.Errorf("a person's pixel: %v", c)
	}
	if c := out.NRGBAAt(3, 1); c != (color.NRGBA{10, 20, 30, 0}) {
		t.Errorf("a pixel behind: %v", c)
	}
	if a := out.NRGBAAt(1, 0).A; a <= out.NRGBAAt(2, 0).A || a == 255 {
		t.Errorf("the edge is not between: %d %d", a, out.NRGBAAt(2, 0).A)
	}
	if s := Share([]float32{1, 0, 0.5, 0.5}); s != 0.5 {
		t.Errorf("Share = %v", s)
	}
}

// TestModel runs the model of HWNPU_CUTOUT_ONNX (onnx/model.onnx of
// onnx-community/mediapipe_selfie_segmentation) on a flat picture: a matte of its size, every
// value a share.
func TestModel(t *testing.T) {
	path := os.Getenv("HWNPU_CUTOUT_ONNX")
	if path == "" || !onnxrt.Available() {
		t.Skip("HWNPU_CUTOUT_ONNX names no model, or ONNX Runtime is not here")
	}
	m, err := OpenCPU(path, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	img := image.NewRGBA(image.Rect(0, 0, 640, 480))
	for i := range img.Pix {
		img.Pix[i] = 128
	}
	a, err := m.Alpha(context.Background(), img)
	if err != nil {
		t.Fatal(err)
	}
	if len(a) != m.W*m.H {
		t.Fatalf("%d values for %dx%d", len(a), m.W, m.H)
	}
	for i, v := range a {
		if !(v >= 0 && v <= 1) {
			t.Fatalf("value %d is %v", i, v)
		}
	}
	if s := Share(a); s > 0.2 {
		t.Errorf("a flat gray picture is %.0f %% a person", s*100)
	}
}
