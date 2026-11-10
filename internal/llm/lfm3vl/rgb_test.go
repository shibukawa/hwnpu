package lfm3vl

import (
	"bytes"
	"image"
	"image/color"
	"math"
	"math/rand/v2"
	"testing"
)

// TestRGBFastPaths: the fast paths give what the generic conversion does.
func TestRGBFastPaths(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	rect := image.Rect(3, 5, 40, 29)
	nrgba := image.NewNRGBA(rect)
	rgba := image.NewRGBA(rect)
	ycc := image.NewYCbCr(rect, image.YCbCrSubsampleRatio420)
	for i := range nrgba.Pix {
		nrgba.Pix[i] = uint8(r.IntN(256))
	}
	for i := range rgba.Pix {
		rgba.Pix[i] = uint8(r.IntN(256))
		if i%4 == 3 {
			rgba.Pix[i] = 255
		}
	}
	for _, p := range [][]uint8{ycc.Y, ycc.Cb, ycc.Cr} {
		for i := range p {
			p[i] = uint8(r.IntN(256))
		}
	}
	for _, img := range []image.Image{nrgba, rgba, ycc, nrgba.SubImage(image.Rect(10, 8, 30, 20)), ycc.SubImage(image.Rect(10, 8, 31, 21))} {
		got := RGB(img)
		b := img.Bounds()
		for y := 0; y < b.Dy(); y++ {
			for x := 0; x < b.Dx(); x++ {
				c := color.NRGBAModel.Convert(img.At(b.Min.X+x, b.Min.Y+y)).(color.NRGBA)
				i := (y*b.Dx() + x) * 3
				if got[i] != c.R || got[i+1] != c.G || got[i+2] != c.B {
					t.Fatalf("%T at (%d, %d): %v, want %v", img, x, y, got[i:i+3], c)
				}
			}
		}
	}
}

// resizeRef is ResizeRGB as first written (a pass at a time with closures per pixel): the faster
// one must give the same bytes.
func resizeRef(src []uint8, sw, sh, w, h int, filter string) []uint8 {
	support, kernel := 1.0, func(x float64) float64 { return math.Max(0, 1-math.Abs(x)) }
	if filter == "bicubic" {
		const a = -0.5
		support = 2
		kernel = func(x float64) float64 {
			x = math.Abs(x)
			switch {
			case x < 1:
				return ((a+2)*x-(a+3))*x*x + 1
			case x < 2:
				return (((x-5)*x+8)*x - 4) * a
			}
			return 0
		}
	}
	pass := func(data []uint8, in, out, n int, at, put func(line, i int) int) []uint8 {
		dst := make([]uint8, n*out*3)
		scale := float64(in) / float64(out)
		fs := math.Max(scale, 1)
		sup := support * fs
		for o := 0; o < out; o++ {
			center := (float64(o) + 0.5) * scale
			lo := max(int(center-sup+0.5), 0)
			hi := min(int(center+sup+0.5), in)
			ws := make([]float64, hi-lo)
			var sum float64
			for j := range ws {
				ws[j] = kernel((float64(j+lo) - center + 0.5) / fs)
				sum += ws[j]
			}
			for line := 0; line < n; line++ {
				for c := 0; c < 3; c++ {
					var acc float64
					for j, wt := range ws {
						acc += wt * float64(data[at(line, lo+j)+c])
					}
					if sum != 0 {
						acc /= sum
					}
					dst[put(line, o)+c] = uint8(math.Max(0, math.Min(255, math.Round(acc))))
				}
			}
		}
		return dst
	}
	tmp := pass(src, sw, w, sh, func(y, x int) int { return (y*sw + x) * 3 }, func(y, x int) int { return (y*w + x) * 3 })
	return pass(tmp, sh, h, w, func(x, y int) int { return (y*w + x) * 3 }, func(x, y int) int { return (y*w + x) * 3 })
}

func TestResizeRGBSameBytes(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for _, c := range []struct{ sw, sh, w, h int }{{800, 533, 640, 426}, {37, 29, 640, 640}, {640, 480, 48, 1000}, {1000, 40, 333, 17}, {64, 64, 64, 32}} {
		src := make([]uint8, c.sw*c.sh*3)
		for i := range src {
			src[i] = uint8(r.IntN(256))
		}
		for _, f := range []string{"bilinear", "bicubic"} {
			got, want := ResizeRGB(src, c.sw, c.sh, c.w, c.h, f), resizeRef(src, c.sw, c.sh, c.w, c.h, f)
			if !bytes.Equal(got, want) {
				t.Errorf("%v %s: the bytes differ", c, f)
			}
		}
	}
}
