package lfm3vl

import (
	"image"
	"image/color"
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
