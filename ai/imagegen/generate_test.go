package imagegen

import (
	"image"
	"testing"
)

func TestSizeFor(t *testing.T) {
	g := &Generator{sizes: []image.Point{{1024, 1024}, {512, 512}, {832, 1248}, {1248, 832}, {1360, 752}}}
	for _, c := range []struct{ w, h, wantW, wantH int }{
		{0, 0, 1024, 1024}, // nothing asked: the first
		{1024, 1024, 1024, 1024},
		{2000, 2000, 1024, 1024}, // a square: the larger one for a large picture
		{400, 400, 512, 512},     // and the smaller for a small one
		{1920, 1080, 1360, 752},
		{3000, 2000, 1248, 832},
		{600, 900, 832, 1248},
		{1100, 1000, 1024, 1024}, // the nearest shape
	} {
		if got := g.SizeFor(c.w, c.h); got != image.Pt(c.wantW, c.wantH) {
			t.Errorf("SizeFor(%d, %d) = %v, want %d x %d", c.w, c.h, got, c.wantW, c.wantH)
		}
	}
}
