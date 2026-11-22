package imagegen

import (
	"image"
	"slices"
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

// TestEditSize: an edit's picture is as big as its biggest reference (flux2klein.EditPatterns).
func TestEditSize(t *testing.T) {
	big, small := image.NewRGBA(image.Rect(0, 0, 1024, 900)), image.NewRGBA(image.Rect(0, 0, 500, 512))
	for _, c := range []struct {
		refs []image.Image
		size image.Point
		divs []int
	}{
		{[]image.Image{big}, image.Pt(1024, 1024), []int{1}},
		{[]image.Image{small}, image.Pt(512, 512), []int{1}},
		{[]image.Image{big, small}, image.Pt(1024, 1024), []int{1, 2}},
		{[]image.Image{small, small}, image.Pt(512, 512), []int{1, 1}},
		{[]image.Image{small, big, small}, image.Pt(1024, 1024), []int{2, 1, 2}},
	} {
		size, divs := editSize(c.refs)
		if size != c.size || !slices.Equal(divs, c.divs) {
			t.Errorf("%d references: %v %v, want %v %v", len(c.refs), size, divs, c.size, c.divs)
		}
	}
}
