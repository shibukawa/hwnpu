package depth

import (
	"image"
	"image/color"
	"math"
	"testing"
)

// TestQuantile checks Quantile against torch.quantile (linear): quantile([1..5, 10], 0.99)
// is 9.75, quantile([3, 1, 2], 0.5) is 2.
func TestQuantile(t *testing.T) {
	if q := Quantile([]float32{10, 1, 2, 3, 4, 5}, 0.99); math.Abs(float64(q)-9.75) > 1e-5 {
		t.Errorf("Quantile = %v, want 9.75", q)
	}
	if q := Quantile([]float32{3, 1, 2}, 0.5); q != 2 {
		t.Errorf("Quantile = %v, want 2", q)
	}
}

// TestSky checks that the sky takes the 99th percentile of the rest's depth.
func TestSky(t *testing.T) {
	r := &Result{Depth: make([]float32, 100), Sky: make([]float32, 100)}
	for i := range r.Depth {
		r.Depth[i] = float32(i)
		if i >= 80 {
			r.Sky[i] = 0.9
		}
	}
	r.fillSky()
	want := Quantile(func() []float32 {
		xs := make([]float32, 80)
		for i := range xs {
			xs[i] = float32(i)
		}
		return xs
	}(), 0.99)
	if r.Depth[90] != want || r.Depth[10] != 10 {
		t.Errorf("sky %v, ground %v; want %v, 10", r.Depth[90], r.Depth[10], want)
	}
}

// TestPatches checks the patch rows' order (channel, row, column) and the class token's row.
func TestPatches(t *testing.T) {
	e := &sized{Config: Config{Mean: []float32{0, 0, 0}, Std: []float32{1, 1, 1}, Patch: 14, Tokens: 512}}
	e.Model.Width, e.Model.Height = 28, 14
	img := image.NewRGBA(image.Rect(0, 0, 28, 14))
	img.Set(15, 2, color.RGBA{255, 0, 51, 255}) // patch 1, row 2, column 1
	p := e.Patches(img)
	for _, v := range p[:patchIn] {
		if v != 0 {
			t.Fatal("the class token's row is not zeros")
		}
	}
	row := p[2*patchIn:]
	if row[0*196+2*14+1] != 1 || math.Abs(float64(row[2*196+2*14+1])-0.2) > 1e-6 || row[1*196+2*14+1] != 0 {
		t.Errorf("patch 1 pixel (2, 1): %v %v %v", row[2*14+1], row[196+2*14+1], row[2*196+2*14+1])
	}
}

// TestNearest checks the size choice by aspect.
func TestNearest(t *testing.T) {
	e := &Estimator{Export: Export{Sizes: [][2]int{{504, 504}, {504, 378}, {378, 504}, {504, 280}, {280, 504}}}}
	for _, c := range []struct{ w, h, ww, wh int }{
		{1024, 1024, 504, 504}, {4000, 3000, 504, 378}, {1080, 1920, 280, 504}, {1024, 640, 504, 280}, {1100, 1000, 504, 504},
	} {
		if w, h := e.Nearest(c.w, c.h); w != c.ww || h != c.wh {
			t.Errorf("Nearest(%d, %d) = %dx%d, want %dx%d", c.w, c.h, w, h, c.ww, c.wh)
		}
	}
}
