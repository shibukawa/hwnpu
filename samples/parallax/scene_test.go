package main

import (
	"image"
	"image/color"
	"testing"
)

// layered is a scene of w x h whose left half is far and blue with a white stripe, and whose
// right half is near and red.
func layered(w, h int) *Scene {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	near := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.RGBA{0, 0, 255, 255}
			switch {
			case x >= w/2:
				c = color.RGBA{255, 0, 0, 255}
				near[y*w+x] = 1
			case x >= w/4 && x < w/4+4:
				c = color.RGBA{255, 255, 255, 255}
			}
			img.SetRGBA(x, y, c)
		}
	}
	return NewScene(img, near, 0, 0)
}

// edge is the first column of row y that is red, and stripe the first that is white.
func columns(img *image.RGBA, y int) (stripe, edge int) {
	stripe, edge = -1, -1
	for x := 0; x < img.Bounds().Dx(); x++ {
		c := img.RGBAAt(x, y)
		if stripe < 0 && c.R > 200 && c.G > 200 && c.B > 200 {
			stripe = x
		}
		if edge < 0 && c.R > 200 && c.G < 60 && c.B < 60 {
			edge = x
		}
	}
	return stripe, edge
}

func TestRender(t *testing.T) {
	const w, h = 400, 40
	s := layered(w, h)
	p := Params{Strength: 0.05} // the nearest point moves 20 pixels; 90 % of the picture is shown
	draw := func(v View) (stripe, edge int) {
		dst := image.NewRGBA(image.Rect(0, 0, w, h))
		s.Render(dst, v, p)
		return columns(dst, h/2)
	}
	stripe0, edge0 := draw(View{})
	// from the middle the picture is only made larger about its centre: the edge stays there
	if edge0 != w/2 {
		t.Errorf("from the middle the near half begins at %d, want %d", edge0, w/2)
	}
	stripeR, edgeR := draw(View{X: 1})
	stripeL, edgeL := draw(View{X: -1})
	// the far stripe does not move (Focus 0), the near half moves against the eye by the shift
	// as it is on the screen (20 pixels of the picture, shown 1/0.9 times as large)
	if stripeR != stripe0 || stripeL != stripe0 {
		t.Errorf("the far stripe is at %d, %d and %d from the right, the middle and the left: it is to stay", stripeR, stripe0, stripeL)
	}
	if got := edge0 - edgeR; got < 21 || got > 23 {
		t.Errorf("with the eye at the right the near half moved %d pixels to the left, want about 22", got)
	}
	// with the eye at the left the near half moves right and uncovers what was behind it, which
	// the picture does not have: the far side is drawn up to the near half's old edge and on
	// (stretched), never the near half's colour where it no longer is
	if edgeL <= edge0 {
		t.Errorf("with the eye at the left the near half begins at %d, want it right of %d", edgeL, edge0)
	}
	// Focus 0.5 turns the picture about its middle: far and near move opposite ways
	p.Focus = 0.5
	stripeF, edgeF := draw(View{X: 1})
	if !(stripeF > stripe0 && edgeF < edge0) {
		t.Errorf("with Focus 0.5 the far stripe went from %d to %d and the near edge from %d to %d: want right and left", stripe0, stripeF, edge0, edgeF)
	}
}

func TestSmallerAndSweep(t *testing.T) {
	s := layered(400, 40).smaller(100)
	if s.W != 100 || s.H != 10 || len(s.Near) != 1000 || s.Near[5*100+10] != 0 || s.Near[5*100+90] != 1 {
		t.Errorf("the smaller scene: %d x %d", s.W, s.H)
	}
	if v := sweep(0); v.X != 1 || v.Y != 0 {
		t.Errorf("a sweep begins at %+v", v)
	}
	if v := sweep(0.25); v.X > 1e-9 || v.X < -1e-9 || v.Y != 0.5 {
		t.Errorf("a quarter of a sweep is at %+v", v)
	}
	if v := (View{}).Follow(View{X: 1, Y: -1}, 0.25); v.X != 0.25 || v.Y != -0.25 {
		t.Errorf("Follow = %+v", v)
	}
}

func TestSpread(t *testing.T) {
	// one near pixel in a row of far ones becomes 2r + 1 wide, then soft at its ends
	near := make([]float32, 11*3)
	near[1*11+5] = 1
	got := Spread(near, 11, 3, 2)
	row := got[1*11 : 2*11]
	if row[5] != 1 || row[0] != 0 || row[10] != 0 || !(row[2] > 0 && row[2] < 1) || row[3] <= row[2] || row[7] != row[3] {
		t.Errorf("spread by 2: %v", row)
	}
	if got[0*11+5] == 0 {
		t.Errorf("the rows above and below are not reached: %v", got[:11])
	}
	if same := Spread(near, 11, 3, 0); &same[0] != &near[0] {
		t.Error("spread by 0 made a copy")
	}
}

func TestLayers(t *testing.T) {
	// the ends stay, the order stays, and the far end gets more of the range
	near := []float32{0, 0.01, 0.02, 0.1, 0.5, 1}
	got := Layers(near, 30)
	if got[0] != 0 || got[5] < 0.999 || got[5] > 1.001 {
		t.Errorf("the ends moved: %v", got)
	}
	for i := 1; i < len(got); i++ {
		if got[i] <= got[i-1] {
			t.Errorf("the order changed at %d: %v", i, got)
		}
	}
	if got[3] < 0.39 || got[3] > 0.42 { // log(4) / log(31)
		t.Errorf("a tenth became %v, want about 0.40", got[3])
	}
	if got[2]-got[1] < 5*(near[2]-near[1]) {
		t.Errorf("the far layers are not pulled apart: %v", got)
	}
	if same := Layers(near, 0); &same[0] != &near[0] {
		t.Error("layers of 0 made a copy")
	}
}
