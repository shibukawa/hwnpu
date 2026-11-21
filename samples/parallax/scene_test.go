package main

import (
	"image"
	"image/color"
	"math"
	"testing"
	"time"

	"github.com/shibukawa/hwnpu/samples/internal/layering"
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

// TestRenderLayers draws a near red layer over a far layer that has a green stripe behind the
// red layer's left edge: with the eye to one side the red layer moves off it, and what it
// uncovers is the far layer's own picture, the green stripe, not the red edge stretched.
func TestRenderLayers(t *testing.T) {
	const w, h = 400, 40
	back := image.NewRGBA(image.Rect(0, 0, w, h))
	front := image.NewNRGBA(image.Rect(0, 0, w, h))
	far := make([]float32, w*h)
	near := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := color.RGBA{0, 0, 255, 255}
			if x >= 200 && x < 210 {
				c = color.RGBA{0, 255, 0, 255} // hidden behind the red layer when the eye is in the middle
			}
			back.SetRGBA(x, y, c)
			if x >= 200 {
				front.SetNRGBA(x, y, color.NRGBA{255, 0, 0, 255})
			}
			near[y*w+x] = 1 // the layer's nearness goes on past its edge
		}
	}
	s := NewScene(front, near, 0, 0)
	s.Behind = NewScene(back, far, 0, 0)
	if n := len(s.layers()); n != 2 {
		t.Fatalf("%d layers", n)
	}
	p := Params{Strength: 0.05} // the nearest point moves 20 pixels
	count := func(v View) (red, green, other int) {
		dst := image.NewRGBA(image.Rect(0, 0, w, h))
		s.Render(dst, v, p)
		for x := 0; x < w; x++ {
			switch c := dst.RGBAAt(x, h/2); {
			case c.A != 255:
				other++
			case c.R > 200 && c.G < 60 && c.B < 60:
				red++
			case c.G > 200 && c.R < 60 && c.B < 60:
				green++
			case c.B > 200 && c.R < 60 && c.G < 60:
			default:
				other++
			}
		}
		return red, green, other
	}
	// from the middle the stripe is hidden
	if _, green, other := count(View{}); green != 0 || other > 2 {
		t.Errorf("from the middle: %d green pixels, %d of other colours", green, other)
	}
	// from one side the red layer has moved off the stripe: all of it shows (10 pixels of the
	// picture, 11 on a screen that shows 90 % of it), and nothing is stretched between
	side := View{X: -1}
	if _, green, _ := count(side); green == 0 {
		side = View{X: 1}
	}
	red, green, other := count(side)
	if green < 10 || green > 12 || other > 2 {
		t.Errorf("from the side: %d red, %d green, %d of other colours; want the stripe's 11 green", red, green, other)
	}
	// one layer with the same picture and a step in its nearness stretches the edge instead:
	// the stripe is not in that picture at all
	flat := layered(w, h)
	dst := image.NewRGBA(image.Rect(0, 0, w, h))
	flat.Render(dst, side, p)
	for x := 0; x < w; x++ {
		if c := dst.RGBAAt(x, h/2); c.G > 200 && c.R < 60 {
			t.Fatalf("a picture of one layer shows green at %d", x)
		}
	}
	// the layers are kept when the scene is made smaller
	if small := s.smaller(100); small.Behind == nil || small.Behind.W != 100 || small.Pix.RGBAAt(10, 5).A != 0 || small.Pix.RGBAAt(90, 5).A != 255 {
		t.Errorf("the smaller scene lost its layers")
	}
}

func TestStackAndProgress(t *testing.T) {
	const w, h = 8, 4
	layer := func(a uint8, n float32) layering.Layer {
		l := layering.Layer{Img: image.NewNRGBA(image.Rect(0, 0, w, h)), Near: make([]float32, w*h)}
		for i := range l.Near {
			l.Img.Pix[4*i], l.Img.Pix[4*i+3] = 200, a
			l.Near[i] = n
		}
		return l
	}
	s := stack([]layering.Layer{layer(255, 0.1), layer(128, 0.5), layer(0, 1)}, 0, 0)
	ls := s.layers()
	if len(ls) != 3 || ls[0].Behind != nil || ls[2] != s || ls[0].Near[0] != 0.1 || ls[2].Near[0] != 1 {
		t.Fatalf("the stack: %d layers", len(ls))
	}
	// a layer's colours hold their alpha in the scene
	if c := ls[1].Pix.RGBAAt(0, 0); c.A != 128 || c.R < 99 || c.R > 101 {
		t.Errorf("the middle layer's pixel: %v", c)
	}

	// the steps are queued for the window, each taken once, with how much of the whole was done
	// when it came; the shares of the whole add up to 1, the window's putting together the rest
	p := newProgress(ls[0].Pix, 3)
	p.show(stageCut, ls[1].Pix, nil, "cut")
	p.set(func(p *progress) { p.total = 0.5 })
	p.ready(s)
	n := p.take()
	if len(n.steps) != 3 || n.steps[0].stage != stageDepth || n.steps[1].text != "cut" || n.steps[2].scene != s {
		t.Fatalf("the steps taken: %+v", n.steps)
	}
	if math.Abs(n.steps[2].total+n.wCompose-1) > 1e-9 || n.steps[1].total != 0 {
		t.Errorf("the shares: %v, %v and %v", n.steps[1].total, n.steps[2].total, n.wCompose)
	}
	if n := p.take(); len(n.steps) != 0 {
		t.Errorf("taken again: %d steps", len(n.steps))
	}
	if one := newProgress(ls[0].Pix, 1); one.wLayers != 0 {
		t.Errorf("a picture of one layer has a share for its layers: %v", one.wLayers)
	}
	if minShow(stageNear) < 3*time.Second {
		t.Errorf("the depth is shown %v", minShow(stageNear))
	}
}
