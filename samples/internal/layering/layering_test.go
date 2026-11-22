package layering

import (
	"context"
	"image"
	"image/color"
	"math"
	"strings"
	"testing"
)

func TestCuts(t *testing.T) {
	// three groups of values: the cuts go in the middle of the gaps between them
	var v []float32
	for i := 0; i < 300; i++ {
		v = append(v, 0.02+0.0001*float32(i))
	}
	for i := 0; i < 100; i++ {
		v = append(v, 0.40+0.001*float32(i))
	}
	for i := 0; i < 200; i++ {
		v = append(v, 0.85+0.0005*float32(i))
	}
	got := cuts(v, 3)
	if len(got) != 2 || !(got[0] > 0.15 && got[0] < 0.30) || !(got[1] > 0.60 && got[1] < 0.75) {
		t.Errorf("cuts = %v, want one between the groups each", got)
	}
	if one := cuts(v, 2); len(one) != 1 || !(one[0] > 0.06 && one[0] < 0.85) {
		t.Errorf("a cut in two = %v", one)
	}
	if none := cuts(v, 1); len(none) != 0 {
		t.Errorf("one class has cuts: %v", none)
	}
}

func TestCurveAndFit(t *testing.T) {
	c := curve([]float32{0, 0.1, 1, 2, -1}, 30)
	if c[0] != 0 || math.Abs(float64(c[1])-0.4037) > 1e-3 || math.Abs(float64(c[2])-1) > 1e-6 || c[3] != c[2] || c[4] != 0 {
		t.Errorf("curve = %v", c)
	}
	// y = 0.5 x + 0.1 outside the mask; inside it y is something else, which is not looked at
	x := make([]float32, 100)
	y := make([]float32, 100)
	mask := make([]float32, 100)
	for i := range x {
		x[i] = float32(i) / 100
		y[i] = 0.5*x[i] + 0.1
		if i >= 60 {
			mask[i], y[i] = 1, 9
		}
	}
	if a, b := fit(x, y, mask); math.Abs(a-0.5) > 1e-4 || math.Abs(b-0.1) > 1e-4 {
		t.Errorf("fit = %v, %v, want 0.5, 0.1", a, b)
	}
	if a, b := fit(x[:4], y[:4], mask[:4]); a != 1 || b != 0 {
		t.Errorf("a fit of too few points = %v, %v", a, b)
	}
}

func TestMasks(t *testing.T) {
	const w, h = 9, 5
	m := make([]float32, w*h)
	m[2*w+4] = 1
	g := grow(m, w, h, 1)
	if g[2*w+3] != 1 || g[1*w+5] != 1 || g[2*w+2] != 0 || covered(g) != 9.0/45 {
		t.Errorf("grown by 1: %v", g)
	}
	s := soften(g, w, h, 1)
	if !(s[2*w+4] == 1 && s[2*w+2] > 0 && s[2*w+2] < 1 && s[2*w+1] == 0) {
		t.Errorf("softened: %v", s[2*w:3*w])
	}
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 10, 250, 10, 255 // a green picture
	}
	c, word := key(img)
	if word == "green" {
		t.Errorf("the key of a green picture is green")
	}
	p := paint(img, m, c)
	if got := p.RGBAAt(4, 2); got != c {
		t.Errorf("painted: %v, want %v", got, c)
	}
	if got := p.RGBAAt(0, 0); got != (color.RGBA{10, 250, 10, 255}) {
		t.Errorf("outside the mask: %v", got)
	}
	half := make([]float32, w*h)
	for i := range half {
		half[i] = 0.5
	}
	if got := mix(img, p, half).RGBAAt(4, 2); got.R != uint8((10+int(c.R)+1)/2) {
		t.Errorf("mixed: %v", got)
	}
	l := cutOut(img, s)
	if l.NRGBAAt(4, 2).A != 255 || l.NRGBAAt(0, 0).A != 0 || l.NRGBAAt(4, 2).G != 250 {
		t.Errorf("cut out: %v %v", l.NRGBAAt(4, 2), l.NRGBAAt(0, 0))
	}
	if st := stretch(img, 18, 10); st.Rect.Dx() != 18 || st.Rect.Dy() != 10 || st.RGBAAt(9, 5).G < 245 {
		t.Errorf("stretched: %v %v", st.Rect, st.RGBAAt(9, 5))
	}
}

func TestSeam(t *testing.T) {
	// a picture of stripes; a square of it is the mask (its edges inside the blocks seam looks at)
	const w, h = 256, 256
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	m := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetRGBA(x, y, color.RGBA{uint8(40 + 20*(y/8%2)), 120, 200, 255})
			if x >= 67 && x < 189 && y >= 67 && y < 189 {
				m[y*w+x] = 1
			}
		}
	}
	if s := seam(img, m, w, h); s > 10 {
		t.Errorf("a picture that goes on across the mask's edge has a seam of %v", s)
	}
	if s := seam(paint(img, m, color.RGBA{0, 60, 20, 255}), m, w, h); s < 80 {
		t.Errorf("a flat shape in the mask has a seam of %v", s)
	}
	if s := seam(img, make([]float32, w*h), w, h); s != 0 {
		t.Errorf("no mask has a seam of %v", s)
	}
}

func TestExtendAndUncurve(t *testing.T) {
	// a layer in the middle of a row: its ends' values go on to the row's ends
	const w, h = 9, 3
	v := make([]float32, w*h)
	mask := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 3; x <= 5; x++ {
			v[y*w+x], mask[y*w+x] = 0.5+0.1*float32(x-3), 1
		}
		v[y*w] = 0.01 // what is behind the layer, which is not to show in it
	}
	got := extend(v, mask, w, h)
	row := got[w : 2*w]
	if row[3] != 0.5 || row[5] != 0.7 || math.Abs(float64(row[0])-0.5) > 1e-6 || math.Abs(float64(row[8])-0.7) > 1e-6 {
		t.Errorf("extended: %v", row)
	}
	if whole := extend(v, make([]float32, w*h), w, h); whole[0] != 0.01 {
		t.Errorf("no mask: %v", whole[:w])
	}
	for _, n := range []float32{0, 0.01, 0.3, 1} {
		c := curve([]float32{n}, 30)[0]
		if back := uncurve(c, 30); math.Abs(float64(back-n)) > 1e-5 {
			t.Errorf("uncurve(curve(%v)) = %v", n, back)
		}
	}
}

// TestSplit parts a picture of three bands, each nearer than the one above, with models that
// stand in for the real ones: the nearness is given back as it was, and what a layer hid is
// drawn in one plain colour.
func TestSplit(t *testing.T) {
	const w, h = 64, 60
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	near := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			band := y / 20 // 0 the farthest
			img.SetRGBA(x, y, color.RGBA{uint8(80 * band), 100, 50, 255})
			near[y*w+x] = []float32{0.02, 0.3, 0.9}[band]
		}
	}
	first := append([]float32(nil), near...)
	draws := 0
	var steps []Step
	m := Models{
		Near: func(ctx context.Context, p *image.RGBA) ([]float32, error) { return first, nil },
		Draw: func(ctx context.Context, prompt string, ref *image.RGBA, seed uint64, progress func(float64), phase func(string)) (*image.RGBA, error) {
			draws++
			if !strings.Contains(prompt, "Remove the") || !strings.Contains(prompt, "no birds.") || seed != 7 {
				t.Errorf("asked %q with seed %d", prompt, seed)
			}
			progress(0.5)
			out := image.NewRGBA(ref.Rect)
			for i := 0; i < len(out.Pix); i += 4 {
				out.Pix[i], out.Pix[i+1], out.Pix[i+2], out.Pix[i+3] = 9, 9, 200, 255
			}
			return out, nil
		},
	}
	layers, err := Split(context.Background(), img, near, m, Options{Layers: 3, Far: 30, Edge: 0, Clean: 0, Hint: "no birds", Seed: 7, Tries: 1,
		Progress: func(s Step) { steps = append(steps, s) }})
	if err != nil {
		t.Fatal(err)
	}
	if len(layers) != 3 || draws != 2 {
		t.Fatalf("%d layers, %d drawings", len(layers), draws)
	}
	// the farthest layer is whole, and drawn where the others were; the others have their band
	far, mid, front := layers[0], layers[1], layers[2]
	if c := far.Img.NRGBAAt(10, 5); c != (color.NRGBA{0, 100, 50, 255}) {
		t.Errorf("the far layer's own band: %v", c)
	}
	if c := far.Img.NRGBAAt(10, 50); c != (color.NRGBA{9, 9, 200, 255}) {
		t.Errorf("the far layer behind the others: %v", c)
	}
	if mid.Img.NRGBAAt(10, 5).A != 0 || mid.Img.NRGBAAt(10, 30) != (color.NRGBA{80, 100, 50, 255}) || mid.Img.NRGBAAt(10, 50) != (color.NRGBA{9, 9, 200, 255}) {
		t.Errorf("the middle layer: %v %v %v", mid.Img.NRGBAAt(10, 5), mid.Img.NRGBAAt(10, 30), mid.Img.NRGBAAt(10, 50))
	}
	if front.Img.NRGBAAt(10, 30).A != 0 || front.Img.NRGBAAt(10, 50) != (color.NRGBA{160, 100, 50, 255}) {
		t.Errorf("the near layer: %v %v", front.Img.NRGBAAt(10, 30), front.Img.NRGBAAt(10, 50))
	}
	// a layer's nearness goes on past its band with the band's value
	if math.Abs(float64(front.Near[5*w+10])-0.9) > 1e-5 || math.Abs(float64(mid.Near[5*w+10])-0.3) > 1e-5 {
		t.Errorf("the nearness past the layers' edges: %v %v", front.Near[5*w+10], mid.Near[5*w+10])
	}
	// the steps: for each of the two layers cut, asked, drawing (begun, half), drawn, parted;
	// the whole grows to 1
	var kinds []string
	last := -1.0
	for _, s := range steps {
		kinds = append(kinds, s.Kind)
		if s.Total < last || s.Of != 3 {
			t.Errorf("step %s of layer %d: the whole at %v after %v, of %d", s.Kind, s.Layer, s.Total, last, s.Of)
		}
		last = s.Total
	}
	want := "cut asked drawing drawing drawn parted cut asked drawing drawing drawn parted"
	if got := strings.Join(kinds, " "); got != want || last != 1 {
		t.Errorf("the steps: %s, ending at %v", got, last)
	}
	if p := steps[5]; len(p.Layers) != 2 || p.Left != 1 || steps[11].Left != 0 || len(steps[11].Layers) != 3 {
		t.Errorf("parted so far: %d layers, %d left", len(p.Layers), p.Left)
	}

	// written and read back, the layers are the same
	dir := t.TempDir()
	if err := Write(dir, layers); err != nil {
		t.Fatal(err)
	}
	back, err := Read(dir)
	if err != nil || len(back) != 3 {
		t.Fatalf("read back: %d layers, %v", len(back), err)
	}
	if back[2].Img.NRGBAAt(10, 50) != front.Img.NRGBAAt(10, 50) || back[2].Img.NRGBAAt(10, 5).A != 0 || math.Abs(float64(back[1].Near[5*w+10]-0.3)) > 1e-4 {
		t.Errorf("read back: %v, nearness %v", back[2].Img.NRGBAAt(10, 50), back[1].Near[5*w+10])
	}
	if _, err := Read(t.TempDir()); err == nil {
		t.Error("an empty directory has layers")
	}
}

// TestSplitKept: when the model leaves the shape in every try, the layer is not cut out and the
// layers cut before are what there is.
func TestSplitKept(t *testing.T) {
	const w, h = 256, 240
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	near := make([]float32, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			band := y / 80
			img.SetRGBA(x, y, color.RGBA{uint8(80 * band), 100, 50, 255})
			near[y*w+x] = []float32{0.02, 0.3, 0.9}[band]
		}
	}
	draws := 0
	var last Step
	m := Models{
		Near: func(ctx context.Context, p *image.RGBA) ([]float32, error) { return near, nil },
		Draw: func(ctx context.Context, prompt string, ref *image.RGBA, seed uint64, progress func(float64), phase func(string)) (*image.RGBA, error) {
			draws++
			if draws == 1 { // the first layer's background goes on from the picture above it
				out := image.NewRGBA(ref.Rect)
				for i := 0; i < len(out.Pix); i += 4 {
					out.Pix[i], out.Pix[i+1], out.Pix[i+2], out.Pix[i+3] = 80, 100, 50, 255
				}
				return out, nil
			}
			return ref, nil // then the painted shape is given back as it is
		},
	}
	layers, err := Split(context.Background(), img, near, m, Options{Layers: 3, Far: 30, Edge: 0, Clean: 0, Tries: 2, Progress: func(s Step) { last = s }})
	if err != nil {
		t.Fatal(err)
	}
	if len(layers) != 2 || draws != 3 || last.Kind != Kept || last.Layer != 2 || last.Total != 1 || len(last.Layers) != 2 {
		t.Fatalf("%d layers after %d drawings; the last step %s of layer %d at %v", len(layers), draws, last.Kind, last.Layer, last.Total)
	}
	// the farthest layer is the picture without the near layer: the middle band is still in it
	if c := layers[0].Img.NRGBAAt(10, 120); c != (color.NRGBA{80, 100, 50, 255}) {
		t.Errorf("the far layer where the middle band is: %v", c)
	}
	if layers[1].Img.NRGBAAt(10, 200).A != 255 || layers[1].Img.NRGBAAt(10, 120).A != 0 {
		t.Errorf("the near layer: %v %v", layers[1].Img.NRGBAAt(10, 200), layers[1].Img.NRGBAAt(10, 120))
	}
}
