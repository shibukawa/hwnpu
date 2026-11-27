package vision

import (
	"image"
	"image/color"
	"slices"
	"testing"
)

// box is the box of corners x1, y1, x2, y2.
func box(x1, y1, x2, y2 float64) Box { return Box{Point{x1, y1}, Point{x2, y2}} }

func TestBox(t *testing.T) {
	b := box(1.5, 2, 11.5, 22)
	if b.Dx() != 10 || b.Dy() != 20 || b.Min != (Point{1.5, 2}) {
		t.Errorf("box %+v", b)
	}
}

func TestEmptyPrompt(t *testing.T) {
	if _, err := (&Segmenter{}).Masks(&PictureEncoding{}, Prompt{}); err == nil {
		t.Error("masks of an empty prompt")
	}
	if (Prompt{Box: &Box{}}).empty() || (Prompt{Negative: []Point{{}}}).empty() || !(Prompt{}).empty() {
		t.Error("an empty prompt")
	}
}

func TestMask(t *testing.T) {
	// a 4 x 3 mask on the pixels (1, 1) and (2, 1), (2, 2)
	k := Mask{W: 4, H: 3, In: make([]bool, 12)}
	for _, i := range []int{5, 6, 10} {
		k.In[i] = true
	}
	if k.Area() != 3 || k.Bounds() != image.Rect(1, 1, 3, 3) {
		t.Errorf("area %d, bounds %v", k.Area(), k.Bounds())
	}
	if (&Mask{W: 2, H: 2, In: make([]bool, 4)}).Bounds() != (image.Rectangle{}) {
		t.Error("an empty mask has bounds")
	}
	g := k.Image()
	if g.GrayAt(1, 1).Y != 255 || g.GrayAt(0, 0).Y != 0 {
		t.Errorf("the mask's picture %v", g.Pix)
	}
	pic := image.NewRGBA(image.Rect(10, 10, 14, 13)) // not at the origin
	pic.Set(12, 12, color.RGBA{200, 100, 50, 255})
	c := k.Cutout(pic)
	if c.NRGBAAt(2, 2) != (color.NRGBA{200, 100, 50, 255}) || c.NRGBAAt(0, 0).A != 0 {
		t.Errorf("the cutout %v %v", c.NRGBAAt(2, 2), c.NRGBAAt(0, 0))
	}
	ms := Masks{Masks: []Mask{{Score: 0.5}, {Score: 0.9}, {Score: 0.7}}}
	if ms.Best() != 1 {
		t.Errorf("best %d", ms.Best())
	}
}

func TestBilinear(t *testing.T) {
	// Bilinear keeps a map of the same size, and halves one to the means of its pixels' pairs
	xs := []float32{0, 2, 4, 6}
	if got := Bilinear(xs, 4, 1, 4, 1); !slices.Equal(got, xs) {
		t.Errorf("the same size: %v", got)
	}
	if got := Bilinear(xs, 4, 1, 2, 1); !slices.Equal(got, []float32{1, 5}) {
		t.Errorf("halved: %v", got)
	}
	d := &DepthMap{W: 2, H: 1, Depth: []float32{2, 4}, Metric: true}
	if n := d.Near(2, 1); n[0] != 1 || n[1] != 0 {
		t.Errorf("near %v", n)
	}
}

func TestLabels(t *testing.T) {
	if len(DefaultLabels()) < 400 || !slices.Equal(ParseLabels("a\n# b\n\n c \n"), []string{"a", "c"}) {
		t.Errorf("%d default labels", len(DefaultLabels()))
	}
	if c := Cosine([]float32{1, 0}, []float32{0.5, 0.5}); c != 0.5 {
		t.Errorf("cosine %v", c)
	}
}
