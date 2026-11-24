package vision

import (
	"image"
	"image/color"
	"math"
	"slices"
	"testing"

	"github.com/shibukawa/bonsaiai/internal/chat"
	"github.com/shibukawa/bonsaiai/internal/imagedev"
	"github.com/shibukawa/bonsaiai/internal/vision/segment"
	"github.com/shibukawa/bonsaiai/internal/vision/siglip"
)

// TestSettings: what the definition of a model of pictures says (common.go).
func TestSettings(t *testing.T) {
	m := chat.Model{Name: "RTMDet-tiny+1a2b3c4d"}
	if threads(m) != 1 || interpreter(m) || program(m, "export") != "" {
		t.Errorf("a model without settings: %d threads, interpreter %v", threads(m), interpreter(m))
	}
	m.Settings = map[string]string{"threads": "4", "cpu": "interpreter"}
	m.Programs = map[string][]string{"export": {"/x/y.amd-npu2"}}
	if threads(m) != 4 || !interpreter(m) || program(m, "export") != "/x/y.amd-npu2" {
		t.Errorf("settings %v: %d threads, interpreter %v", m.Settings, threads(m), interpreter(m))
	}
	if shown(m.Name) != "RTMDet-tiny" || shown("YuNet") != "YuNet" {
		t.Errorf("shown %q", shown(m.Name))
	}
	// a device for every NPU is chosen as --device is; one of models.yaml for one NPU is not
	if d := imagedev.DeviceFlag(chat.Model{Device: map[string]string{"amd": "cpu", "apple": "cpu", "intel": "cpu"}}); d != "cpu" {
		t.Errorf("a definition's device for every NPU: %q", d)
	}
	if d := imagedev.DeviceFlag(chat.Model{Device: map[string]string{"amd": "cpu"}}); d != "" {
		t.Errorf("models.yaml's device for AMD: %q", d)
	}
}

func TestBox(t *testing.T) {
	b := box(float32(1.5), 2, 11.5, 22)
	if b.Dx() != 10 || b.Dy() != 20 || b.Min != (Point{1.5, 2}) {
		t.Errorf("box %+v", b)
	}
	if m := b.add(image.Pt(10, 20)); m.Min != (Point{11.5, 22}) || m.Dx() != 10 {
		t.Errorf("moved %+v", m)
	}
}

func TestPrompt(t *testing.T) {
	p := Prompt{Points: []Point{{1, 2}}, Negative: []Point{{3, 4}, {5, 6}}, Box: &Box{Point{7, 8}, Point{9, 10}}}
	want := []segment.Point{{X: 1, Y: 2, Label: 1}, {X: 3, Y: 4, Label: 0}, {X: 5, Y: 6, Label: 0}, {X: 7, Y: 8, Label: 2}, {X: 9, Y: 10, Label: 3}}
	if got := p.points(); !slices.Equal(got, want) {
		t.Errorf("points %v, want %v", got, want)
	}
	if _, err := (&Segmenter{}).Masks(&PictureEncoding{}, Prompt{}); err == nil {
		t.Error("masks of an empty prompt")
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

func TestRank(t *testing.T) {
	l := &Labeler{m: &siglip.Model{Export: siglip.Export{LogitScale: 10}}}
	pic := Embedding{1, 0}
	c := &Candidates{Labels: []string{"far", "near", "middle"}}
	c.Embeddings = []Embedding{{0, 1}, {1, 0}, {float32(math.Sqrt(0.5)), float32(math.Sqrt(0.5))}}
	got := l.Rank(pic, c, RankOptions{})
	if len(got) != 3 || got[0].Label != "near" || got[1].Label != "middle" || got[2].Label != "far" {
		t.Errorf("ranked %v", got)
	}
	if got := l.Rank(pic, c, RankOptions{Top: 1}); len(got) != 1 || got[0].Label != "near" {
		t.Errorf("the top one: %v", got)
	}
	if got := l.Rank(pic, c, RankOptions{Min: 0.6}); len(got) != 2 {
		t.Errorf("at least 0.6: %v", got)
	}
	if Cosine(pic, c.Embeddings[2]) < 0.7 || Cosine(pic, c.Embeddings[0]) != 0 {
		t.Error("cosines")
	}
	if _, err := l.Candidates(t.Context(), []string{"a"}, CandidateOptions{Template: "no braces"}); err == nil {
		t.Error("a template without {}")
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
