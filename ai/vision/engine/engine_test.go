package engine

import (
	"image"
	"math"
	"slices"
	"testing"

	"github.com/shibukawa/hwnpu/ai/vision"
	"github.com/shibukawa/hwnpu/internal/chat"
	"github.com/shibukawa/hwnpu/internal/imagedev"
	"github.com/shibukawa/hwnpu/internal/pose"
	"github.com/shibukawa/hwnpu/internal/vision/depth"
	"github.com/shibukawa/hwnpu/internal/vision/detect"
	"github.com/shibukawa/hwnpu/internal/vision/moge"
	"github.com/shibukawa/hwnpu/internal/vision/segment"
	"github.com/shibukawa/hwnpu/internal/vision/siglip"
	"github.com/shibukawa/hwnpu/internal/visionwire"
)

// TestSettings: what the definition of a model of pictures says (ai/vision's common.go).
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
	if b.Dx() != 10 || b.Dy() != 20 || b.Min != (vision.Point{X: 1.5, Y: 2}) {
		t.Errorf("box %+v", b)
	}
	if m := add(b, image.Pt(10, 20)); m.Min != (vision.Point{X: 11.5, Y: 22}) || m.Dx() != 10 {
		t.Errorf("moved %+v", m)
	}
}

func TestPrompt(t *testing.T) {
	p := vision.Prompt{Points: []vision.Point{{X: 1, Y: 2}}, Negative: []vision.Point{{X: 3, Y: 4}, {X: 5, Y: 6}},
		Box: &vision.Box{Min: vision.Point{X: 7, Y: 8}, Max: vision.Point{X: 9, Y: 10}}}
	want := []segment.Point{{X: 1, Y: 2, Label: 1}, {X: 3, Y: 4, Label: 0}, {X: 5, Y: 6, Label: 0}, {X: 7, Y: 8, Label: 2}, {X: 9, Y: 10, Label: 3}}
	if got := points(p); !slices.Equal(got, want) {
		t.Errorf("points %v, want %v", got, want)
	}
}

func TestRank(t *testing.T) {
	l := &labeler{m: &siglip.Model{Export: siglip.Export{LogitScale: 10}}}
	pic := vision.Embedding{1, 0}
	c := &vision.Candidates{Labels: []string{"far", "near", "middle"}}
	c.Embeddings = []vision.Embedding{{0, 1}, {1, 0}, {float32(math.Sqrt(0.5)), float32(math.Sqrt(0.5))}}
	got := l.Rank(pic, c, vision.RankOptions{})
	if len(got) != 3 || got[0].Label != "near" || got[1].Label != "middle" || got[2].Label != "far" {
		t.Errorf("ranked %v", got)
	}
	if got := l.Rank(pic, c, vision.RankOptions{Top: 1}); len(got) != 1 || got[0].Label != "near" {
		t.Errorf("the top one: %v", got)
	}
	if got := l.Rank(pic, c, vision.RankOptions{Min: 0.6}); len(got) != 2 {
		t.Errorf("at least 0.6: %v", got)
	}
	if vision.Cosine(pic, c.Embeddings[2]) < 0.7 || vision.Cosine(pic, c.Embeddings[0]) != 0 {
		t.Error("cosines")
	}
	if _, err := l.Candidates(t.Context(), []string{"a"}, vision.CandidateOptions{Template: "no braces"}); err == nil {
		t.Error("a template without {}")
	}
}

// TestWire: what ai/vision takes from internal/visionwire is what the engine's packages have.
func TestWire(t *testing.T) {
	if !slices.Equal(visionwire.COCO, detect.COCO) {
		t.Error("visionwire.COCO is not detect.COCO")
	}
	if !slices.Equal(visionwire.Keypoints, pose.Names) {
		t.Error("visionwire.Keypoints is not pose.Names")
	}
	if visionwire.SkyThreshold != depth.SkyThreshold {
		t.Error("visionwire.SkyThreshold is not depth.SkyThreshold")
	}
	if !slices.Equal(visionwire.DefaultLabels(), siglip.DefaultLabels()) {
		t.Error("visionwire.DefaultLabels is not siglip.DefaultLabels")
	}
	s := "a\n# b\n\n  c d \n"
	if !slices.Equal(visionwire.ParseLabels(s), siglip.ParseLabels(s)) {
		t.Error("visionwire.ParseLabels is not siglip.ParseLabels")
	}
	a, b := []float32{0.25, -1, 3}, []float32{2, 0.5, -0.125}
	if visionwire.Dot(a, b) != siglip.Dot(a, b) {
		t.Error("visionwire.Dot is not siglip.Dot")
	}
	xs := []float32{0, 1, 4, 9, 16, 25}
	for _, sz := range [][2]int{{3, 2}, {5, 3}, {2, 1}, {7, 4}} {
		if !slices.Equal(visionwire.Bilinear(xs, 3, 2, sz[0], sz[1]), moge.Bilinear(xs, 3, 2, sz[0], sz[1])) {
			t.Errorf("visionwire.Bilinear is not moge.Bilinear at %v", sz)
		}
	}
}
