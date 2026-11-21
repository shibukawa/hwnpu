package vision

import (
	"image"
	"math"
	"testing"

	"github.com/shibukawa/hwnpu/ai/flow"
)

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestHeadOf(t *testing.T) {
	frame := image.Rect(0, 0, 640, 480)
	face := func(cx, cy, apart float64) Face {
		return Face{Box: image.Rect(int(cx-apart), int(cy-apart), int(cx+apart), int(cy+apart)),
			RightEye: Point{cx - apart/2, cy}, LeftEye: Point{cx + apart/2, cy}}
	}
	// a camera of 60 degrees across 640 pixels has a focal length of 554 pixels: eyes 63 mm
	// apart that are 63 pixels apart are 0.554 m away
	h := HeadOf(face(320, 240, 63), frame, HeadOptions{})
	if !near(h.X, 0, 1e-9) || !near(h.Y, 0, 1e-9) || !near(h.Z, 0.554, 0.001) {
		t.Errorf("a face in the middle: %+v", h)
	}
	// twice as large in the frame: half as far
	if h := HeadOf(face(320, 240, 126), frame, HeadOptions{}); !near(h.Z, 0.277, 0.001) {
		t.Errorf("a face twice as large is %.3f m away, want 0.277", h.Z)
	}
	// on the frame's left and upper side: the person's right (the camera looks at them), and up
	h = HeadOf(face(160, 120, 63), frame, HeadOptions{})
	if !near(h.X, 0.5, 1e-9) || !near(h.Y, 0.5, 1e-9) || !near(h.Right, 0.16, 0.001) || !near(h.Up, 0.12, 0.001) {
		t.Errorf("a face on the frame's upper left: %+v", h)
	}
	// a wider camera: the same face is nearer
	if h := HeadOf(face(320, 240, 63), frame, HeadOptions{FOV: 90}); !near(h.Z, 0.32, 0.001) {
		t.Errorf("with a field of view of 90 degrees: %.3f m, want 0.32", h.Z)
	}
	// without the eyes, the box
	box := Face{Box: image.Rect(250, 170, 390, 310)}
	if h := HeadOf(box, frame, HeadOptions{}); !near(h.X, 0, 1e-9) || !near(h.Z, 0.554*63/(140/2.2), 0.01) {
		t.Errorf("a face without landmarks: %+v", h)
	}

	// the nearest of several is the largest
	found := []Face{face(100, 100, 20), face(500, 300, 80), face(300, 200, 40)}
	if h, ok := NearestHead(found, frame, HeadOptions{}); !ok || !near(h.X, -(500.0-320)/320, 1e-9) {
		t.Errorf("the nearest head: %+v %v", h, ok)
	}
	if _, ok := NearestHead(nil, frame, HeadOptions{}); ok {
		t.Error("a head without a face")
	}
}

func TestHeadNode(t *testing.T) {
	frame := image.Rect(0, 0, 640, 480)
	one := Face{Box: image.Rect(280, 200, 360, 280), RightEye: Point{300, 240}, LeftEye: Point{340, 240}}
	src := flow.Source[Found]("found").From(flow.From(Found{Frame: frame}, Found{Frame: frame, Faces: []Face{one}}, Found{Frame: frame}))
	var got []Head
	src.To(HeadNode(HeadOptions{})).To(flow.Sink[Head]("heads").Do(func(c flow.Chunk[Head]) error {
		got = append(got, c.Data)
		return nil
	}))
	if err := flow.Run(t.Context(), src); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || !near(got[0].X, 0, 1e-9) {
		t.Errorf("heads %+v, want one, for the frame with a face", got)
	}
}

func TestNear(t *testing.T) {
	// depths 1, 2, 4 and none: the inverse, from the farthest (0) to the nearest (1)
	d := &DepthMap{W: 2, H: 2, Depth: []float32{1, 2, 4, 0}}
	n := d.Near(2, 2)
	if !near(float64(n[0]), 1, 1e-6) || !near(float64(n[1]), (0.5-0.25)/0.75, 1e-6) || n[2] != 0 || n[3] != 0 {
		t.Errorf("near %v", n)
	}
	if img := d.NearImage(4, 4); img.Bounds().Dx() != 4 || img.GrayAt(0, 0).Y != 255 || img.GrayAt(0, 3).Y != 0 {
		t.Errorf("the nearness as a picture of 4 x 4: %v", img.Pix)
	}
	if n := (&DepthMap{W: 1, H: 1, Depth: []float32{3}}).Near(2, 2); len(n) != 4 || n[0] != 0 {
		t.Errorf("a flat depth: %v", n)
	}
}
