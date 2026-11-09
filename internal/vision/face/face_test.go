package face

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestChoose(t *testing.T) {
	d := &Detector{Export: Export{Sizes: [][2]int{{320, 256}, {640, 480}}}}
	for _, c := range []struct{ w, h, W, H int }{
		{640, 480, 640, 480},   // fits as it is
		{320, 240, 320, 256},   // fits both: the smaller
		{2048, 1150, 640, 480}, // scaled down least
		{100, 80, 320, 256},
	} {
		if W, H := d.Choose(c.w, c.h); W != c.W || H != c.H {
			t.Errorf("Choose(%d, %d) = %dx%d, want %dx%d", c.w, c.h, W, H, c.W, c.H)
		}
	}
	d.Size = [2]int{320, 256}
	if W, H := d.Choose(640, 480); W != 320 || H != 256 {
		t.Errorf("Size: %dx%d", W, H)
	}
}

// TestDecode decodes one confident cell of stride 8 as OpenCV's postProcess does.
func TestDecode(t *testing.T) {
	d := &Detector{Export: Export{Channels: 64}, Score: 0.9, NMS: 0.3, TopK: 5000}
	d.Head.Cls, d.Head.Obj, d.Head.BBox, d.Head.Kps = 0, 1, 2, 6
	W, H, s := 32, 16, 8
	out := make([]byte, (W/s)*(H/s)*64*2)
	put := func(cell, ch int, v float32) { binary.LittleEndian.PutUint16(out[2*(cell*64+ch):], toBF16(v)) }
	for cell := 0; cell < (W/s)*(H/s); cell++ {
		put(cell, 0, -10)
	}
	cell := 1*(W/s) + 2 // row 1, column 2
	put(cell, 0, 8)
	put(cell, 1, 8)
	put(cell, 2, 0.5)
	put(cell, 3, 0.25)
	put(cell, 4, float32(math.Log(4)))
	put(cell, 5, 0)
	put(cell, 6, -1)
	put(cell, 7, 1)
	f := d.decode(out, W, H, s)
	if len(f) != 1 {
		t.Fatalf("%d faces", len(f))
	}
	// cx = (2 + 0.5) 8 = 20, cy = (1 + 0.25) 8 = 10, w = 4 x 8 = 32, h = 8
	if g := f[0]; math.Abs(float64(g.X-4)) > 0.1 || math.Abs(float64(g.Y-6)) > 0.1 || math.Abs(float64(g.W-32)) > 0.2 || g.H != 8 {
		t.Errorf("box %+v", g)
	}
	if l := f[0].Landmarks[0]; l != [2]float32{8, 16} {
		t.Errorf("landmark %v", l)
	}
}

func TestSuppress(t *testing.T) {
	d := &Detector{NMS: 0.3, TopK: 5000}
	f := d.suppress([]Face{
		{X: 0, Y: 0, W: 10, H: 10, Score: 0.91},
		{X: 1, Y: 1, W: 10, H: 10, Score: 0.95},
		{X: 50, Y: 50, W: 10, H: 10, Score: 0.92},
	})
	if len(f) != 2 || f[0].Score != 0.95 || f[1].Score != 0.92 {
		t.Errorf("%+v", f)
	}
}
