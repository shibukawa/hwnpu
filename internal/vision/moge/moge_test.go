package moge

import (
	"math"
	"testing"
)

// TestRecoverFocalShift builds a point map from a known focal length and z shift (points =
// (uv / focal * z', z' - shift) for the true depth z') and checks they come back.
func TestRecoverFocalShift(t *testing.T) {
	const w, h = 96, 72
	focal, shift := 0.9, 0.35
	uv := ViewUV(w, h)
	pts := make([]float32, 3*w*h)
	for i := 0; i < w*h; i++ {
		z := 2 + math.Sin(float64(i%w)/7) + 0.5*math.Cos(float64(i/w)/5) // the true depth
		pts[3*i] = float32(uv[2*i] / focal * z)
		pts[3*i+1] = float32(uv[2*i+1] / focal * z)
		pts[3*i+2] = float32(z - shift)
	}
	f, s := RecoverFocalShift(pts, nil, w, h)
	if math.Abs(f-focal) > 1e-3 || math.Abs(s-shift) > 1e-3 {
		t.Errorf("focal %v shift %v, want %v %v", f, s, focal, shift)
	}
}

// TestViewUV checks normalized_view_plane_uv's corners: (-+ w / diagonal, -+ h / diagonal)
// times (n - 1) / n.
func TestViewUV(t *testing.T) {
	uv := ViewUV(4, 3)
	d := math.Hypot(4, 3)
	if math.Abs(uv[0]-(-4/d*3/4)) > 1e-9 || math.Abs(uv[1]-(-3/d*2/3)) > 1e-9 {
		t.Errorf("top left %v %v", uv[0], uv[1])
	}
	last := 2 * (4*3 - 1)
	if math.Abs(uv[last]-4/d*3/4) > 1e-9 || math.Abs(uv[last+1]-3/d*2/3) > 1e-9 {
		t.Errorf("bottom right %v %v", uv[last], uv[last+1])
	}
}

// TestBilinear checks torch's align_corners=False: 2 -> 4 gives 0, 0.25, 0.75, 1 of [0, 1].
func TestBilinear(t *testing.T) {
	got := Bilinear([]float32{0, 1}, 2, 1, 4, 1)
	want := []float32{0, 0.25, 0.75, 1}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-6 {
			t.Fatalf("Bilinear = %v, want %v", got, want)
		}
	}
}
