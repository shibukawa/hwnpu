package anima

import "testing"

// A grid whose latents are no multiple of 16 (iron_anima.GridShape): rows padded to a multiple
// of 16 pixels, the convolutions' rows to Ha; a square of 16's multiple as before (the rows as iron_anima.GridShape gives them).
func TestGridShapeOdd(t *testing.T) {
	for _, c := range []struct {
		g          GridShape
		wp, ha, rs int
	}{
		{GridShape{H: 128, W: 128, Pad: 16}, 144, 128, 18944},
		{GridShape{H: 156, W: 104, Pad: 16}, 128, 160, 20992},
		{GridShape{H: 146, W: 110, Pad: 16}, 128, 160, 20992},
	} {
		if c.g.Wp() != c.wp || c.g.Ha() != c.ha || c.g.Rows() != c.rs {
			t.Errorf("%+v: Wp %d Ha %d Rows %d, want %d %d %d", c.g, c.g.Wp(), c.g.Ha(), c.g.Rows(), c.wp, c.ha, c.rs)
		}
	}
}
