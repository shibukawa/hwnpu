package anima

import (
	"math"
	"testing"
)

// TestSigmas compares Sigmas with FlowMatchEulerDiscreteScheduler's for Anima (shift 3, 8
// steps), as tools/ref/ref_anima.py records them.
func TestSigmas(t *testing.T) {
	want := []float64{1.0, 0.95454544, 0.9, 0.8333333, 0.75, 0.64285713, 0.5, 0.3, 0.0}
	got := Sigmas(8, 3)
	if len(got) != len(want) {
		t.Fatalf("%d sigmas, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-6 {
			t.Errorf("sigma %d: %v, want %v", i, got[i], want[i])
		}
	}
	if s := Sigmas(1, 3); s[0] != 1 || s[1] != 0 {
		t.Errorf("one step: %v", s)
	}
}

// TestTimestepSinusoid compares with diffusers' Timesteps(2048, flip_sin_to_cos=True) at 0.7.
func TestTimestepSinusoid(t *testing.T) {
	e := TimestepSinusoid(0.7, 2048)
	want := map[int]float64{0: 0.7648421872844885, 1: 0.7688650293233295, 500: 0.9999695971789181, 1023: 0.9999999975055283,
		1024: 0.644217687237691, 1500: 0.009676454577983338, 2047: 7.063245308017101e-05}
	for i, w := range want {
		if math.Abs(float64(e[i])-w) > 1e-6 {
			t.Errorf("element %d: %v, want %v", i, e[i], w)
		}
	}
}

// TestGridShape checks the padded grid against iron_anima.GridShape at the sizes of a 1024 x
// 1024 image's decoder.
func TestGridShape(t *testing.T) {
	for _, c := range []struct {
		g         GridShape
		rows, p00 int
		pLast     int // the last interior pixel's row
	}{
		{GridShape{H: 128, W: 128, C: 384, Pad: 16}, 18944, 146, 146 + 127*144 + 127},
		{GridShape{H: 1024, W: 1024, C: 96, Pad: 16}, 1067264, 1042, 1042 + 1023*1040 + 1023},
	} {
		g := c.g
		if g.Rows() != c.rows || g.Pixel(0, 0) != c.p00 || g.Pixel(g.H-1, g.W-1) != c.pLast {
			t.Errorf("%+v: rows %d, pixel(0,0) %d, last %d; want %d, %d, %d", g, g.Rows(), g.Pixel(0, 0), g.Pixel(g.H-1, g.W-1), c.rows, c.p00, c.pLast)
		}
		if g.Rows()%256 != 0 || g.Pixel(g.H-1, g.W-1)+g.Wp() >= g.Rows() {
			t.Errorf("%+v: the bottom zero row does not fit", g)
		}
	}
}

func TestCondRows(t *testing.T) {
	c := make([]float32, 4*3)
	if n := condRows(c, 3); n != 1 {
		t.Fatalf("all zeros: %d rows, want 1", n)
	}
	c[1*3+2] = 0.5
	if n := condRows(c, 3); n != 2 {
		t.Fatalf("%d rows, want 2", n)
	}
	c[3*3] = -1
	if n := condRows(c, 3); n != 4 {
		t.Fatalf("%d rows, want 4", n)
	}
}
