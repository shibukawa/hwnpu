package qwenimage

import (
	"math"
	"testing"
)

func TestSigmas(t *testing.T) {
	// Turbo: the checkpoint's sigmas through a scheduler of shift 1, the terminal 0 after them
	turbo := &Pipeline{Model: Model{SampleSigmas: []float64{1, 0.9, 0.5}, Scheduler: Scheduler{Shift: 1}}}
	if got := turbo.Sigmas(0); len(got) != 4 || got[0] != 1 || got[1] != 0.9 || got[2] != 0.5 || got[3] != 0 {
		t.Errorf("Turbo's sigmas: %v", got)
	}
	// the full model (not distilled): a linear grid, the dynamic shift and the terminal stretched to 0.02
	term := 0.02
	base := &Pipeline{Dit: DitConfig{Width: 1024, Height: 1024, VAEScale: 16},
		Model: Model{Scheduler: Scheduler{Shift: 1, Dynamic: true, BaseSeq: 256, MaxSeq: 8192, BaseShift: 0.5, MaxShift: 0.9, ShiftTerminal: &term}}}
	s := base.Sigmas(40)
	if len(s) != 41 || math.Abs(s[0]-1) > 1e-12 || math.Abs(s[39]-0.02) > 1e-12 || s[40] != 0 {
		t.Fatalf("base sigmas: first %v, last two %v %v (len %d)", s[0], s[39], s[40], len(s))
	}
	for i := 1; i < len(s); i++ {
		if s[i] >= s[i-1] {
			t.Fatalf("sigma %d (%v) is not below %v", i, s[i], s[i-1])
		}
	}
}

func TestRope(t *testing.T) {
	c := DitConfig{Width: 64, Height: 32, VAEScale: 16, HeadDim: 128, Axes: []int{16, 56, 56}, Theta: 10000}
	pos := imagePositions(c, 7)
	if len(pos) != 2*4 {
		t.Fatalf("%d positions", len(pos))
	}
	// the frame at the text's length, height and width centred on zero (QwenImage21Rope)
	if pos[0] != [3]float64{7, -1, -2} || pos[len(pos)-1] != [3]float64{7, 0, 1} {
		t.Errorf("positions %v .. %v", pos[0], pos[len(pos)-1])
	}
	a := ropeAngles(c, pos[:1])
	// pair 0 is the frame axis's frequency 1: angle 7; pair 8 the height's first: angle -1
	if math.Abs(float64(a[0])-math.Cos(7)) > 1e-6 || math.Abs(float64(a[1])-math.Sin(7)) > 1e-6 ||
		math.Abs(float64(a[16])-math.Cos(-1)) > 1e-6 || math.Abs(float64(a[17])-math.Sin(-1)) > 1e-6 {
		t.Errorf("angles %v", a[:18])
	}
}
