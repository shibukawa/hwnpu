package chroma

import (
	"math"
	"testing"
)

func TestShiftedSigmas(t *testing.T) {
	// a linear grid through Chroma1-HD's shift 3: 3 s / (1 + 2 s)
	want := []float64{1, 0.9, 0.75, 0.5, 0}
	got := ShiftedSigmas(4, 3)
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("sigmas %v, want %v", got, want)
		}
	}
	if s := Sigmas(4); s[1] != 0.75 || s[4] != 0 {
		t.Errorf("unshifted sigmas %v", s)
	}
}

func TestModInput(t *testing.T) {
	p := &Pipeline{C: Config{Mods: 3, ModRows: 4, ApproxIn: 64}}
	x := p.modInput(500)
	if len(x) != 4*64 {
		t.Fatalf("%d values", len(x))
	}
	// each row: the timestep's 16, a zero guidance's 16 (cosines 1, sines 0), its index's 32
	if x[0] != float32(math.Cos(500)) || x[16] != 1 || x[24] != 0 || x[64+32] != float32(math.Cos(1000)) {
		t.Errorf("row 0 %v, row 1's index %v", x[:32], x[64+32])
	}
	for _, v := range x[3*64:] {
		if v != 0 {
			t.Fatal("the padding row is not zero")
		}
	}
}
