package supertonic

import (
	"math"
	"testing"
)

// TestHostSin lays the step's embedding as the graph's Concat does (sin then cos, the batch
// twice) and the rotary embedding of positions of a padded sequence by its own length.
func TestHostSin(t *testing.T) {
	h := hostSin{Scale: 1000, Freqs: make([]float64, ropeDim), Theta: make([]float64, ropeDim)}
	for j := range ropeDim {
		h.Freqs[j] = math.Pow(1e-4, float64(j)/(ropeDim-1))
		h.Theta[j] = 10 * math.Pow(1.0/7500, float64(j)/(ropeDim-1))
	}
	tm := h.time(3, 8)
	if len(tm) != 2*2*ropeDim {
		t.Fatalf("time: %d values", len(tm))
	}
	for b := range 2 {
		for j := range ropeDim {
			a := 3.0 / 8 * 1000 * h.Freqs[j]
			if s, c := tm[b*64+j], tm[b*64+32+j]; math.Abs(float64(s)-math.Sin(a)) > 1e-6 || math.Abs(float64(c)-math.Cos(a)) > 1e-6 {
				t.Fatalf("time[%d][%d]: %g %g, want %g %g", b, j, s, c, math.Sin(a), math.Cos(a))
			}
		}
	}
	sin, cos := h.rope(5, 8) // 5 positions padded to 8
	if len(sin) != 2*8*ropeDim || len(cos) != len(sin) {
		t.Fatalf("rope: %d %d values", len(sin), len(cos))
	}
	for b := range 2 {
		for i := range 8 {
			for j := range ropeDim {
				a := float64(i) / 5 * h.Theta[j]
				k := (b*8+i)*ropeDim + j
				if math.Abs(float64(sin[k])-math.Sin(a)) > 1e-6 || math.Abs(float64(cos[k])-math.Cos(a)) > 1e-6 {
					t.Fatalf("rope[%d][%d][%d]", b, i, j)
				}
			}
		}
	}
}

func TestPadMask(t *testing.T) {
	got := pad([]float32{1, 2, 3, 4, 5, 6}, 2, 3, 5)
	want := []float32{1, 2, 3, 0, 0, 4, 5, 6, 0, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("pad: %v", got)
		}
	}
	if m := mask(2, 4); m[0] != 1 || m[1] != 1 || m[2] != 0 || m[3] != 0 {
		t.Fatalf("mask: %v", m)
	}
	s := &npuStages{progs: []npuSize{{T: 64, L: 32}, {T: 128, L: 64}, {T: 256, L: 128}}}
	for _, c := range []struct{ T, L, want int }{{10, 10, 32}, {70, 10, 64}, {10, 40, 64}, {200, 100, 128}, {300, 10, 0}, {10, 200, 0}} {
		p := s.size(c.T, c.L)
		if got := 0; p != nil {
			got = p.L
			if got != c.want {
				t.Errorf("size(%d, %d): %d, want %d", c.T, c.L, got, c.want)
			}
		} else if c.want != 0 {
			t.Errorf("size(%d, %d): none, want %d", c.T, c.L, c.want)
		}
	}
}
