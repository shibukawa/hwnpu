package hwnpu

import (
	"math"
	"testing"
)

func TestF16RoundTrip(t *testing.T) {
	cases := []struct {
		in   float32
		bits uint16
	}{
		{0, 0x0000},
		{1, 0x3c00},
		{-2, 0xc000},
		{65504, 0x7bff},
		{1e6, 0x7c00},                       // overflow to +Inf
		{float32(math.Pow(2, -24)), 0x0001}, // smallest subnormal
		{0.333251953125, 0x3555},
	}
	for _, c := range cases {
		if got := f32ToF16(c.in); got != c.bits {
			t.Errorf("f32ToF16(%v) = %#04x, want %#04x", c.in, got, c.bits)
		}
		if c.bits != 0x7c00 {
			if got := f16ToF32(c.bits); got != c.in {
				t.Errorf("f16ToF32(%#04x) = %v, want %v", c.bits, got, c.in)
			}
		}
	}
	for h := 0; h < 0x7c00; h++ { // every finite positive half survives a round trip
		if got := f32ToF16(f16ToF32(uint16(h))); got != uint16(h) {
			t.Fatalf("round trip of %#04x gave %#04x", h, got)
		}
	}
	if !math.IsNaN(float64(f16ToF32(f32ToF16(float32(math.NaN()))))) {
		t.Error("NaN not preserved")
	}
}
