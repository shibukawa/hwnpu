package segment

import (
	"encoding/binary"
	"math"
	"testing"
)

// TestResize checks the half-pixel bilinear stretch: a constant stays, and a 2 x 1 ramp's
// doubling clamps at the edges (outputs at -0.25, 0.25, 0.75, 1.25 of the input).
func TestResize(t *testing.T) {
	px := []float32{0, 1, 0, 1, 0, 1} // 3 channels of 2 x 1
	got := Resize(px, 2, 1, 4, 1)
	want := []float32{0, 0.25, 0.75, 1}
	for i, v := range want {
		if math.Abs(float64(got[i]-v)) > 1e-6 {
			t.Fatalf("Resize = %v, want %v in each channel", got[:4], want)
		}
	}
}

// TestPatchRows checks a patch row's order (channel, row, column) and the normalization.
func TestPatchRows(t *testing.T) {
	px := make([]float32, 3*Size*Size)
	// patch (1, 2): channel 2's pixel (row 3, column 5) of it
	px[(2*Size+16+3)*Size+32+5] = 1
	rows := PatchRows(px)
	at := func(r, k int) float32 {
		return math.Float32frombits(uint32(binary.LittleEndian.Uint16(rows[2*(r*768+k):])) << 16)
	}
	r, k := 1*Grid+2, 2*256+3*16+5
	if v, want := at(r, k), (1-mean[2])/std[2]; math.Abs(float64(v-want)) > 0.01 {
		t.Errorf("patch row %d value %d = %v, want %v", r, k, v, want)
	}
	if v, want := at(r, k-1), -mean[2]/std[2]; math.Abs(float64(v-want)) > 0.01 {
		t.Errorf("its neighbour = %v, want %v", v, want)
	}
}

func TestMaskBox(t *testing.T) {
	k := Mask{W: 4, H: 3, In: make([]bool, 12)}
	if k.Box() != [4]int{} || k.Area() != 0 {
		t.Errorf("an empty mask's box %v, area %d", k.Box(), k.Area())
	}
	k.In[1*4+1], k.In[2*4+2] = true, true
	if k.Box() != [4]int{1, 1, 3, 3} || k.Area() != 2 {
		t.Errorf("box %v, area %d", k.Box(), k.Area())
	}
}
