package mil

import (
	"math"
	"testing"
)

func TestRelativeBuckets(t *testing.T) {
	// transformers' T5Attention._relative_position_bucket(k - q, bidirectional, 32, 128)
	want := map[[2]int]int{{0, 0}: 0, {0, 1}: 17, {1, 0}: 1, {0, 7}: 23, {0, 8}: 24, {0, 20}: 26, {0, 100}: 31,
		{0, 511}: 31, {511, 0}: 15, {300, 250}: 13, {10, 3}: 7}
	b := RelativeBuckets(512, 32, 128)
	for qk, w := range want {
		if got := b[qk[0]*512+qk[1]]; got != w {
			t.Errorf("bucket of (%d, %d) = %d, want %d", qk[0], qk[1], got, w)
		}
	}
}

func TestRopeValues(t *testing.T) {
	c := Dit{Width: 64, Height: 32, HeadDim: 128, Axes: []int{16, 56, 56}, TextRows: 3}
	cos, sin := RopeValues(c)
	if len(cos) != c.Rows()*128 {
		t.Fatalf("%d values for %d rows", len(cos), c.Rows())
	}
	// row 5: patch (1, 1) of a 2 x 4 grid; pair 8 is the row axis' first frequency (1), pair 36
	// the column axis' first, pair 9 the row axis' second (10000^(-2/56))
	r := 5
	at := func(v []float32, i int) float64 { return float64(v[r*128+i]) }
	if math.Abs(at(cos, 8)-math.Cos(1)) > 1e-6 || math.Abs(at(sin, 8+64)-math.Sin(1)) > 1e-6 {
		t.Errorf("row pair: cos %v sin %v", at(cos, 8), at(sin, 72))
	}
	if math.Abs(at(cos, 36)-math.Cos(1)) > 1e-6 || math.Abs(at(sin, 9)-math.Sin(math.Pow(10000, -2.0/56))) > 1e-6 {
		t.Errorf("column pair %v, second row frequency %v", at(cos, 36), at(sin, 9))
	}
	if at(cos, 0) != 1 || at(sin, 0) != 0 {
		t.Error("the first axis (the frame) turns")
	}
	// the text's rows do not turn
	for i := 0; i < 128; i++ {
		if cos[8*128+i] != 1 || sin[8*128+i] != 0 {
			t.Fatal("a text row turns")
		}
	}
}

func TestPermuteRows(t *testing.T) {
	// two heads of 4 rows of 2: new row i of a head is old 2i, new i + 2 old 2i + 1
	v := []float32{0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7}
	got := PermuteRows(v, 8, 4)
	want := []float32{0, 0, 2, 2, 1, 1, 3, 3, 4, 4, 6, 6, 5, 5, 7, 7}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%v, want %v", got, want)
		}
	}
}

func TestGroups(t *testing.T) {
	c := Dit{Doubles: 19, Singles: 38}
	if g := DoubleGroups(c, 4); len(g) != 5 || g[4] != [2]int{16, 3} {
		t.Errorf("doubles %v", g)
	}
	if g := SingleGroups(c, 8); len(g) != 5 || g[0] != [2]int{0, 8} || g[4] != [2]int{31, 7} {
		t.Errorf("singles %v", g)
	}
	if c.Mods() != 344 {
		t.Errorf("%d modulation rows", c.Mods())
	}
}

func TestVaeStages(t *testing.T) {
	v := Vae{Channels: []int{128, 256, 512, 512}, Layers: 2, Groups: 32, Latent: 16}
	s := VaeStages(v, 512, 512)
	if len(s) != 10 {
		t.Fatalf("%d stages", len(s))
	}
	// the mid block at 64 x 64, the last up block at full size, 256 to 128 channels first
	if s[0].C != 16 || s[0].H != 64 || s[1].C != 512 || s[2].H != 128 {
		t.Errorf("stages 0 to 2: %+v %+v %+v", s[0], s[1], s[2])
	}
	if s[6].C != 256 || s[6].H != 512 || s[6].Ops[0].Out != 128 || s[9].Ops[0].Kind != "head" || s[9].C != 128 {
		t.Errorf("stages 6, 9: %+v %+v", s[6], s[9])
	}
}
