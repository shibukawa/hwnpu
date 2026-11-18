package mlpackage

import (
	"slices"
	"testing"
)

// Neighbouring equal tables are made to differ by one ulp of their largest entry, in a copy;
// tables that differ, or equal ones of different column blocks, are left alone.
func TestDistinctNeighbours(t *testing.T) {
	a := []uint16{0x3c00, 0xbc00, 0x3800, 0x0000} // 1, -1, 0.5, 0
	b := []uint16{0x3c00, 0x3c00, 0x3c00, 0x3c00}
	// 3 groups x 2 column blocks of 4 entries: group 1 repeats group 0's block 0, group 2 repeats
	// group 1's block 1; group 1's block 1 equals group 0's block 0 (another block: left alone)
	lut := slices.Concat(a, b, a, a, b, a)
	in := slices.Clone(lut)
	out := DistinctNeighbours(lut, 3, 2, 4)
	if !slices.Equal(lut, in) {
		t.Fatal("the input was changed")
	}
	want := slices.Concat(a, b, []uint16{0x3bff, 0xbc00, 0x3800, 0x0000}, a, b, []uint16{0x3bff, 0xbc00, 0x3800, 0x0000})
	if !slices.Equal(out, want) {
		t.Fatalf("got % x\nwant % x", out, want)
	}
	// a run of three: each differs from the one before
	r := DistinctNeighbours(slices.Concat(a, a, a), 3, 1, 4)
	for g := 1; g < 3; g++ {
		if slices.Equal(r[g*4:g*4+4], r[g*4-4:g*4]) {
			t.Fatalf("groups %d and %d equal: % x", g-1, g, r)
		}
	}
	// zeros get the smallest subnormal
	z := DistinctNeighbours(make([]uint16, 8), 2, 1, 4)
	if !slices.Equal(z, []uint16{0, 0, 0, 0, 1, 0, 0, 0}) {
		t.Fatalf("zeros: % x", z)
	}
	// nothing equal: the same slice back
	d := slices.Concat(a, b)
	if got := DistinctNeighbours(d, 2, 1, 4); &got[0] != &d[0] {
		t.Fatal("copied with nothing to change")
	}
}
