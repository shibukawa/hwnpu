package experts

import (
	"path/filepath"
	"testing"
)

func TestUsageFile(t *testing.T) {
	p := filepath.Join(t.TempDir(), "u.bin")
	u := NewUsage(2, 4)
	u.Add(1, [][]int{{0, 3}, {3, 2}})
	u.Add(5, [][]int{{0}}) // outside: ignored
	for range 2 {
		if err := u.AddTo(p); err != nil {
			t.Fatal(err)
		}
	}
	v, err := ReadUsage(p)
	if err != nil {
		t.Fatal(err)
	}
	want := []uint32{0, 0, 0, 0, 2, 0, 2, 4}
	for i, c := range v.Counts {
		if c != want[i] {
			t.Fatalf("counts %v, want %v", v.Counts, want)
		}
	}
	big := NewUsage(2, 4)
	big.Counts[7] = usageCap
	if err := v.Merge(big); err != nil {
		t.Fatal(err)
	}
	if v.Counts[7] != (usageCap+4)>>1 || v.Counts[4] != 1 {
		t.Fatalf("halved counts %v", v.Counts)
	}
}
