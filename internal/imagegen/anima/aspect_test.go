package anima

import "testing"

// TestParseAspect: each aspect ratio name gives its pixel size, and pixels are refused.
func TestParseAspect(t *testing.T) {
	want := map[string][2]int{
		"1:1": {1024, 1024}, "1:1s": {512, 512}, "2:3": {832, 1248}, "3:4": {880, 1168},
		"9:16": {752, 1360}, "3:2": {1248, 832}, "4:3": {1168, 880}, "16:9": {1360, 752},
	}
	for name, wh := range want {
		w, h, err := ParseAspect(name)
		if err != nil || w != wh[0] || h != wh[1] {
			t.Errorf("ParseAspect(%q) = %dx%d, %v; want %dx%d", name, w, h, err, wh[0], wh[1])
		}
		if got := AspectName(wh[0], wh[1]); got != name {
			t.Errorf("AspectName(%dx%d) = %q, want %q", wh[0], wh[1], got, name)
		}
	}
	for _, bad := range []string{"1024", "1024x1024", "5:4", ""} {
		if _, _, err := ParseAspect(bad); err == nil {
			t.Errorf("ParseAspect(%q) accepted, want an error", bad)
		}
	}
}
