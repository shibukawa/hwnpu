package main

import "testing"

// TestSize: an aspect ratio name or a pixel count (a square) gives the picture's width and height;
// side takes the squares alone.
func TestSize(t *testing.T) {
	ok := []struct {
		size  string
		w, h  int
		sside int // side's result, 0 for a non-square
	}{
		{"1:1", 1024, 1024, 1024},
		{"1:1s", 512, 512, 512},
		{"3:2", 1248, 832, 0},
		{"16:9", 1360, 752, 0},
		{"9:16", 752, 1360, 0},
		{"512", 512, 512, 512},
	}
	for _, c := range ok {
		f := convertFlags{Size: c.size}
		w, h, err := f.size()
		if err != nil || w != c.w || h != c.h {
			t.Errorf("size(%s) = %d, %d, %v; want %d, %d", c.size, w, h, err, c.w, c.h)
		}
		side, err := f.side("M")
		if c.sside == 0 {
			if err == nil {
				t.Errorf("side(%s) = %d, want an error (not a square)", c.size, side)
			}
		} else if err != nil || side != c.sside {
			t.Errorf("side(%s) = %d, %v; want %d", c.size, side, err, c.sside)
		}
	}
	for _, s := range []string{"1168x880", "0", "2:4", "-512"} {
		if _, _, err := (convertFlags{Size: s}).size(); err == nil {
			t.Errorf("size(%s) = nil error, want an error", s)
		}
	}
}
