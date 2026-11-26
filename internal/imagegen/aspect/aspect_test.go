package aspect

import "testing"

// Each aspect ratio's name gives its size, a multiple of 16, and back.
func TestParse(t *testing.T) {
	for _, a := range All {
		w, h, err := Parse(a.Name)
		if err != nil || w != a.Width || h != a.Height {
			t.Errorf("Parse(%q) = %dx%d, %v", a.Name, w, h, err)
		}
		if w%16 != 0 || h%16 != 0 {
			t.Errorf("%s: %dx%d is no multiple of 16", a.Name, w, h)
		}
		if got := Name(w, h); got != a.Name {
			t.Errorf("Name(%d, %d) = %q, want %q", w, h, got, a.Name)
		}
		// the latents' (h/8) x (w/8) within 16384 pixels: a VAE's middle attention
		if w/8*(h/8) > 16384 {
			t.Errorf("%s: %dx%d's latents are %d pixels", a.Name, w, h, w/8*(h/8))
		}
	}
	if w, h, err := ParseOrSide("768", 16); err != nil || w != 768 || h != 768 {
		t.Errorf("ParseOrSide(768) = %dx%d, %v", w, h, err)
	}
	for _, bad := range []string{"1000x1000", "5:4", ""} {
		if _, _, err := ParseOrSide(bad, 16); err == nil {
			t.Errorf("ParseOrSide(%q) accepted", bad)
		}
	}
}

// TestParseOrSideWxH: WIDTHxHEIGHT in multiples of the model's pixels.
func TestParseOrSideWxH(t *testing.T) {
	if w, h, err := ParseOrSide("768x1152", 16); err != nil || w != 768 || h != 1152 {
		t.Fatalf("768x1152: %d %d %v", w, h, err)
	}
	if _, _, err := ParseOrSide("770x1152", 16); err == nil {
		t.Fatal("770x1152: no error")
	}
	if w, h, err := ParseOrSide("2:3", 16); err != nil || w != 832 || h != 1248 {
		t.Fatalf("2:3: %d %d %v", w, h, err)
	}
}

// TestWith: a model's own sizes of some ratios, the others and All as they were.
func TestWith(t *testing.T) {
	as := With(All, Aspect{"16:9", 1344, 768})
	if w, h, err := ParseOrSideIn("16:9", as, 16); err != nil || w != 1344 || h != 768 {
		t.Fatalf("16:9: %d %d %v", w, h, err)
	}
	if w, h, err := ParseOrSideIn("2:3", as, 16); err != nil || w != 832 || h != 1248 {
		t.Fatalf("2:3: %d %d %v", w, h, err)
	}
	if w, h, _ := Parse("16:9"); w != 1360 || h != 752 {
		t.Fatalf("All's 16:9 changed: %d %d", w, h)
	}
}

func TestTag(t *testing.T) {
	for _, c := range []struct {
		sizes [][2]int
		want  string
	}{
		{[][2]int{{880, 1168}}, "880x1168"},
		{[][2]int{{512, 512}, {1024, 1024}}, "512+1024"},
		{[][2]int{{1024, 768}, {1024, 1024}}, "1024x768+1024"},
	} {
		if got := Tag(c.sizes); got != c.want {
			t.Errorf("Tag(%v) = %q, want %q", c.sizes, got, c.want)
		}
	}
	if a := Tag(SizesOf(All)); len(a) != len("8sizes-")+8 || a[:7] != "8sizes-" {
		t.Errorf("Tag of All %q", a)
	}
}
