package aspect

import "testing"

// Each aspect ratio's name gives its size, of the model's multiple, and back.
func TestParse(t *testing.T) {
	for _, m := range []int{16, 32} {
		for _, a := range Of(m) {
			w, h, err := Parse(a.Name, m)
			if err != nil || w != a.Width || h != a.Height {
				t.Errorf("Parse(%q, %d) = %dx%d, %v", a.Name, m, w, h, err)
			}
			if w%m != 0 || h%m != 0 {
				t.Errorf("%s at %d: %dx%d is no multiple", a.Name, m, w, h)
			}
			if got := Name(w, h, m); got != a.Name {
				t.Errorf("Name(%d, %d, %d) = %q, want %q", w, h, m, got, a.Name)
			}
			// the latents' (h/8) x (w/8) within 16384 pixels: a VAE's middle attention
			if w/8*(h/8) > 16384 {
				t.Errorf("%s at %d: %dx%d's latents are %d pixels", a.Name, m, w, h, w/8*(h/8))
			}
		}
	}
	if w, h, err := ParseOrSide("768", 16); err != nil || w != 768 || h != 768 {
		t.Errorf("ParseOrSide(768) = %dx%d, %v", w, h, err)
	}
	for _, bad := range []string{"1024x1024", "5:4", ""} {
		if _, _, err := ParseOrSide(bad, 16); err == nil {
			t.Errorf("ParseOrSide(%q) accepted", bad)
		}
	}
}
