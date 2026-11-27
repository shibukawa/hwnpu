package engine

import "testing"

func TestHalve(t *testing.T) {
	for _, c := range []struct{ in, a, b string }{
		{"総額は過去最大となり、防衛費と社会保障費が大きく増えています。", "総額は過去最大となり、", "防衛費と社会保障費が大きく増えています。"},
		{"あいうえおかきくけこ", "あいうえお", "かきくけこ"},
		{"one two three four", "one two", "three four"},
	} {
		a, b, ok := halve(c.in)
		if !ok || a != c.a || b != c.b {
			t.Errorf("halve(%q): %q %q %v", c.in, a, b, ok)
		}
	}
	if _, _, ok := halve("あい"); ok {
		t.Error("a text of 2 cut")
	}
}

func TestSupertonicLang(t *testing.T) {
	for in, want := range map[string]string{"ja": "ja", "en-GB": "en", "Japanese": "ja", "": "en", "xx": "na"} {
		if got := supertonicLang(in); got != want {
			t.Errorf("supertonicLang(%q) = %q, want %q", in, got, want)
		}
	}
}
