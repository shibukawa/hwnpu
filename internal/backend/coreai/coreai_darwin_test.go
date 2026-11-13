package coreai

import "testing"

func TestSplitFunction(t *testing.T) {
	for _, c := range []struct{ in, path, fn string }{
		{"/a/m.aimodel", "/a/m.aimodel", "main"},
		{"/a/m.aimodel#f128", "/a/m.aimodel", "f128"},
		{"/a/m.aimodel#", "/a/m.aimodel#", "main"},
		{"/a/x#y.mlpackage", "/a/x#y.mlpackage", "main"},
		{"/a/x#y.mlpackage#f8", "/a/x#y.mlpackage#f8", "main"},
	} {
		if p, fn := splitFunction(c.in); p != c.path || fn != c.fn {
			t.Errorf("splitFunction(%q) = %q, %q, want %q, %q", c.in, p, fn, c.path, c.fn)
		}
	}
}

func TestFullyPlaced(t *testing.T) {
	// the shape of a Core AI cache entry's MPSGraph manifest (Entry Function Attributes)
	two := []byte(`<dict>
	<key>f128_VnUU9jg5arAyGObDMTSq7g_0</key>
	<array><string>mps.aneAlignedIO</string></array>
	<key>f8_Sfb_LWKqZ7kLACNxv16tNg_1</key>
	<array><string>mps.fullyPlacedOnANE</string></array>
</dict>`)
	if !fullyPlaced(two, "f8") {
		t.Error("f8 is marked fully placed")
	}
	if fullyPlaced(two, "f128") {
		t.Error("f128 is not marked")
	}
	one := []byte(`<string>mps.fullyPlacedOnANE</string>`)
	if !fullyPlaced(one, "main") {
		t.Error("a manifest without entries: the mark anywhere")
	}
}
