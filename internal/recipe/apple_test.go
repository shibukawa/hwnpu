package recipe

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// A model's apple has a version from 1, and its convert holds numbers, strings and lists of them;
// version and convert are a model's alone.
func TestParseRejectsAppleVersions(t *testing.T) {
	head := "version: 1\ntools: {coreai-build: 1}\n"
	for _, c := range []struct{ body, want string }{
		{"models:\n  M:\n    apple: {code: [internal/sfx]}\n", "version"},
		{"models:\n  M:\n    apple: {version: 1, convert: {group: {a: 1}}}\n", "group"},
		{"models:\n  M:\n    apple: {version: 1, convert: {sizes: [[1, 2]]}}\n", "sizes[0]"},
		{"apple: {version: 1}\n", "a model's"},
		{"apple: {convert: {group: 1}}\n", "a model's"},
		{"apple: {prebuilt: true}\n", "a model's"},
	} {
		_, err := Parse("t", []byte(head+c.body))
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%q: %v, want an error of %q", c.body, err, c.want)
		}
	}
	f, err := Parse("t", []byte(head+"apple: {code: [a]}\nmodels:\n  M:\n    apple: {version: 2, code: [b, a], convert: {group: 4, format: lut8, rules: [wo=fp16], sizes: [80, 160], scale: 0.5}}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if f.AppleVersion("M") != 2 || f.AppleVersion("N") != 0 {
		t.Errorf("versions %d %d", f.AppleVersion("M"), f.AppleVersion("N"))
	}
	if got := f.AppleCode("M"); !slices.Equal(got, []string{"a", "b"}) {
		t.Errorf("code %v", got)
	}
	c := f.ConvertOf("M")
	if n, err := c.Int("M", "group"); n != 4 || err != nil {
		t.Errorf("group %d %v", n, err)
	}
	if s, err := c.String("M", "format"); s != "lut8" || err != nil {
		t.Errorf("format %q %v", s, err)
	}
	if s, err := c.Strings("M", "rules"); !slices.Equal(s, []string{"wo=fp16"}) || err != nil {
		t.Errorf("rules %v %v", s, err)
	}
	if s, err := c.Ints("M", "sizes"); !slices.Equal(s, []int{80, 160}) || err != nil {
		t.Errorf("sizes %v %v", s, err)
	}
	if x, err := c.Float("M", "scale"); x != 0.5 || err != nil {
		t.Errorf("scale %v %v", x, err)
	}
	for _, k := range []string{"none", "format"} {
		if _, err := c.Int("M", k); err == nil || !strings.Contains(err.Error(), "models.M.apple.convert") {
			t.Errorf("Int(%s): %v", k, err)
		}
	}
	// the fingerprint is of convert alone, its keys in order
	g, _ := Parse("t", []byte(head+"models:\n  M:\n    apple: {version: 3, archs: [h13g], convert: {scale: 0.5, sizes: [80, 160], rules: [wo=fp16], format: lut8, group: 4}}\n"))
	if f.AppleFingerprint("M") != g.AppleFingerprint("M") {
		t.Error("the fingerprint changed with the version, the archs or the keys' order")
	}
}

// The code's hash is of what it does: not of its comments or its layout, and of its tests not.
func TestCodeHash(t *testing.T) {
	dir := t.TempDir()
	write := func(name, src string) {
		t.Helper()
		if err := writeFile(dir+"/p/"+name, src); err != nil {
			t.Fatal(err)
		}
	}
	write("a.go", "package p\n\n// A is one.\nconst A = 1\n")
	h1, err := CodeHash(dir, []string{"p"})
	if err != nil {
		t.Fatal(err)
	}
	write("a.go", "package p\n\n// A is one, as said.\nconst   A = 1\n")
	write("a_test.go", "package p\n\nconst B = 2\n")
	if h2, _ := CodeHash(dir, []string{"p"}); h2 != h1 {
		t.Error("a comment, the layout or a test changed the hash")
	}
	write("a.go", "package p\n\nconst A = 2\n")
	h3, _ := CodeHash(dir, []string{"p"})
	if h3 == h1 {
		t.Error("a value changed and the hash did not")
	}
	if h4, err := CodeHash(dir, []string{"p/a.go"}); err != nil || h4 == "" {
		t.Errorf("one file: %q %v", h4, err)
	}
	write("run.go", "package p\n\nconst R = 1\n")
	if h5, _ := CodeHash(dir, []string{"p", "!p/run.go"}); h5 != h3 {
		t.Error("a file left out (!) changed the hash")
	}
	if _, err := CodeHash(dir, []string{"q"}); err == nil {
		t.Error("a package with no Go files: no error")
	}
}

func writeFile(name, src string) error {
	if err := os.MkdirAll(filepath.Dir(name), 0o755); err != nil {
		return err
	}
	return os.WriteFile(name, []byte(src), 0o644)
}
