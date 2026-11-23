package recipe

import (
	"os"
	"strconv"
	"strings"
	"testing"
)

const appleVersionsFile = "testdata/apple-versions.txt"

// A model's Core AI conversion, its recipe's apple.convert or the Go code that writes its
// programs (apple.code), changed under the same version would be taken for the exports of before
// (their names stay): testdata/apple-versions.txt holds what each version of each recipe is (its
// AppleFingerprint and the CodeHash of its code), and a change asks for a new version. go test
// ./internal/recipe -run TestAppleVersions -update writes it after the version is raised, or
// after the exports were made again and came out byte for byte the same (docs/model-recipes.md).
func TestAppleVersions(t *testing.T) {
	f, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	var lines []string
	now := map[string][]string{}
	for _, name := range f.sortedModels() {
		v := f.AppleVersion(name)
		if v == 0 {
			continue
		}
		code, err := CodeHash("../..", f.AppleCode(name))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		row := []string{name, itoa(v), f.AppleFingerprint(name), code}
		lines = append(lines, strings.Join(row, " "))
		now[name] = row
	}
	if *update {
		body := "# model, version of its recipe for Core AI (model-recipes.yaml apple.version), AppleFingerprint, CodeHash: written by\n# go test ./internal/recipe -run TestAppleVersions -update\n" + strings.Join(lines, "\n") + "\n"
		if err := os.WriteFile(appleVersionsFile, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	raw, err := os.ReadFile(appleVersionsFile)
	if err != nil {
		t.Fatal(err)
	}
	was := map[string][]string{}
	for _, l := range strings.Split(string(raw), "\n") {
		if f := strings.Fields(l); len(f) == 4 && !strings.HasPrefix(l, "#") {
			was[f[0]] = f
		}
	}
	const bump = "raise models.%s.apple.version (or make its exports again and see them the same), then go test ./internal/recipe -run TestAppleVersions -update"
	for _, name := range f.sortedModels() {
		got, ok := now[name]
		if !ok {
			continue
		}
		w, ok := was[name]
		switch {
		case !ok:
			t.Errorf("%s: no line in %s (go test ./internal/recipe -run TestAppleVersions -update)", name, appleVersionsFile)
		case w[1] == got[1] && w[2] != got[2]:
			t.Errorf("%s: its apple.convert changed under version %s: "+bump, name, got[1], name)
		case w[1] == got[1] && w[3] != got[3]:
			t.Errorf("%s: the code that writes its programs (%s) changed under version %s: "+bump, name, strings.Join(f.AppleCode(name), " "), got[1], name)
		case w[1] != got[1] && atoi(w[1]) >= atoi(got[1]):
			t.Errorf("%s: version %s, was %s: a recipe's version only goes up", name, got[1], w[1])
		case w[1] != got[1]:
			t.Errorf("%s: version %s is new: go test ./internal/recipe -run TestAppleVersions -update", name, got[1])
		}
	}
}

func itoa(n int) string { return strconv.Itoa(n) }
