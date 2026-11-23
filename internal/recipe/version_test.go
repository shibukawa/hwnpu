package recipe

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
)

var update = flag.Bool("update", false, "write testdata/amd-versions.txt from the recipes")

const versionsFile = "testdata/amd-versions.txt"

// A model's recipe for AMD's NPU, or the files of tools/amd its script reads, changed under the
// same version would be taken for the exports of before: testdata/amd-versions.txt holds what
// each version of each recipe is (its Fingerprint and ToolsHash), and a change asks for a new
// version. go test ./internal/recipe -run TestAMDVersions -update writes it after the version
// is raised.
func TestAMDVersions(t *testing.T) {
	f, err := Builtin()
	if err != nil {
		t.Fatal(err)
	}
	var lines []string
	now := map[string][]string{}
	for _, name := range f.sortedModels() {
		m, ok := f.AMDOf(name)
		if !ok {
			continue
		}
		tools, err := m.ToolsHash()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		row := []string{name, fmt.Sprint(m.Version), m.Fingerprint(), tools}
		lines = append(lines, strings.Join(row, " "))
		now[name] = row
	}
	if *update {
		body := "# model, version of its recipe for AMD's NPU (model-recipes.yaml), Fingerprint, ToolsHash: written by\n# go test ./internal/recipe -run TestAMDVersions -update\n" + strings.Join(lines, "\n") + "\n"
		if err := os.WriteFile(versionsFile, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	raw, err := os.ReadFile(versionsFile)
	if err != nil {
		t.Fatal(err)
	}
	was := map[string][]string{}
	for _, l := range strings.Split(string(raw), "\n") {
		if f := strings.Fields(l); len(f) == 4 && !strings.HasPrefix(l, "#") {
			was[f[0]] = f
		}
	}
	const bump = "raise models.%s.amd.version, then go test ./internal/recipe -run TestAMDVersions -update"
	for _, name := range f.sortedModels() {
		got, ok := now[name]
		if !ok {
			continue
		}
		w, ok := was[name]
		switch {
		case !ok:
			t.Errorf("%s: no line in %s (go test ./internal/recipe -run TestAMDVersions -update)", name, versionsFile)
		case w[1] == got[1] && w[2] != got[2]:
			t.Errorf("%s: its recipe for AMD's NPU changed under version %s: "+bump, name, got[1], name)
		case w[1] == got[1] && w[3] != got[3]:
			t.Errorf("%s: the files of tools/amd %s reads changed under version %s: "+bump, name, f.Models[name].AMD.Script, got[1], name)
		case w[1] != got[1] && atoi(w[1]) >= atoi(got[1]):
			t.Errorf("%s: version %s, was %s: a recipe's version only goes up", name, got[1], w[1])
		case w[1] != got[1]:
			t.Errorf("%s: version %s is new: go test ./internal/recipe -run TestAMDVersions -update", name, got[1])
		}
	}
}

func atoi(s string) int { n, _ := strconv.Atoi(s); return n }
