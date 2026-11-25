package chat

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/shibukawa/bonsaiai"
	"github.com/shibukawa/bonsaiai/internal/cache"
	"github.com/shibukawa/bonsaiai/internal/recipe"
)

// The built-in recipes name models of models.yaml, and every model with an export of
// iron_export_lm.py has its arguments (AMDExport) from its recipe.
func TestRecipesNameModels(t *testing.T) {
	f, err := recipe.Builtin()
	if err != nil {
		t.Fatal(err)
	}
	for name, r := range f.Models {
		m, ok := Lookup(name)
		if !ok {
			t.Errorf("%s is not a model of models.yaml", name)
			continue
		}
		if m.Recipe != name {
			t.Errorf("%s: Recipe %q", name, m.Recipe)
		}
		if r.AMD != nil && r.AMD.Script == LMTool && len(m.AMDExport) == 0 {
			t.Errorf("%s: no AMDExport from its recipe", name)
		}
		if r.AMD != nil {
			for gb := range r.AMD.Variants {
				if _, ok := m.Variants[gb]; !ok {
					t.Errorf("%s: the recipe's variant of %d GB is not one of models.yaml", name, gb)
				}
			}
		}
	}
}

// A model like another is made by its recipe: the same arguments of iron_export_lm.py, of its
// variants too.
func TestRecipeOfLike(t *testing.T) {
	if _, err := userFiles(t, "version: 1\nmodels:\n  my-gptoss:\n    like: gpt-oss-20b\n    tag: mine\n"); err != nil {
		t.Fatal(err)
	}
	base, _ := Lookup("gpt-oss-20b")
	m, ok := Lookup("my-gptoss")
	if !ok || m.Recipe != "gpt-oss-20b" || !slices.Equal(m.AMDExport, base.AMDExport) ||
		!slices.Equal(m.On(24).AMDExport, base.On(24).AMDExport) || slices.Equal(m.On(24).AMDExport, m.AMDExport) {
		t.Errorf("like gpt-oss-20b: recipe %q, %v, on 24 GB %v", m.Recipe, m.AMDExport, m.On(24).AMDExport)
	}
}

func TestLMArgs(t *testing.T) {
	if got, err := lmArgs([]string{"lfm3", recipe.In, recipe.Out, "--q4"}); err != nil || !slices.Equal(got, []string{"lfm3", "--q4"}) {
		t.Errorf("%v %v", got, err)
	}
	for _, bad := range [][]string{{"lfm3", recipe.Out, recipe.In}, {"lfm3", recipe.In, recipe.Out, recipe.Sizes}, {recipe.In, recipe.Out}} {
		if _, err := lmArgs(bad); err == nil {
			t.Errorf("%v: no error", bad)
		}
	}
}

// The built-in models.yaml gives no version of amd-xdna2: model-recipes.yaml's recipes do.
func TestNoAMDVersionInModels(t *testing.T) {
	for i, l := range strings.Split(string(bonsaiai.ModelsYAML), "\n") {
		if l = strings.TrimSpace(l); !strings.HasPrefix(l, "#") && strings.HasPrefix(l, "targets:") && strings.Contains(l, cache.AMDTarget) {
			t.Errorf("models.yaml:%d gives a version of amd-xdna2: write it in model-recipes.yaml (models.NAME.amd.version)", i+1)
		}
	}
	f, _ := recipe.Builtin()
	for _, m := range Catalog {
		if r, ok := f.AMDOf(m.Recipe); ok && m.Targets[cache.AMDTarget] != r.Version {
			t.Errorf("%s: version %d, its recipe's %d", m.Name, m.Targets[cache.AMDTarget], r.Version)
		}
	}
}

// Every pin is of its model's version: raising a recipe's version asks for a pin of it.
func TestPinsOfVersion(t *testing.T) {
	for _, m := range Catalog {
		for target, pins := range m.Pins {
			for export, p := range pins {
				if !p.OfVersion(m.Targets[target]) {
					t.Errorf("%s: %s %s is pinned at %s, of no version %d (bonsaiai-dev models package %s/%s, then bonsaiai-dev models pin %s)",
						m.Name, target, export, p.At, m.Targets[target], m.Name, target, m.Name)
				}
			}
		}
	}
	for _, c := range []struct {
		at   string
		v    int
		want bool
	}{{"v1", 1, true}, {"v0-1a2b3c4d", 1, true}, {"v0-1a2b3c4d", 0, true}, {"v1", 2, false}, {"v0-1a2b3c4d", 2, false}, {"v2-1a2b3c4d", 2, true}, {"v12", 12, true}} {
		if got := (Pin{At: c.at}).OfVersion(c.v); got != c.want {
			t.Errorf("%s of version %d: %v", c.at, c.v, got)
		}
	}
}

// A pin of the version before a raise is not taken: the version's export is made or fetched.
func TestPinOfOldVersionNotTaken(t *testing.T) {
	base, _ := Lookup("Qwen3.5-0.8B")
	export := "Qwen3.5-0.8B.amd-npu2-q6"
	base.Pins = map[string]map[string]Pin{cache.AMDTarget: {export: {At: fmt.Sprintf("v%d", base.Targets[cache.AMDTarget]), Digest: "sha256:" + strings.Repeat("0", 64)}}}
	p := placeOf("iron-lm", base, cache.AMDTarget, export, map[string]string{"model": base.Name})
	if p.pin == nil || p.pin.At != base.Pins[cache.AMDTarget][export].At {
		t.Fatalf("version %d: pin %+v not taken", base.Targets[cache.AMDTarget], p.pin)
	}
	raised := base
	raised.Targets = map[string]int{cache.AMDTarget: base.Targets[cache.AMDTarget] + 1}
	if p := placeOf("iron-lm", raised, cache.AMDTarget, export, map[string]string{"model": base.Name}); p.pin != nil || !strings.HasSuffix(p.entry.Path, fmt.Sprintf("/v%d", raised.Targets[cache.AMDTarget])) {
		t.Errorf("raised to %d: pin %+v, path %s", raised.Targets[cache.AMDTarget], p.pin, p.entry.Path)
	}
}
