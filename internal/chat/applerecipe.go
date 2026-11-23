package chat

import (
	"fmt"

	"github.com/shibukawa/hwnpu/internal/cache"
	"github.com/shibukawa/hwnpu/internal/llm"
	"github.com/shibukawa/hwnpu/internal/recipe"
	shared "github.com/shibukawa/hwnpu/internal/store"
)

// AppleConvert is the apple.convert of model name's recipe (model-recipes.yaml: its own, or
// that of the model it is like) and the recipe's model, which errors of its values name: what
// its Core AI conversion is made with. It fails when the model has no recipe.
func AppleConvert(name string) (recipe.Convert, string, error) {
	f, err := recipe.Builtin()
	if err != nil {
		return nil, "", err
	}
	by := name
	if m, ok := Lookup(name); ok && m.Recipe != "" {
		by = m.Recipe
	}
	if f.AppleVersion(by) == 0 {
		return nil, "", fmt.Errorf("%s has no recipe for Core AI (model-recipes.yaml models.%s.apple)", name, name)
	}
	return f.ConvertOf(by), by, nil
}

func init() { llm.StaleExport = staleExport }

// staleExport reports whether the Core AI export (its entry.json, when it has one) is of another
// version than its model's recipe (Targets: apple.version). An export of no version (v0-HASH,
// made before the recipes gave one) is of version 1.
func staleExport(export string) bool {
	e, err := shared.ReadEntry(export)
	if err != nil || e.Kind != CoreAIKind || e.Model == "" {
		return false
	}
	m, ok := Lookup(e.Model)
	if !ok {
		return false
	}
	want := m.Targets[cache.CoreAITarget]
	got := max(e.Version, 1)
	return want > 0 && got != want
}

// FillConvert fills, of a Core AI conversion of model name, the whole numbers of ints and the
// strings of strs it leaves out (0, "") from its recipe's apple.convert, by their keys there.
func FillConvert(name string, ints map[string]*int, strs map[string]*string) error {
	c, by, err := AppleConvert(name)
	if err != nil {
		return err
	}
	for key, p := range ints {
		if *p <= 0 {
			if *p, err = c.Int(by, key); err != nil {
				return err
			}
		}
	}
	for key, p := range strs {
		if *p == "" {
			if *p, err = c.String(by, key); err != nil {
				return err
			}
		}
	}
	return nil
}
