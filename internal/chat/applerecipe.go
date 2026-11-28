package chat

import (
	"cmp"
	"context"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/shibukawa/hwnpu/internal/backend/coreai"
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

// staleExport reports whether the Core AI export or Core ML package (its entry.json, when it has
// one) is of another version than its model's recipe (Targets: apple.version). An export of no version (v0-HASH,
// made before the recipes gave one) is of version 1. export is what the export's mark names:
// the export, or for a chat model the program inside it (model.aimodel, model.parts.json).
func staleExport(export string) bool {
	if !strings.EqualFold(filepath.Ext(export), ".coreai") {
		if dir, ok := coreai.ExportOf(export); ok {
			export = dir
		}
	}
	e, err := shared.ReadEntry(export)
	if err != nil || (e.Kind != CoreAIKind && e.Kind != CoreMLKind) || e.Model == "" {
		return false
	}
	m, ok := Lookup(e.Model)
	if !ok {
		return false
	}
	want := m.Targets[cache.CoreAITarget]
	if e.Kind == CoreMLKind {
		want = m.Targets[cache.CoreMLTarget]
	}
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

// ConvertOf is the apple.convert of the model ctx is for (WithModel, else the one whose
// directory holds path, as the shared layers' keys take it) and its recipe's model, as
// AppleConvert: for a conversion whose code has the model only in its context.
func ConvertOf(ctx context.Context, path string) (recipe.Convert, string, error) {
	m, ok := ctx.Value(modelKey{}).(Model)
	if !ok && path != "" {
		m, _, ok = modelOfDir(path)
	}
	if !ok {
		return nil, "", fmt.Errorf("chat: no model for the recipe of the conversion of %s (WithModel)", path)
	}
	return AppleConvert(cmp.Or(m.Recipe, m.Name))
}

// ConvertString is the string key of the apple.convert of ConvertOf(ctx, path).
func ConvertString(ctx context.Context, path, key string) (string, error) {
	c, by, err := ConvertOf(ctx, path)
	if err != nil {
		return "", err
	}
	return c.String(by, key)
}

// ConvertInt is the whole number key of the apple.convert of ConvertOf(ctx, path).
func ConvertInt(ctx context.Context, path, key string) (int, error) {
	c, by, err := ConvertOf(ctx, path)
	if err != nil {
		return 0, err
	}
	return c.Int(by, key)
}

// ConvertFormat is the weight format (apple.convert's format) a conversion into path for
// backend writes, of the model ctx is for (ConvertOf): the recipe's for Core AI, else code's
// (what other backends write).
func ConvertFormat(ctx context.Context, path, backend, code string) (string, error) {
	if backend != CoreAITarget {
		return code, nil
	}
	return ConvertString(ctx, path, "format")
}
