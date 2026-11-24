package chat

import (
	"context"
	"fmt"
	"io"
	"slices"

	"github.com/shibukawa/bonsaiai/internal/cache"
	"github.com/shibukawa/bonsaiai/internal/recipe"
)

// LMTool is the script of tools/amd that exports the language models (and d1's decision
// models) for AMD's NPU: a recipe of it gives the model its AMDExport.
const LMTool = "iron_export_lm.py"

// recipeOf is the model whose recipe model name is made by: name when model-recipes.yaml has
// one for it, else that of the model it is like (and so on); "" for none.
func (t *table) recipeOf(name string) string {
	f, err := recipe.Builtin()
	if err != nil {
		return ""
	}
	for seen := []string{}; name != "" && !slices.Contains(seen, name); {
		if _, ok := f.Models[name]; ok {
			return name
		}
		seen = append(seen, name)
		e, ok := t.models[name]
		if !ok {
			return ""
		}
		name = e.Like
	}
	return ""
}

// applyRecipe gives model m what its recipe (of the model named by, recipeOf) says a run needs:
// the arguments of tools/amd/iron_export_lm.py (AMDExport), the model's and each variant's.
func applyRecipe(m *Model, by string) error {
	m.Recipe = by
	if by == "" {
		return nil
	}
	f, err := recipe.Builtin()
	if err != nil {
		return err
	}
	// the version of what it makes for Core AI: the recipe's apple.version, as amd's below
	if v := f.AppleVersion(by); v > 0 {
		if _, ok := m.Targets[cache.CoreAITarget]; !ok {
			if m.Targets == nil {
				m.Targets = map[string]int{}
			}
			m.Targets[cache.CoreAITarget] = v
		}
	}
	r, ok := f.AMDOf(by)
	if !ok {
		return nil
	}
	// the version of what it makes for AMD's NPU: the recipe's (the built-in models.yaml gives
	// none; a file of one's own may, over it)
	if _, ok := m.Targets[cache.AMDTarget]; r.Version > 0 && !ok {
		if m.Targets == nil {
			m.Targets = map[string]int{}
		}
		m.Targets[cache.AMDTarget] = r.Version
	}
	if r.Script != LMTool {
		return nil
	}
	if m.AMDExport, err = lmArgs(r.Args); err != nil {
		return fmt.Errorf("model %s: model-recipes.yaml models.%s.amd: %w", m.Name, by, err)
	}
	for gb, args := range r.Variants {
		v, ok := m.Variants[gb]
		if !ok {
			return fmt.Errorf("model %s: model-recipes.yaml models.%s.amd.variants: %d GB is not a variant of the model (models.yaml variants)", m.Name, by, gb)
		}
		if v.AMDExport, err = lmArgs(args); err != nil {
			return fmt.Errorf("model %s: model-recipes.yaml models.%s.amd.variants.%d: %w", m.Name, by, gb, err)
		}
		m.Variants[gb] = v
	}
	return nil
}

// lmArgs are a recipe's args of iron_export_lm.py as AMDExport holds them: the exporter's model
// and options, without the checkpoint ({in}, after the model) and the export ({out}).
func lmArgs(args []string) ([]string, error) {
	if len(args) < 3 || args[1] != recipe.In || args[2] != recipe.Out {
		return nil, fmt.Errorf("the args of %s are MODEL %s %s OPTIONS", LMTool, recipe.In, recipe.Out)
	}
	out := []string{args[0]}
	for _, a := range args[3:] {
		if a == recipe.In || a == recipe.Out || len(a) > 1 && a[0] == '{' {
			return nil, fmt.Errorf("the options of %s take no placeholder (%s)", LMTool, a)
		}
		out = append(out, a)
	}
	return out, nil
}

// AMDGeneration is the generation of AMD's NPU the exports bonsaiai runs are of (cache.AMDTarget).
var AMDGeneration = recipe.GenerationOf(cache.AMDTarget)

// AMDCommand is how model m's export for AMD's NPU (cache.AMDTarget) is made, as its recipe says
// (model-recipes.yaml amd, Model.Recipe; a variant's args on the machine m runs as): the script
// of tools/amd and its arguments with vars put in (recipe.In: what the caller prepared; the
// values of its kind: recipe.Sizes, ...), ExportOut for the export's folder. want are the
// scripts the caller reads the exports of: another is refused.
func AMDCommand(m Model, vars map[string][]string, want ...string) (recipe.Command, error) {
	if m.Recipe == "" {
		return recipe.Command{}, fmt.Errorf("%s has no recipe for AMD's NPU (model-recipes.yaml models.%s.amd)", m.Name, m.Name)
	}
	f, err := recipe.Builtin()
	if err != nil {
		return recipe.Command{}, err
	}
	c, err := f.AMDCommand(m.Recipe, m.Machine, AMDGeneration, vars)
	if err != nil {
		return c, err
	}
	if len(want) > 0 && !slices.Contains(want, c.Script) {
		return c, fmt.Errorf("model-recipes.yaml models.%s.amd.script: %s, but bonsaiai reads the exports of %v for %s", m.Recipe, c.Script, want, m.Name)
	}
	return c, nil
}

// RunAMDRecipe makes model m's export for AMD's NPU in out as its recipe says (AMDCommand: vars,
// want), with RunIRON.
func RunAMDRecipe(ctx context.Context, python []string, m Model, vars map[string][]string, out string, log io.Writer, want ...string) error {
	c, err := AMDCommand(m, vars, want...)
	if err != nil {
		return err
	}
	f, err := recipe.Builtin()
	if err != nil {
		return err
	}
	return RunIRON(ctx, python, m.Name, c.Script, c.Args, out, log, f.AMDFinish(m.Recipe))
}

// AMDOptions are the arguments of model m's recipe for AMD's NPU but those of recipe.In and
// recipe.Out, with vars put in (AMDCommand): what keys an export of a kind keyed by its
// arguments. nil when the recipe does not give them.
func AMDOptions(m Model, vars map[string][]string, want ...string) []string {
	c, err := AMDCommand(m, vars, want...)
	if err != nil {
		return nil
	}
	return c.Options
}

// InVar is the vars of a recipe of only recipe.In: in.
func InVar(in string) map[string][]string { return map[string][]string{recipe.In: {in}} }
