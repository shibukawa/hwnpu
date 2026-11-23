package hwnpu

import _ "embed"

// ModelsYAML is models.yaml: the models hwnpu knows and the ones each use recommends
// (internal/chat reads it, docs/model-table.md).
//
//go:embed models.yaml
var ModelsYAML []byte

// RecipesYAML is model-recipes.yaml: how hwnpu-dev makes the models' conversions, what only the
// build needs (internal/recipe reads it, docs/model-recipes.md).
//
//go:embed model-recipes.yaml
var RecipesYAML []byte
