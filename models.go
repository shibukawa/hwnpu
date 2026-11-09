package hwnpu

import _ "embed"

// ModelsYAML is models.yaml: the models hwnpu knows and the ones each use recommends
// (internal/chat reads it, docs/model-table.md).
//
//go:embed models.yaml
var ModelsYAML []byte
