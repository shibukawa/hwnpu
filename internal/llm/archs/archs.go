// Package archs links in every supported architecture. Its import list is the whitelist of
// models that hwnpu-convert handles; a new architecture is added here.
package archs

import (
	_ "github.com/shibukawa/hwnpu/internal/llm/bert"
	_ "github.com/shibukawa/hwnpu/internal/llm/lfm3"
	_ "github.com/shibukawa/hwnpu/internal/llm/lfm3audio"
	_ "github.com/shibukawa/hwnpu/internal/llm/lfm3gen"
	_ "github.com/shibukawa/hwnpu/internal/llm/lfm3vl"
	_ "github.com/shibukawa/hwnpu/internal/llm/plamo"
)
