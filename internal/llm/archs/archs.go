// Package archs links in every supported architecture. Its import list is the whitelist of
// models that bonsaiai-dev convert handles; a new architecture is added here.
package archs

import (
	_ "github.com/shibukawa/bonsaiai/internal/llm/bert"
	_ "github.com/shibukawa/bonsaiai/internal/llm/gemma4"
	_ "github.com/shibukawa/bonsaiai/internal/llm/gptoss"
	_ "github.com/shibukawa/bonsaiai/internal/llm/lfm2moe"
	_ "github.com/shibukawa/bonsaiai/internal/llm/lfm3"
	_ "github.com/shibukawa/bonsaiai/internal/llm/lfm3audio"
	_ "github.com/shibukawa/bonsaiai/internal/llm/lfm3gen"
	_ "github.com/shibukawa/bonsaiai/internal/llm/lfm3vl"
	_ "github.com/shibukawa/bonsaiai/internal/llm/ouro"
	_ "github.com/shibukawa/bonsaiai/internal/llm/plamo"
	_ "github.com/shibukawa/bonsaiai/internal/llm/plamo3"
	_ "github.com/shibukawa/bonsaiai/internal/llm/qwen3"
	_ "github.com/shibukawa/bonsaiai/internal/llm/qwen35"
	_ "github.com/shibukawa/bonsaiai/internal/llm/whisper"
)
