package main

// The models run in this process with HWNPU_DAEMON=off: the speech models until ai/engine imports
// their engine, and ai/engine for the others, which ai/speech brought before
// (docs/design/hwnpud.md, "Clients without the engine", stage 4).
import (
	_ "github.com/shibukawa/hwnpu/ai/engine"
	_ "github.com/shibukawa/hwnpu/ai/speech/engine"
)
