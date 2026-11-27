// Package engine runs hwnpu's models in this process: imported (import _
// "github.com/shibukawa/hwnpu/ai/engine"), the ai package, modelstore and the packages of models
// (ai/text, ...) prepare and load the models here when they do not go through hwnpud
// (HWNPU_DAEMON=off), as hwnpud's engine, hwnpu-dev and the tests do. Without it a client runs its
// models through hwnpud alone (docs/design/hwnpud.md, "Clients without the engine").
package engine

import (
	_ "github.com/shibukawa/hwnpu/ai/text/engine" // the text models
	_ "github.com/shibukawa/hwnpu/internal/aiengine"
)
