package main

// hwnpu-audio runs the sound models in its process with HWNPU_DAEMON=off, until ai/engine
// imports ai/sound/engine (docs/design/hwnpud.md, "Clients without the engine", stage 4).
import _ "github.com/shibukawa/hwnpu/ai/sound/engine"
