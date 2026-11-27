package main

// hwnpu-chat runs its models in its process with HWNPU_DAEMON=off, until it goes through hwnpud
// alone (docs/design/hwnpud.md, "Clients without the engine", stage 5).
import _ "github.com/shibukawa/hwnpu/ai/engine"
