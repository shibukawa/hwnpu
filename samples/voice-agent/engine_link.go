package main

// The engine lets the sample run its models in its own process (HWNPU_DAEMON=off); without it, an
// app runs them through hwnpud alone (docs/design/hwnpud.md, "Clients without the engine").
import _ "github.com/shibukawa/hwnpu/ai/engine"
