package hwnpu

import (
	"github.com/shibukawa/hwnpu/internal/backend"
	"github.com/shibukawa/hwnpu/internal/backend/coreai"
	"github.com/shibukawa/hwnpu/internal/backend/coreml"
)

// Core ML first: it is the default on Apple's Neural Engine. Core AI (macOS 27, the same
// .mlpackage programs lowered to Core AI's) is chosen by name (WithBackend("coreai"),
// HWNPU_BACKEND=coreai).
func init() {
	backend.Register(coreml.Provider{})
	backend.Register(coreai.Provider{})
}
