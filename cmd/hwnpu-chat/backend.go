package main

import (
	"context"
	"os"
	"runtime"
	"strings"

	"github.com/shibukawa/hwnpu/internal/chatcmd"
)

// The names of the backends hwnpu-chat runs on, and the variable that names one (internal/llm's
// BackendCoreML, BackendCoreAI and BackendEnv).
const (
	backendEnv    = "HWNPU_BACKEND"
	backendCoreML = "coreml"
	backendCoreAI = "coreai"
)

// chatDevice is the device backend hwnpu-chat converts for and runs on for the one a command names
// (name, else HWNPU_BACKEND, else the NPU found): on a Mac Core AI's, whatever names Apple's Neural
// Engine (coreml included, as HWNPU_BACKEND may for the other commands); Core ML is not used. Its
// chat and embedding models are converted for Core AI from their checkpoints (docs/design/apple-coreai.md).
func chatDevice(name string) string {
	if name == "" {
		name = strings.TrimSpace(os.Getenv(backendEnv))
	}
	if runtime.GOOS == "darwin" {
		switch name {
		case "", backendCoreML, backendCoreAI:
			return backendCoreAI
		}
	}
	return name
}

// resolveBackend is chatDevice(name), checked on a Mac by hwnpud, where the models run: Core AI
// needs macOS 27.
func resolveBackend(ctx context.Context, name string) (string, error) {
	d := chatDevice(name)
	if d == backendCoreAI {
		if err := chatcmd.Device(ctx, d); err != nil {
			return "", err
		}
	}
	return d, nil
}
